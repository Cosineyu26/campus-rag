import asyncio
import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .config import SiteConfig
from .models import RawPage


class FetchError(Exception):
    pass


@dataclass
class PageData:
    final_url: str
    html: str
    links: list[str]


async def fetch_page_default(url: str, client=None) -> PageData:
    """crawl4ai 封装层：crawl4ai 的 API 变动只允许改这个函数。"""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=CrawlerRunConfig())
    if not result or not result.success:
        raise FetchError(f"fetch failed: {url}")
    # crawl4ai>=0.9 的 result.links 形如 {"internal": [{"href","text"},...], "external": [...]}
    raw_links = (result.links or {}).get("internal", []) + (result.links or {}).get("external", [])
    links = [str(l.get("href", "")) for l in raw_links if isinstance(l, dict) and l.get("href")]
    return PageData(final_url=str(result.url or url), html=result.html or "", links=links)


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_DATE_RE = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")


def _extract_title(html: str) -> str:
    m = _TITLE_RE.search(html)
    return m.group(1).strip() if m else ""


def _extract_date(html: str) -> date | None:
    m = _DATE_RE.search(html)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _load_robots(site: SiteConfig) -> RobotFileParser | None:
    rp = RobotFileParser()
    rp.set_url(f"https://{site.allowed_domain}/robots.txt")
    try:
        rp.read()
        return rp
    except Exception:
        return None  # robots.txt 拉取失败则放行（本系统为低频礼貌爬取）


class Crawler:
    def __init__(self, site: SiteConfig, data_dir: Path,
                 fetch_page=None, client=None, robots=None):
        # robots 显式传入才启用（默认为 None 不检查）——由 sync 用 _load_robots 加载，
        # 避免测试环境隐式发起网络请求
        self.site = site
        self.site_dir = data_dir / "raw" / site.category
        self.fetch_page = fetch_page or fetch_page_default
        self.client = client
        self.robots = robots
        self.failures: list[str] = []  # 本次爬取失败的页面 URL（供 sync 判断站点爬取健康度）

    async def crawl(self) -> list[RawPage]:
        import httpx

        if self.client is None:
            self.client = httpx.AsyncClient(headers={"User-Agent": "campus-rag/0.1 (+educational)"})
        self.site_dir.mkdir(parents=True, exist_ok=True)
        pages: list[RawPage] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((u, 0) for u in self.site.entry_urls)

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > self.site.max_depth:
                continue
            if urlparse(url).netloc != self.site.allowed_domain:
                continue
            if self.robots and not self.robots.can_fetch("*", url):
                continue
            visited.add(url)
            try:
                page = await self.fetch_page(url, self.client)
            except Exception as e:  # 单页失败不中断整轮，但记录失败供 vanish 阶段排除误标
                print(f"[crawler] skip {url}: {e}")
                self.failures.append(url)
                continue

            pdf_files = await self._download_pdfs(page)
            html_path, title, published = self._save(url, page.html, pdf_files)
            pages.append(RawPage(url=url, category=self.site.category,
                                 html_path=html_path, title=title,
                                 published_at=published, pdf_files=pdf_files))

            for link in page.links:
                nxt = urljoin(url, link)
                if urlparse(nxt).netloc == self.site.allowed_domain and nxt not in visited:
                    queue.append((nxt, depth + 1))
            await asyncio.sleep(self.site.delay_seconds)

        aclose = getattr(self.client, "aclose", None)
        if aclose is not None:
            await aclose()
        return pages

    async def _download_pdfs(self, page: PageData) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        pdf_dir = self.site_dir / "pdfs"
        for link in page.links:
            if not link.lower().endswith(".pdf"):
                continue
            pdf_url = urljoin(page.final_url, link)
            try:
                resp = await self.client.get(pdf_url)
                if resp.status_code != 200:
                    continue
                path = pdf_dir / f"{_url_hash(pdf_url)}.pdf"
                pdf_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                out.append((pdf_url, path))
            except Exception as e:
                print(f"[crawler] pdf download failed {pdf_url}: {e}")
        return out

    def _save(self, url: str, html: str, pdf_files: list[tuple[str, Path]]):
        stem = _url_hash(url)
        html_path = self.site_dir / f"{stem}.html"
        html_path.write_text(html, encoding="utf-8")
        meta = {
            "url": url, "category": self.site.category,
            "title": _extract_title(html),
            "published_at": _extract_date(html).isoformat() if _extract_date(html) else None,
            "pdf_files": [str(p) for _, p in pdf_files],
        }
        Path(str(html_path) + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return html_path, meta["title"], _extract_date(html)
