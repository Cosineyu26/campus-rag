# 数据管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现校园 RAG 系统的数据管线：爬取学校官网 → 解析清洗 → 条款级分块 → 嵌入 → 写入 Qdrant，带每周增量同步（新增/更新/过期标记）。

**Architecture:** 独立 Python 包 `pipeline/`，纯 CPU 运行（嵌入走 HTTP 调用 `embedding_service/`，跑在 V100 上）。各环节（爬虫/解析/分块/嵌入/入库/注册表）为独立模块，通过 dataclass（`RawPage`/`Document`/`Chunk`/`Embedding`）衔接，可单测、可替换。增量同步由 `sync.run_full_sync()` 编排，URL+内容哈希比对，过期只标记不删除。

**Tech Stack:** Python 3.11、crawl4ai、trafilatura、PyMuPDF、Qdrant（client）、SQLAlchemy 2.0 + MySQL 8、sentence-transformers（BGE-M3，独立 FastAPI 服务）、pytest + respx。

**Spec:** `2026-09-02-campus-rag-design.md`（§4 数据管线、§8 数据模型、§14 目录结构）

## Global Constraints

- 本计划只实现**数据管线子系统**（pipeline/ + embedding_service/ + MySQL/Qdrant 的 compose 配置）；后端、检索链、前端、评测各自后续单独出计划。
- Python 3.11+；管线包名 `pipeline`，模块导入统一 `from pipeline.xxx import ...`。
- V100 不支持 bf16，模型加载一律 fp16 或默认精度；不要下载 AWQ/GPTQ 格式模型。
- MySQL 必须 `utf8mb4`；中文内容为主。
- Qdrant 集合名 `campus_kb`；稠密向量 1024 维 + 稀疏向量双命名向量；点 ID = `{content_hash}-{index:04d}`。
- 内容哈希 = SHA-256(text)（64 位十六进制）。
- 分块：中文按字符数近似（500 字 ≈ 500~800 token），目标 500 字符、重叠 60 字符；教务条款优先按"第X条/一、"边界切。
- 增量语义：内容变化 → 删旧块入新块；页面消失 → 注册表与 Qdrant 均标记 `stale`，**不删除**。
- 爬取礼貌性：遵守 robots.txt（拉取失败则放行）、请求间隔 2.5 秒、同域名、深度 ≤3。
- `data/`、`models/` 目录已被 .gitignore 忽略，不得入库。
- 测试用 SQLite 替代 MySQL（SQLAlchemy 抽象保证兼容），单测不依赖真实模型/外网。
- crawl4ai 的 API 变动风险：所有 crawl4ai 调用收敛在 `pipeline/crawler.py` 的 `fetch_page_default()` 内，实现时若与最新版 API 不符，以官方文档为准只改该函数。

---

### Task 1: 项目骨架与基础设施（compose + 包 + 配置 + 数据模型）

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `pipeline/pyproject.toml`
- Create: `pipeline/config.yaml.example`
- Create: `pipeline/pipeline/__init__.py`
- Create: `pipeline/pipeline/config.py`
- Create: `pipeline/pipeline/models.py`
- Create: `pipeline/tests/__init__.py`

**Interfaces:**
- Consumes: 无（首任务）
- Produces: `load_config(path) -> PipelineConfig`、`Crawler` 所需 `SiteConfig`；dataclass `RawPage/Document/Chunk/Embedding`（后续所有任务引用）；compose 提供 MySQL(`localhost:3306`, 库 `campus_rag`, 用户 `campus`) 与 Qdrant(`localhost:6333`)。

- [ ] **Step 1: 写 docker-compose.yml**

```yaml
services:
  mysql:
    image: mysql:8
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:-root}
      MYSQL_DATABASE: campus_rag
      MYSQL_USER: campus
      MYSQL_PASSWORD: ${MYSQL_PASSWORD:-campus}
    command: --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 5s
      timeout: 3s
      retries: 10

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  mysql_data:
  qdrant_data:
```

- [ ] **Step 2: 写 .env.example**

```bash
MYSQL_ROOT_PASSWORD=root
MYSQL_PASSWORD=campus
# 复制为 .env 后按需修改
```

- [ ] **Step 3: 写 pipeline/pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "campus-rag-pipeline"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "crawl4ai>=0.7",
    "trafilatura>=1.8",
    "pymupdf>=1.24",
    "qdrant-client>=1.10",
    "sqlalchemy>=2.0",
    "pymysql>=1.1",
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "respx>=0.21",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: 写 pipeline/pipeline/models.py（后续所有模块的公共类型）**

```python
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

Embedding = tuple[list[float], dict[int, float]]  # (dense, sparse{token_id: weight})


@dataclass
class RawPage:
    """爬虫落盘的原始页面：HTML 文件 + 元数据 + PDF 附件清单。"""
    url: str
    category: str
    html_path: Path
    title: str = ""
    published_at: Optional[date] = None
    pdf_files: list[tuple[str, Path]] = field(default_factory=list)  # [(原始URL, 本地路径)]
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class Document:
    """一篇可入库文档（HTML 正文或 PDF 全文）。"""
    url: str
    category: str
    title: str
    text: str
    content_hash: str          # SHA-256(text)，即 doc_id
    published_at: Optional[date] = None
    effective_date: Optional[date] = None
    source_path: Optional[Path] = None


@dataclass
class Chunk:
    """分块结果，chunk_id 即 Qdrant 点 ID。"""
    chunk_id: str              # f"{content_hash}-{index:04d}"
    doc_id: str                # content_hash
    text: str
    chunk_index: int
    title: str
    url: str
    category: str
    effective_date: Optional[date]
    crawled_at: datetime
```

- [ ] **Step 5: 写 pipeline/pipeline/config.py**

```python
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import RawPage  # noqa: F401  确保包结构可导入


@dataclass
class SiteConfig:
    name: str
    category: str
    entry_urls: list[str]
    allowed_domain: str
    max_depth: int = 3
    delay_seconds: float = 2.5


@dataclass
class PipelineConfig:
    sites: list[SiteConfig]
    data_dir: Path = Path("data")
    chunk_size: int = 500
    chunk_overlap: int = 60
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = "campus_kb"
    mysql_url: str = field(default_factory=lambda: os.getenv(
        "MYSQL_URL", "mysql+pymysql://campus:campus@localhost:3306/campus_rag?charset=utf8mb4"))
    embedding_url: str = field(default_factory=lambda: os.getenv("EMBEDDING_URL", "http://localhost:8001"))


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sites = [SiteConfig(**s) for s in raw["sites"]]
    opts = dict(raw.get("options", {}))
    if "data_dir" in opts:  # dataclass 不做类型转换，YAML 里的路径字符串必须显式转 Path
        opts["data_dir"] = Path(opts["data_dir"])
    return PipelineConfig(sites=sites, **opts)
```

- [ ] **Step 6: 写 pipeline/config.yaml.example**

```yaml
sites:
  - name: 教务处规章制度
    category: 教务政策
    entry_urls:
      - https://jwc.example.edu.cn/rule/
    allowed_domain: jwc.example.edu.cn
    max_depth: 3
    delay_seconds: 2.5
  - name: 新生指南
    category: 新生问答
    entry_urls:
      - https://zs.example.edu.cn/newstu/
    allowed_domain: zs.example.edu.cn

options:
  data_dir: data
  chunk_size: 500
  chunk_overlap: 60
```

- [ ] **Step 7: 跑冒烟验证**

Run:
```bash
cd pipeline && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "from pipeline.config import load_config, PipelineConfig; c = load_config('config.yaml.example'); print(len(c.sites), c.qdrant_collection)"
# 以下 docker 冒烟需要本机装有 Docker。若本机无 Docker（如 Windows 开发机），
# 跳过 docker 两条命令不阻塞——Task 10 在部署机上统一验证 MySQL/Qdrant。
docker compose -f ../docker-compose.yml up -d
docker compose -f ../docker-compose.yml ps
```
Expected: 输出 `2 campus_kb`；有 Docker 时 `ps` 显示 mysql、qdrant 两容器 running（healthy）。

- [ ] **Step 8: 提交**

```bash
git add docker-compose.yml .env.example pipeline/
git commit -m "chore: 管线项目骨架与基础设施（compose/配置/数据模型）"
```

---

### Task 2: 爬虫（定向 BFS + PDF 下载 + 落盘）

**Files:**
- Create: `pipeline/pipeline/crawler.py`
- Test: `pipeline/tests/test_crawler.py`

**Interfaces:**
- Consumes: `SiteConfig`（Task 1）
- Produces: `Crawler(site, data_dir, fetch_page=None, client=None, robots=None)`；`async crawl() -> list[RawPage]`；`PageData(final_url, html, links)`；`FetchError`；`fetch_page_default(url, client)`（crawl4ai 封装层）。`client` 仅用于下载 PDF（httpx AsyncClient 接口：`await client.get(url)` 返回带 `status_code`/`content` 的对象）。

- [ ] **Step 1: 写失败测试**

```python
import asyncio
import json
from pathlib import Path

from pipeline.crawler import Crawler, FetchError, PageData
from pipeline.config import SiteConfig


class FakeFetcher:
    """按 url 返回固定页面；未收录的 url 抛 FetchError。"""
    def __init__(self, pages: dict[str, PageData]):
        self.pages = pages
        self.calls: list[str] = []

    async def __call__(self, url, client=None):
        self.calls.append(url)
        if url not in self.pages:
            raise FetchError(f"404 {url}")
        return self.pages[url]


class FakeClient:
    """仅响应 PDF 下载请求。"""
    def __init__(self):
        self.downloaded: list[str] = []

    async def get(self, url):
        self.downloaded.append(url)
        return type("Resp", (), {"status_code": 200, "content": b"%PDF-1.4 fake"})()


def make_site(tmp_path: Path):
    return SiteConfig(name="测试栏目", category="测试",
                      entry_urls=["https://school.edu.cn/a"],
                      allowed_domain="school.edu.cn",
                      max_depth=2, delay_seconds=0.01)


HTML = "<html><head><title>测试页</title></head><body>正文</body></html>"


def test_crawl_collects_pages_and_pdfs(tmp_path):
    fetcher = FakeFetcher({
        "https://school.edu.cn/a": PageData("https://school.edu.cn/a", HTML,
                                            ["/b", "https://other.edu.cn/x", "/f.pdf"]),
        "https://school.edu.cn/b": PageData("https://school.edu.cn/b", HTML, []),
    })
    client = FakeClient()
    crawler = Crawler(make_site(tmp_path), tmp_path, fetch_page=fetcher, client=client)
    pages = asyncio.run(crawler.crawl())
    assert {p.url for p in pages} == {"https://school.edu.cn/a", "https://school.edu.cn/b"}
    assert "https://other.edu.cn/x" not in fetcher.calls  # 跨域链接不跟随
    assert pages[0].pdf_files and pages[0].pdf_files[0][0].endswith("f.pdf")
    assert pages[0].pdf_files[0][1].exists()               # PDF 已落盘
    assert pages[0].html_path.exists()                     # HTML 已落盘
    assert pages[0].title == "测试页"
    meta = json.loads(Path(str(pages[0].html_path) + ".meta.json").read_text("utf-8"))
    assert meta["category"] == "测试"


def test_crawl_respects_depth_and_visited(tmp_path):
    fetcher = FakeFetcher({
        "https://school.edu.cn/a": PageData("https://school.edu.cn/a", HTML, ["/a"]),
    })
    crawler = Crawler(make_site(tmp_path), tmp_path, fetch_page=fetcher)
    asyncio.run(crawler.crawl())
    assert fetcher.calls.count("https://school.edu.cn/a") == 1  # 自环只爬一次
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_crawler.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.crawler'`

- [ ] **Step 3: 实现 pipeline/pipeline/crawler.py**

```python
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
            except Exception as e:  # 单页失败不中断整轮
                print(f"[crawler] skip {url}: {e}")
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/test_crawler.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline/crawler.py pipeline/tests/test_crawler.py
git commit -m "feat: 定向爬虫（BFS/PDF 下载/落盘/robots）"
```

---

### Task 3: 解析与清洗（HTML/PDF → Document + 元数据提取）

**Files:**
- Create: `pipeline/pipeline/parser.py`
- Test: `pipeline/tests/test_parser.py`

**Interfaces:**
- Consumes: `RawPage`（Task 1/2）
- Produces: `parse(raw: RawPage) -> list[Document]`（HTML 正文 + 每份 PDF 各一篇；空文本自动丢弃）；`sha256(text) -> str`；`extract_effective_date(text) -> date | None`；`clean_text(text) -> str`。Document.url 对 PDF 取原始 PDF 链接（引用可跳转）。

- [ ] **Step 1: 写失败测试**

```python
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF，仅用于构造 PDF fixture

from pipeline.models import RawPage
from pipeline.parser import clean_text, extract_effective_date, parse, sha256


def make_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


HTML = """<html><head><title>学生公寓管理规定</title></head>
<body><nav>导航噪声</nav>
<article><h1>学生公寓管理规定</h1>
<p>公寓实行晚23:00锁门制度。本规定自2025年9月1日起施行。</p></article>
<footer>版权所有</footer></body></html>"""


def test_parse_html_extracts_body_and_metadata(tmp_path):
    html = tmp_path / "p.html"
    html.write_text(HTML, encoding="utf-8")
    raw = RawPage(url="https://school.edu.cn/p", category="教务政策", html_path=html, title="")
    docs = parse(raw)
    assert len(docs) == 1
    d = docs[0]
    assert "导航噪声" not in d.text and "版权所有" not in d.text  # 正文提取剥离噪声
    assert "23:00锁门" in d.text
    assert d.effective_date == date(2025, 9, 1)
    assert d.content_hash == sha256(d.text)


def test_parse_pdf(tmp_path):
    html = tmp_path / "p.html"
    html.write_text(HTML, encoding="utf-8")
    pdf = tmp_path / "rule.pdf"
    make_pdf(pdf, "学生选课管理办法。本规定自2024年3月1日起执行。")
    raw = RawPage(url="https://school.edu.cn/p", category="教务政策", html_path=html,
                  pdf_files=[("https://school.edu.cn/r.pdf", pdf)])
    docs = parse(raw)
    assert len(docs) == 2
    pdf_doc = docs[1]
    assert pdf_doc.url == "https://school.edu.cn/r.pdf"
    assert pdf_doc.effective_date == date(2024, 3, 1)
    assert "选课管理办法" in pdf_doc.text


def test_parse_skips_empty(tmp_path):
    html = tmp_path / "empty.html"
    html.write_text("<html><body></body></html>", encoding="utf-8")
    raw = RawPage(url="https://school.edu.cn/e", category="教务政策", html_path=html)
    assert parse(raw) == []


def test_clean_text_collapses_blank_lines():
    assert clean_text("第一行\n\n  \n第二行\n") == "第一行\n第二行"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_parser.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.parser'`

- [ ] **Step 3: 实现 pipeline/pipeline/parser.py**

```python
import hashlib
import re
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
import trafilatura

from .models import Document, RawPage

_EFFECTIVE_PATTERNS = [
    re.compile(r"自(\d{4})年(\d{1,2})月(\d{1,2})日起?(?:施行|执行|实施)"),
    re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(?:施行|执行|生效)"),
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """去除行首尾空白并合并连续空行。"""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_effective_date(text: str) -> date | None:
    for pat in _EFFECTIVE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _parse_html(raw: RawPage) -> Document:
    html = Path(raw.html_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_text(trafilatura.extract(html, include_comments=False) or "")
    return Document(url=raw.url, category=raw.category, title=raw.title or "",
                    text=text, content_hash=sha256(text),
                    published_at=raw.published_at,
                    effective_date=extract_effective_date(text),
                    source_path=raw.html_path)


def _parse_pdf(pdf_url: str, pdf_path: Path, raw: RawPage) -> Document:
    with fitz.open(pdf_path) as doc:
        text = clean_text("\n".join(page.get_text() for page in doc))
    return Document(url=pdf_url, category=raw.category, title=raw.title or "",
                    text=text, content_hash=sha256(text),
                    published_at=raw.published_at,
                    effective_date=extract_effective_date(text),
                    source_path=pdf_path)


def parse(raw: RawPage) -> list[Document]:
    """一个 RawPage → 若干 Document（HTML 正文 + 每份 PDF 各一篇），空文本丢弃。"""
    docs = [_parse_html(raw)]
    docs += [_parse_pdf(url, path, raw) for url, path in raw.pdf_files]
    return [d for d in docs if d.text]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/test_parser.py -v`
Expected: PASS（4 tests）

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline/parser.py pipeline/tests/test_parser.py
git commit -m "feat: 解析清洗（HTML/PDF → Document + 生效日期提取）"
```

---

### Task 4: 分块（条款级 + 定长兜底）

**Files:**
- Create: `pipeline/pipeline/chunker.py`
- Test: `pipeline/tests/test_chunker.py`

**Interfaces:**
- Consumes: `Document`（Task 1/3）
- Produces: `chunk(doc, size=500, overlap=60) -> list[Chunk]`；`split_by_clauses(text) -> list[str]`；`split_by_size(text, size, overlap) -> list[str]`。Chunk 的 `chunk_id = f"{doc.content_hash}-{index:04d}"`，元数据全部继承自 Document。

- [ ] **Step 1: 写失败测试**

```python
from datetime import date

from pipeline.chunker import (chunk, split_by_clauses, split_by_headings,
                              split_by_size)
from pipeline.models import Document


def doc(text: str) -> Document:
    from pipeline.parser import sha256
    return Document(url="https://school.edu.cn/r", category="教务政策", title="某规定",
                    text=text, content_hash=sha256(text),
                    effective_date=date(2025, 9, 1))


def test_split_by_headings_splits_on_markdown():
    text = "# 第一章\n第一条 内容。\n# 第二章\n第二条 内容。"
    parts = split_by_headings(text)
    assert len(parts) == 2
    assert parts[0].startswith("# 第一章") and parts[1].startswith("# 第二章")


def test_split_by_clauses_uses_article_boundaries():
    text = "第一章 总则\n第一条 学生应当遵守纪律。\n第二条 考试作弊记零分。\n第三条 请假需审批。"
    parts = split_by_clauses(text)
    assert len(parts) == 3
    assert "第一条" in parts[0] and "第二条" in parts[1] and "第三条" in parts[2]


def test_split_by_clauses_single_when_no_clauses():
    text = "这是一段没有任何条款标记的普通文本。"
    assert split_by_clauses(text) == [text]


def test_split_by_size_overlaps():
    text = "字" * 1000
    parts = split_by_size(text, size=400, overlap=60)
    assert len(parts) == 3
    assert parts[0][-60:] == parts[1][:60]  # 重叠正确


def test_chunk_generates_deterministic_ids_and_metadata():
    d = doc("第一条 内容甲。" * 30)
    chunks = chunk(d, size=100, overlap=20)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"{d.content_hash}-{i:04d}"
        assert c.doc_id == d.content_hash
        assert c.url == d.url and c.category == d.category
        assert c.effective_date == d.effective_date
        assert c.chunk_index == i
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_chunker.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.chunker'`

- [ ] **Step 3: 实现 pipeline/pipeline/chunker.py**

```python
import re
from datetime import datetime

from .models import Chunk, Document

_CLAUSE_RE = re.compile(r"(?:第[一二三四五六七八九十百零〇\d]+条|[一二三四五六七八九十]+、)")
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def split_by_headings(text: str) -> list[str]:
    """按 Markdown 标题行切分（trafilatura 保留的标题结构）。无标题时返回原文单段。"""
    matches = list(_HEADING_LINE_RE.finditer(text))
    if not matches:
        return [text]
    parts, last = [], 0
    for m in matches:
        if m.start() > last:
            parts.append(text[last:m.start()])
        last = m.start()
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def split_by_clauses(text: str) -> list[str]:
    """按条款级边界切分（'第一条'、'一、'）。无条款标记时返回原文单段。"""
    matches = list(_CLAUSE_RE.finditer(text))
    if len(matches) < 2:
        return [text]
    parts, last = [], matches[0].start()
    for m in matches[1:]:
        parts.append(text[last:m.start()])
        last = m.start()
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def split_by_size(text: str, size: int, overlap: int) -> list[str]:
    """定长切分（中文按字符数近似 token 数），相邻块重叠 `overlap` 字符。"""
    if not text.strip():
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk(doc: Document, size: int = 500, overlap: int = 60) -> list[Chunk]:
    """三级切分：标题结构 → 条款级 → 定长兜底；chunk_id 确定性生成（幂等入库的基础）。"""
    out: list[Chunk] = []
    crawled_at = datetime.now()
    for section in split_by_headings(doc.text):
        for piece in split_by_clauses(section):
            for text in split_by_size(piece, size, overlap):
                idx = len(out)
                out.append(Chunk(
                    chunk_id=f"{doc.content_hash}-{idx:04d}",
                    doc_id=doc.content_hash,
                    text=text, chunk_index=idx, title=doc.title, url=doc.url,
                    category=doc.category, effective_date=doc.effective_date,
                    crawled_at=crawled_at,
                ))
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/test_chunker.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline/chunker.py pipeline/tests/test_chunker.py
git commit -m "feat: 条款级分块（定长兜底、确定性 chunk_id）"
```

---

### Task 5: 嵌入客户端（HTTP 契约先行）

**Files:**
- Create: `pipeline/pipeline/embedder.py`
- Test: `pipeline/tests/test_embedder.py`

**Interfaces:**
- Consumes: `Embedding = tuple[list[float], dict[int, float]]`（Task 1）
- Produces: `EmbedderClient(base_url)`；`embed(texts: list[str]) -> list[Embedding]`。调用约定（Task 6 实现）：`POST {base}/embed`，请求体 `{"texts": [...]}`，响应 `{"dense": [[...]], "sparse": [{"token_id_str": weight, ...}]}`。稀疏权重键在传输层为字符串（JSON 限制），客户端转回 int。

- [ ] **Step 1: 写失败测试**

```python
import httpx
import respx

from pipeline.embedder import EmbedderClient


def test_embed_maps_response(respx_mock):
    route = respx_mock.post("http://embed:8001/embed").mock(return_value=httpx.Response(200, json={
        "dense": [[0.1, 0.2], [0.3, 0.4]],
        "sparse": [{"7": 1.5, "99": 0.8}, {"7": 0.9}],
    }))
    client = EmbedderClient("http://embed:8001")
    embs = client.embed(["问题一", "问题二"])
    assert route.called
    assert embs[0] == ([0.1, 0.2], {7: 1.5, 99: 0.8})
    assert embs[1] == ([0.3, 0.4], {7: 0.9})


def test_embed_propagates_http_error(respx_mock):
    respx_mock.post("http://embed:8001/embed").mock(return_value=httpx.Response(500))
    client = EmbedderClient("http://embed:8001")
    try:
        client.embed(["x"])
        assert False, "should raise"
    except httpx.HTTPStatusError:
        pass
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_embedder.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.embedder'`

- [ ] **Step 3: 实现 pipeline/pipeline/embedder.py**

```python
import httpx

from .models import Embedding


class EmbedderClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[Embedding]:
        resp = self._client.post(f"{self.base_url}/embed", json={"texts": texts})
        resp.raise_for_status()
        data = resp.json()
        return [
            (dense, {int(k): v for k, v in sparse.items()})
            for dense, sparse in zip(data["dense"], data["sparse"])
        ]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/test_embedder.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline/embedder.py pipeline/tests/test_embedder.py
git commit -m "feat: 嵌入服务 HTTP 客户端（dense+sparse 契约）"
```

---

### Task 6: 嵌入服务（FastAPI + BGE-M3，V100 容器）

**Files:**
- Create: `embedding_service/requirements.txt`
- Create: `embedding_service/app.py`
- Create: `embedding_service/Dockerfile`
- Create: `embedding_service/conftest.py`（空文件；让 pytest 把 embedding_service/ 加入 sys.path，`import app` 才能成功）
- Test: `embedding_service/tests/test_app.py`

**Interfaces:**
- Consumes: Task 5 定义的 HTTP 契约
- Produces: `POST /embed`：`{"texts": [...]}` → `{"dense": [[...]], "sparse": [{"str_id": float}]}`；`GET /healthz` → `{"status": "ok"}`。模型加载函数 `get_model()` 可被测试 monkeypatch。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeModel:
    def encode(self, texts, batch_size=None, return_dense=None, return_sparse=None,
               normalize_embeddings=None):  # normalize_embeddings: 与 app.py 调用保持一致
        return {
            "dense_vecs": np.array([[0.1] * 3] * len(texts)),
            "lexical_weights": [{0: 1.0, 5: 0.5}] * len(texts),
        }


@pytest.fixture()
def client(monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_model", FakeModel())
    return TestClient(app_module.app)


def test_embed_contract(client):
    resp = client.post("/embed", json={"texts": ["你好", "世界"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["dense"]) == 2
    assert data["sparse"] == [{"0": 1.0, "5": 0.5}, {"0": 1.0, "5": 0.5}]


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
```

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
cd embedding_service && python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/pytest tests/test_app.py -v
```
Expected: FAIL，`ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: 实现 embedding_service/requirements.txt**

```txt
fastapi>=0.115
uvicorn>=0.30
sentence-transformers>=3.0,<6.0   # <6.0: 6.x 移除了 return_dense/return_sparse API（app.py 依赖）
numpy>=1.26
httpx>=0.27
```

- [ ] **Step 4: 实现 embedding_service/app.py**

```python
import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.getenv("BGE_M3_DIR", "BAAI/bge-m3")

app = FastAPI(title="embedding-service")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_DIR)
    return _model


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    model = get_model()
    enc = model.encode(
        req.texts, batch_size=32,
        return_dense=True, return_sparse=True, normalize_embeddings=True,
    )
    dense = enc["dense_vecs"].tolist()
    sparse = [{str(k): float(v) for k, v in row.items()} for row in enc["lexical_weights"]]
    return {"dense": dense, "sparse": sparse}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
```

- [ ] **Step 5: 实现 embedding_service/Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd embedding_service && .venv/bin/pytest tests/test_app.py -v`
Expected: PASS（2 tests）

- [ ] **Step 7: 提交**

```bash
git add embedding_service/
git commit -m "feat: BGE-M3 嵌入服务（FastAPI，dense+sparse 双输出）"
```

---

### Task 7: 文档注册表（SQLAlchemy + MySQL，测试走 SQLite）

**Files:**
- Create: `pipeline/pipeline/registry.py`
- Test: `pipeline/tests/test_registry.py`

**Interfaces:**
- Consumes: 无（`RegistryRecord` 为独立 dataclass）
- Produces: `DocumentRegistry(db_url)`；`list_all() -> dict[str, RegistryRecord]`（url → 记录，含 stale）；`upsert(rec: RegistryRecord)`；`mark_stale(urls: list[str])`。生产用 `mysql+pymysql://...`，测试用 `sqlite:///:memory:`。

- [ ] **Step 1: 写失败测试**

```python
from datetime import date

from pipeline.registry import DocumentRegistry, RegistryRecord


def rec(url, h="abc", status="active"):
    return RegistryRecord(url=url, title="t", category="c",
                          content_hash=h, status=status,
                          effective_date=date(2025, 9, 1))


def test_upsert_and_list(tmp_path):
    db = tmp_path / "t.db"
    reg = DocumentRegistry(f"sqlite:///{db.as_posix()}")  # as_posix: Windows 路径兼容
    reg.upsert(rec("https://x/a"))
    reg.upsert(rec("https://x/a", h="newhash"))  # 重复 upsert = 更新
    reg.upsert(rec("https://x/b"))
    rows = reg.list_all()
    assert set(rows) == {"https://x/a", "https://x/b"}
    assert rows["https://x/a"].content_hash == "newhash"


def test_mark_stale(tmp_path):
    db = tmp_path / "t.db"
    reg = DocumentRegistry(f"sqlite:///{db.as_posix()}")
    reg.upsert(rec("https://x/a"))
    reg.upsert(rec("https://x/b"))
    reg.mark_stale(["https://x/a"])
    rows = reg.list_all()
    assert rows["https://x/a"].status == "stale"
    assert rows["https://x/b"].status == "active"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_registry.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.registry'`

- [ ] **Step 3: 实现 pipeline/pipeline/registry.py**

```python
from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import Date, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # String(768): utf8mb4 下 768*4=3072 字节 = InnoDB 唯一索引键上限（1024 会 ERROR 1071）
    url: Mapped[str] = mapped_column(String(768), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="active")
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    published_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_crawled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


@dataclass
class RegistryRecord:
    url: str
    title: str
    category: str
    content_hash: str
    status: str = "active"
    effective_date: date | None = None
    published_at: date | None = None


def _to_record(row: DocumentRow) -> RegistryRecord:
    return RegistryRecord(url=row.url, title=row.title, category=row.category,
                          content_hash=row.content_hash, status=row.status,
                          effective_date=row.effective_date, published_at=row.published_at)


class DocumentRegistry:
    """文档注册表：增量比对的事实来源（测试用 SQLite，生产用 MySQL，同一套 ORM）。"""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)

    def list_all(self) -> dict[str, RegistryRecord]:
        with Session(self.engine) as s:
            rows = s.query(DocumentRow).all()
        return {r.url: _to_record(r) for r in rows}

    def upsert(self, rec: RegistryRecord):
        with Session(self.engine) as s:
            row = s.query(DocumentRow).filter_by(url=rec.url).one_or_none()
            if row is None:
                row = DocumentRow(url=rec.url)
                s.add(row)
            row.title = rec.title
            row.category = rec.category
            row.content_hash = rec.content_hash
            row.status = rec.status
            row.effective_date = rec.effective_date
            row.published_at = rec.published_at
            row.last_crawled_at = datetime.now()
            row.last_ingested_at = datetime.now()
            s.commit()

    def mark_stale(self, urls: list[str]):
        if not urls:
            return
        with Session(self.engine) as s:
            s.query(DocumentRow).filter(DocumentRow.url.in_(urls)).update(
                {"status": "stale"}, synchronize_session=False)
            s.commit()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/test_registry.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline/registry.py pipeline/tests/test_registry.py
git commit -m "feat: 文档注册表（SQLAlchemy，增量比对事实来源）"
```

---

### Task 8: 入库器（Qdrant：建集合、upsert、按 URL 删除/标记 stale）

**Files:**
- Create: `pipeline/pipeline/ingester.py`
- Test: `pipeline/tests/test_ingester.py`

**Interfaces:**
- Consumes: `Chunk`、`Embedding`（Task 1）
- Produces: `ensure_collection(client, name)`；`upsert_chunks(client, name, chunks, embeddings)`；`delete_by_url(client, name, url)`；`mark_stale_by_url(client, name, url)`。`client` 为 qdrant_client 的 `QdrantClient`（测试注入 Fake）。

- [ ] **Step 1: 写失败测试**

```python
import uuid
from datetime import date, datetime

from pipeline.ingester import (delete_by_url, ensure_collection,
                               mark_stale_by_url, upsert_chunks)
from pipeline.models import Chunk


class FakeQdrant:
    def __init__(self):
        self.collections: dict[str, dict] = {}
        self.upserts: dict[str, list] = {}
        self.deletes: list = []
        self.payloads: list = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config):
        self.collections[collection_name] = {"vectors": vectors_config, "sparse": sparse_vectors_config}

    def upsert(self, collection_name, points, wait=True):
        self.upserts.setdefault(collection_name, []).extend(points)

    def delete(self, collection_name, points_selector):
        self.deletes.append((collection_name, points_selector))

    def set_payload(self, collection_name, payload, points):
        self.payloads.append((collection_name, payload, points))


def make_chunk(i=0):
    return Chunk(chunk_id=f"hash-{i:04d}", doc_id="hash", text=f"文本{i}",
                 chunk_index=i, title="标题", url="https://x/a", category="教务政策",
                 effective_date=date(2025, 9, 1), crawled_at=datetime(2026, 9, 2))


def test_upsert_creates_collection_and_points():
    client = FakeQdrant()
    chunks = [make_chunk(0), make_chunk(1)]
    embs = [([0.1, 0.2], {3: 1.0}), ([0.2, 0.3], {4: 0.5})]
    upsert_chunks(client, "campus_kb", chunks, embs)
    assert "campus_kb" in client.collections
    points = client.upserts["campus_kb"]
    # 点 ID 为 chunk_id 的确定性 UUID v5（Qdrant 不接受任意字符串 ID）
    assert [p.id for p in points] == [
        str(uuid.uuid5(uuid.NAMESPACE_URL, "hash-0000")),
        str(uuid.uuid5(uuid.NAMESPACE_URL, "hash-0001")),
    ]
    assert points[0].payload["url"] == "https://x/a"
    assert points[0].payload["status"] == "active"
    assert points[0].payload["effective_date"] == "2025-09-01"
    assert points[0].vector["sparse"].indices == [3]
    assert points[0].vector["dense"] == [0.1, 0.2]


def test_upsert_with_real_client_contract():
    """用 qdrant-client 真实本地引擎验证点 ID/UPSERT 契约（防 Fake 宽松掩盖）。"""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    chunks = [make_chunk(0)]
    # 稠密向量须 1024 维——真实引擎按集合配置校验维度（Fake 不校验）
    upsert_chunks(client, "campus_kb", chunks, [([0.1] * 1024, {3: 1.0})])
    hit = client.retrieve("campus_kb", [str(uuid.uuid5(uuid.NAMESPACE_URL, "hash-0000"))])
    assert len(hit) == 1 and hit[0].payload["url"] == "https://x/a"


def test_delete_and_mark_stale_use_url_filter():
    client = FakeQdrant()
    delete_by_url(client, "campus_kb", "https://x/a")
    mark_stale_by_url(client, "campus_kb", "https://x/a")
    assert len(client.deletes) == 1 and len(client.payloads) == 1
    assert client.payloads[0][1] == {"status": "stale"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_ingester.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.ingester'`

- [ ] **Step 3: 实现 pipeline/pipeline/ingester.py**

```python
import uuid

from qdrant_client import QdrantClient, models as qm
from qdrant_client.models import (Distance, PointStruct, SparseVector,
                                  SparseVectorParams, VectorParams)

from .models import Chunk, Embedding


def ensure_collection(client: QdrantClient, name: str):
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )


def _url_filter(url: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="url", match=qm.MatchValue(value=url))])


def _point_id(chunk_id: str) -> str:
    """Qdrant 点 ID 只接受 UUID/无符号整数——chunk_id 确定性映射为 UUID v5。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(client: QdrantClient, name: str,
                  chunks: list[Chunk], embeddings: list[Embedding]):
    ensure_collection(client, name)
    points = [
        PointStruct(
            id=_point_id(chunk.chunk_id),
            vector={
                "dense": dense,
                "sparse": SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
            },
            payload={
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "title": chunk.title,
                "url": chunk.url,
                "category": chunk.category,
                "effective_date": chunk.effective_date.isoformat() if chunk.effective_date else None,
                "chunk_index": chunk.chunk_index,
                "status": "active",
                "crawled_at": chunk.crawled_at.isoformat(),
            },
        )
        for chunk, (dense, sparse) in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=name, points=points, wait=True)


def delete_by_url(client: QdrantClient, name: str, url: str):
    """内容变化时删旧块（按 url 过滤，旧块 doc_id 已变也删得掉）。"""
    client.delete(collection_name=name,
                  points_selector=qm.FilterSelector(filter=_url_filter(url)))


def mark_stale_by_url(client: QdrantClient, name: str, url: str):
    """页面消失时标记过期，不删除（保留审计痕迹）。"""
    client.set_payload(collection_name=name, payload={"status": "stale"},
                       points=qm.FilterSelector(filter=_url_filter(url)))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/test_ingester.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: 提交**

```bash
git add pipeline/pipeline/ingester.py pipeline/tests/test_ingester.py
git commit -m "feat: 入库器（Qdrant 建集合/upsert/按URL删除与过期标记）"
```

---

### Task 9: 增量同步编排 + CLI 入口

**Files:**
- Create: `pipeline/pipeline/sync.py`
- Create: `pipeline/pipeline/cli.py`
- Create: `pipeline/pipeline/__main__.py`
- Test: `pipeline/tests/test_sync.py`

**Interfaces:**
- Consumes: Task 2~8 全部产物
- Produces: `run_full_sync(config: PipelineConfig, *, crawler_factory=None, parse=parse, chunk=chunk, embedder=None, qdrant=None, registry=None) -> SyncReport`（依赖全部可注入，测试用 Fake）；`SyncReport(new, updated, stale, failed: list[str])`；CLI：`python -m pipeline run --config config.yaml`。

- [ ] **Step 1: 写失败测试**

```python
from datetime import date

from pipeline.config import PipelineConfig, SiteConfig
from pipeline.models import Document, RawPage
from pipeline.parser import sha256
from pipeline.sync import SyncReport, run_full_sync


class FakeCrawler:
    def __init__(self, pages):
        self.pages = pages

    async def crawl(self):
        return self.pages


class FakeEmbedder:
    def __init__(self, dim=4):
        self.calls = []
        self.dim = dim

    def embed(self, texts):
        self.calls.extend(texts)
        return [([0.1] * self.dim, {1: 0.5}) for _ in texts]


class FakeRegistry:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.staled: list[str] = []

    def list_all(self):
        return dict(self.rows)

    def upsert(self, rec):
        self.rows[rec.url] = rec

    def mark_stale(self, urls):
        self.staled.extend(urls)


class FakeQdrant:
    def __init__(self):
        self.deleted: list[str] = []
        self.staled: list[str] = []
        self.upserts: list[tuple] = []

    def collection_exists(self, name):
        return True

    def create_collection(self, **kw):
        pass

    def upsert(self, collection_name, points, wait=True):
        self.upserts.append((collection_name, points))

    def delete(self, collection_name, points_selector):
        self.deleted.append(points_selector)

    def set_payload(self, collection_name, payload, points):
        self.staled.append(payload)


def make_config(tmp_path):
    return PipelineConfig(
        sites=[SiteConfig(name="t", category="教务政策",
                          entry_urls=["https://x/a"], allowed_domain="x")],
        data_dir=tmp_path)


def make_doc(url, text):
    return Document(url=url, category="教务政策", title="t", text=text,
                    content_hash=sha256(text), effective_date=date(2025, 9, 1))


def test_sync_new_updated_stale(tmp_path):
    from pipeline.registry import RegistryRecord
    config = make_config(tmp_path)
    crawler = FakeCrawler([RawPage(url="https://x/a", category="教务政策",
                                   html_path=tmp_path / "a.html")])
    # 注意：必须注入 parse（不读真实文件）——默认 parse 会读取不存在的 html_path
    parse_a = lambda raw: [make_doc("https://x/a", "第一条 新内容。" * 20)]
    # 注册表里 a 是旧哈希、b 页面已消失
    reg = FakeRegistry({
        "https://x/a": RegistryRecord(url="https://x/a", title="t", category="教务政策",
                                      content_hash="oldhash", status="active"),
        "https://x/b": RegistryRecord(url="https://x/b", title="t", category="教务政策",
                                      content_hash="oldhash", status="active"),
    })
    qdrant = FakeQdrant()
    emb = FakeEmbedder()
    report = run_full_sync(config, crawler_factory=lambda site: crawler,
                           parse=parse_a, embedder=emb, qdrant=qdrant,
                           registry=reg)

    assert isinstance(report, SyncReport)
    assert report.new == 1 and report.updated == 1
    assert reg.staled == ["https://x/b"]          # 消失页面标记 stale
    assert len(qdrant.staled) == 1                # Qdrant 同步标记
    assert len(qdrant.deleted) == 1               # 内容变化的旧块被删
    assert len(qdrant.upserts) == 2               # 新块 + 更新块各入库一次
    assert report.failed == []


def test_sync_unchanged_doc_is_skipped(tmp_path):
    from pipeline.registry import RegistryRecord
    config = make_config(tmp_path)
    d = make_doc("https://x/a", "同样的内容")
    crawler = FakeCrawler([RawPage(url=d.url, category="教务政策", html_path=tmp_path / "a.html")])
    reg = FakeRegistry({"https://x/a": RegistryRecord(url=d.url, title="t",
                                                      category="教务政策",
                                                      content_hash=d.content_hash)})
    qdrant = FakeQdrant()
    report = run_full_sync(config, crawler_factory=lambda site: crawler,
                           parse=lambda raw: [d], embedder=FakeEmbedder(),
                           qdrant=qdrant, registry=reg)
    assert report.new == 0 and report.updated == 0
    assert qdrant.upserts == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd pipeline && .venv/bin/pytest tests/test_sync.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'pipeline.sync'`

- [ ] **Step 3: 实现 pipeline/pipeline/sync.py**

```python
import asyncio
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from .chunker import chunk
from .config import PipelineConfig
from .crawler import Crawler, _load_robots
from .embedder import EmbedderClient
from .ingester import delete_by_url, mark_stale_by_url, upsert_chunks
from .models import Document, RawPage
from .parser import parse
from .registry import DocumentRegistry, RegistryRecord


@dataclass
class SyncReport:
    new: int = 0
    updated: int = 0
    stale: int = 0
    failed: list[str] = field(default_factory=list)


def run_full_sync(config: PipelineConfig, *,
                  crawler_factory=None, parse=parse, chunk=chunk,
                  embedder=None, qdrant=None, registry=None) -> SyncReport:
    """一次完整增量同步：爬取 → 解析 → 比对 → 入库/更新/标记过期。"""
    report = SyncReport()
    reg = registry or DocumentRegistry(config.mysql_url)
    known = reg.list_all()
    q = qdrant or QdrantClient(url=config.qdrant_url)
    emb = embedder or EmbedderClient(config.embedding_url)

    docs: list[Document] = []
    for site in config.sites:
        crawler = (crawler_factory(site) if crawler_factory
                   else Crawler(site, config.data_dir,
                                robots=_load_robots(site)))
        raw_pages: list[RawPage] = asyncio.run(crawler.crawl())
        for raw in raw_pages:
            try:
                docs.extend(parse(raw))
            except Exception as e:
                report.failed.append(f"{raw.url}: {e}")

    crawled_urls = {d.url for d in docs}
    for doc in docs:
        rec = known.get(doc.url)
        if rec is None:
            kind = "new"
        elif rec.content_hash != doc.content_hash:
            kind = "updated"
        else:
            continue  # 未变化，跳过
        try:
            chunks = chunk(doc, config.chunk_size, config.chunk_overlap)
            embeddings = emb.embed([c.text for c in chunks])
            if kind == "updated":
                delete_by_url(q, config.qdrant_collection, doc.url)
            upsert_chunks(q, config.qdrant_collection, chunks, embeddings)
            reg.upsert(RegistryRecord(
                url=doc.url, title=doc.title, category=doc.category,
                content_hash=doc.content_hash, status="active",
                effective_date=doc.effective_date, published_at=doc.published_at))
            if kind == "new":
                report.new += 1
            else:
                report.updated += 1
        except Exception as e:
            report.failed.append(f"{doc.url}: {e}")

    vanished = [url for url in known if url not in crawled_urls]
    if vanished:
        reg.mark_stale(vanished)
        for url in vanished:
            try:
                mark_stale_by_url(q, config.qdrant_collection, url)
            except Exception as e:
                report.failed.append(f"{url}: {e}")
        report.stale = len(vanished)
    return report
```

- [ ] **Step 4: 实现 pipeline/pipeline/cli.py 与 __main__.py**

```python
import argparse
import logging

from .config import load_config
from .sync import run_full_sync


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline", description="校园RAG数据管线")
    parser.add_argument("command", choices=["run"], default="run", nargs="?")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    report = run_full_sync(config)
    logging.info("同步完成: new=%s updated=%s stale=%s failed=%s",
                 report.new, report.updated, report.stale, report.failed)
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# pipeline/pipeline/__main__.py
from .cli import main

main()
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd pipeline && .venv/bin/pytest tests/ -v`
Expected: PASS（全部 19 tests）

- [ ] **Step 6: 提交**

```bash
git add pipeline/pipeline/sync.py pipeline/pipeline/cli.py pipeline/pipeline/__main__.py pipeline/tests/test_sync.py
git commit -m "feat: 增量同步编排（新增/更新/过期标记）+ CLI 入口"
```

---

### Task 10: 真实环境冒烟 + 定时任务接入

**Files:**
- Create: `deploy/crontab.example`
- Create: `pipeline/smoke_config.yaml`（单栏目小范围配置，冒烟后由正式 config.yaml 替代）

**Interfaces:**
- Consumes: 全部任务 + 真实 MySQL/Qdrant/嵌入服务

- [ ] **Step 1: 启动基础设施与嵌入服务**

```bash
docker compose -f docker-compose.yml up -d          # MySQL + Qdrant
cd embedding_service
docker build -t campus-rag-embedding .
# 从 ModelScope 下载 BGE-M3 到宿主机目录（V100 机器上执行）：
#   pip install modelscope && modelscope download --model BAAI/bge-m3 --local_dir /opt/models/bge-m3
docker run -d --name embedding --gpus '"device=0"' \
  -e BGE_M3_DIR=/models/bge-m3 \
  -v /opt/models/bge-m3:/models/bge-m3 \
  -p 8001:8001 campus-rag-embedding
curl http://localhost:8001/healthz    # 期望 {"status":"ok"}
```

- [ ] **Step 2: 写 pipeline/smoke_config.yaml（只爬一个栏目页面，验证全链路）**

```yaml
sites:
  - name: 冒烟栏目
    category: 教务政策
    entry_urls:
      - https://jwc.example.edu.cn/rule/   # 替换为学校真实栏目页
    allowed_domain: jwc.example.edu.cn
    max_depth: 1
    delay_seconds: 2.5
```

- [ ] **Step 3: 跑一次完整同步**

Run:
```bash
cd pipeline && .venv/bin/python -m pipeline run --config smoke_config.yaml
```
Expected: 日志输出 `同步完成: new=N updated=0 stale=0 failed=[]`（N>0）

- [ ] **Step 4: 验证数据落库**

Run:
```bash
docker compose exec mysql mysql -ucampus -pcampus campus_rag -e \
  "SELECT url, category, status, content_hash FROM documents LIMIT 10;"
curl "http://localhost:6333/collections/campus_kb/points/scroll?limit=3"
```
Expected: documents 表有 N 行 active 记录；Qdrant scroll 返回含 `text`/`url`/`status` payload 的点。

- [ ] **Step 5: 再跑一次验证幂等**

Run: `cd pipeline && .venv/bin/python -m pipeline run --config smoke_config.yaml`
Expected: 日志输出 `new=0 updated=0 stale=0`（未变化内容全部跳过）。

- [ ] **Step 6: 配置每周定时任务**

创建 `deploy/crontab.example`：

```cron
# 每周日 03:17 增量同步（V100 机器 root 或专用用户）
17 3 * * 0 cd /opt/campus-rag/pipeline && /opt/campus-rag/pipeline/.venv/bin/python -m pipeline run --config config.yaml >> /var/log/campus-rag-pipeline.log 2>&1
```

部署方式（在 V100 机器上）：
```bash
mkdir -p /var/log && touch /var/log/campus-rag-pipeline.log
crontab -e   # 粘贴 crontab.example 内容（路径按实际部署目录调整）
```

- [ ] **Step 7: 提交**

```bash
git add deploy/crontab.example pipeline/smoke_config.yaml
git commit -m "chore: 管线定时任务模板与冒烟配置"
```

---

## 收尾自检清单（执行者完成全部任务后）

- [ ] `pipeline/.venv/bin/pytest tests/ -v` 全绿（19 tests；嵌入服务另有 2 tests）
- [ ] 冒烟同步两次：第一次 new>0，第二次全 0（幂等验证）
- [ ] documents 表与 Qdrant payload 数据一致
- [ ] 设计文档 §4 的每个环节都有对应实现：爬取(crawler) / 清洗解析(parser) / 分块(chunker) / 嵌入(embedder+service) / 入库(ingester) / 增量(sync+registry) / 调度(crontab)
