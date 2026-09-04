import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.crawler import Crawler, FetchError, PageData, links_from_result
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


def _fake_result(links=None):
    return SimpleNamespace(success=True, url="https://school.edu.cn/p",
                           html="<html></html>", links=links)


def test_links_from_result_merges_internal_and_external():
    """dict 形 links：internal+external 合并按序提取 href。"""
    result = _fake_result(links={
        "internal": [{"href": "/a", "text": "甲"}, {"href": "/b", "text": "乙"}],
        "external": [{"href": "https://school.edu.cn/c", "text": "丙"}],
    })
    assert links_from_result(result) == ["/a", "/b", "https://school.edu.cn/c"]


def test_links_from_result_handles_empty_links():
    """空/缺失 links 都返回空列表，不抛异常。"""
    assert links_from_result(_fake_result(links=None)) == []
    assert links_from_result(_fake_result(links={})) == []
    assert links_from_result(_fake_result(links={"internal": [], "external": []})) == []


def test_links_from_result_skips_malformed_entries():
    """非 dict 条目、缺 href、空 href 一律剔除，只留有效链接。"""
    result = _fake_result(links={
        "internal": [{"href": "/a"}, {"text": "无href"}, "坏条目", None, {"href": ""}],
        "external": [{"href": "/b"}, 42],
    })
    assert links_from_result(result) == ["/a", "/b"]
