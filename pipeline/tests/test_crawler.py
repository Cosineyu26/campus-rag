import asyncio
import json
from pathlib import Path

from pipeline.crawler import (Crawler, FetchError, PageData,
                              extract_links_from_html)
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


HTML_FIXTURE = """<html><body>
<a href="/list1.htm">列表1</a>
<a href="info/1041/10608.htm">详情</a>
<a href="javascript:void(0)">跳过</a>
<a href="#frag">锚点</a>
<a href="mailto:x@y.edu.cn">邮件</a>
<a href="/list1.htm">重复</a>
<a href="http://outer.edu.cn/x">外域</a>
<a>无href</a>
</body></html>"""


def test_extract_links_from_html_dedups_and_filters():
    """纯 HTML 链接提取：去重保序、剔除 js/mailto/锚点，保留相对与外域原样。"""
    assert extract_links_from_html(HTML_FIXTURE) == [
        "/list1.htm", "info/1041/10608.htm",
        "http://outer.edu.cn/x",
    ]


def test_extract_links_from_html_empty():
    assert extract_links_from_html("<html></html>") == []


def test_fetch_page_default_http(respx_mock):
    """fetch_page_default 用注入的 client 抓静态页：状态码/重定向/链接提取。"""
    import httpx
    from pipeline.crawler import fetch_page_default

    respx_mock.get("https://school.edu.cn/p.htm").mock(
        return_value=httpx.Response(200, text=HTML_FIXTURE))
    client = httpx.AsyncClient()
    page = asyncio.run(fetch_page_default("https://school.edu.cn/p.htm", client=client))
    assert page.final_url == "https://school.edu.cn/p.htm"
    assert "/list1.htm" in page.links and "javascript:void(0)" not in page.links
    asyncio.run(client.aclose())


def test_fetch_page_default_raises_on_http_error(respx_mock):
    """非 200 状态抛 FetchError（单页失败不中断整轮的语义依赖它）。"""
    import httpx
    from pipeline.crawler import fetch_page_default

    respx_mock.get("https://school.edu.cn/404.htm").mock(
        return_value=httpx.Response(404))
    client = httpx.AsyncClient()
    try:
        asyncio.run(fetch_page_default("https://school.edu.cn/404.htm", client=client))
        assert False, "should raise"
    except FetchError as e:
        assert "404" in str(e)
    finally:
        asyncio.run(client.aclose())
