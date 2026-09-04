from datetime import date

from pipeline.config import PipelineConfig, SiteConfig
from pipeline.models import Document, RawPage
from pipeline.parser import sha256
from pipeline.sync import SyncReport, run_full_sync


class FakeCrawler:
    def __init__(self, pages):
        self.pages = pages
        self.failures: list[str] = []  # 与真实 Crawler 对齐：失败的页面 URL

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
        self.staled: list[dict] = []
        self.reactivated: list[dict] = []
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
        if payload.get("status") == "active":
            self.reactivated.append(payload)
        else:
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
    # 注意：必须注入 parse（不读取真实文件）——默认 parse 会读取不存在的 html_path
    # fixture 修正：brief 原版 parse 只产出 a 一篇（注册表已有旧哈希 → 只能算 updated，
    # 无法满足下方 new==1 与 upserts==2），故补一篇注册表没有的 c 验证 new 分支。
    parse_a = lambda raw: [make_doc("https://x/a", "第一条 新内容。" * 20),
                           make_doc("https://x/c", "新发布的校园新政策。" * 10)]
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
    assert report.new == 0 and report.updated == 0 and report.revived == 0
    assert qdrant.upserts == []


def test_sync_revives_unchanged_stale_doc(tmp_path):
    """内容未变但注册表为 stale 的文档应被复活（恢复 active），而非直接跳过。"""
    from pipeline.registry import RegistryRecord
    config = make_config(tmp_path)
    d = make_doc("https://x/a", "同样的内容")
    crawler = FakeCrawler([RawPage(url=d.url, category="教务政策", html_path=tmp_path / "a.html")])
    reg = FakeRegistry({"https://x/a": RegistryRecord(url=d.url, title="t",
                                                      category="教务政策",
                                                      content_hash=d.content_hash,
                                                      status="stale")})
    qdrant = FakeQdrant()
    report = run_full_sync(config, crawler_factory=lambda site: crawler,
                           parse=lambda raw: [d], embedder=FakeEmbedder(),
                           qdrant=qdrant, registry=reg)
    assert report.revived == 1
    assert report.new == 0 and report.updated == 0
    assert reg.rows["https://x/a"].status == "active"    # 注册表已复活
    assert qdrant.reactivated == [{"status": "active"}]  # Qdrant 侧同步复活
    assert qdrant.upserts == []                          # 内容未变不重复入库


def test_sync_vanish_excludes_failed_site_category(tmp_path):
    """站点爬取失败（零页面或有失败）时其栏目 URL 不被误标 stale；健康站点照常标记。"""
    from pipeline.registry import RegistryRecord
    broken = FakeCrawler([])
    broken.failures = ["https://x/a"]                    # 教务政策站点爬取异常
    healthy = FakeCrawler([RawPage(url="https://x/b", category="学术科研",
                                   html_path=tmp_path / "b.html")])
    sites = [
        SiteConfig(name="s1", category="教务政策", entry_urls=["https://x/a"], allowed_domain="x"),
        SiteConfig(name="s2", category="学术科研", entry_urls=["https://x/b"], allowed_domain="x"),
    ]
    config = PipelineConfig(sites=sites, data_dir=tmp_path)
    reg = FakeRegistry({
        "https://x/c": RegistryRecord(url="https://x/c", title="t", category="教务政策",
                                      content_hash="h", status="active"),
        "https://x/d": RegistryRecord(url="https://x/d", title="t", category="学术科研",
                                      content_hash="h", status="active"),
    })
    qdrant = FakeQdrant()
    report = run_full_sync(config,
                           crawler_factory=lambda site: broken if site.category == "教务政策" else healthy,
                           parse=lambda raw: [make_doc(raw.url, "学术新闻。" * 5)],
                           embedder=FakeEmbedder(), qdrant=qdrant, registry=reg)
    assert report.stale == 1
    assert reg.staled == ["https://x/d"]                 # 只有健康站点的消失页标 stale
    assert "https://x/c" not in reg.staled               # 失败站点的消失页被豁免
    assert len(qdrant.staled) == 1
