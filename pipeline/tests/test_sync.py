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
    assert report.new == 0 and report.updated == 0
    assert qdrant.upserts == []
