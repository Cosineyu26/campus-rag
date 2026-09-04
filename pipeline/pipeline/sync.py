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
