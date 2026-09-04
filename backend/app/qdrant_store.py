"""Qdrant 混合检索：稠密+稀疏 prefetch + RRF 融合，payload 层 status 过滤。"""
from dataclasses import dataclass

from qdrant_client import QdrantClient, models as qm


@dataclass
class SearchHit:
    text: str
    url: str
    title: str
    category: str
    effective_date: str | None
    score: float


def build_qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def _active_filter() -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="status",
                                             match=qm.MatchValue(value="active"))])


def hybrid_search(client: QdrantClient, collection: str,
                  dense: list[float], sparse: dict[int, float], *,
                  status: str = "active",
                  top_k: int = 50, limit: int = 20) -> list[SearchHit]:
    """稠密+稀疏双路各取 top_k，RRF 融合后返回 limit 条（含 payload 过滤）。"""
    f = qm.Filter(must=[qm.FieldCondition(key="status",
                                          match=qm.MatchValue(value=status))]) \
        if status else None
    resp = client.query_points(
        collection_name=collection,
        prefetch=[
            qm.Prefetch(query=dense, using="dense", limit=top_k, filter=f),
            qm.Prefetch(query=qm.SparseVector(indices=list(sparse.keys()),
                                              values=list(sparse.values())),
                        using="sparse", limit=top_k, filter=f),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit, with_payload=True,
    )
    hits = []
    for p in resp.points:
        pl = p.payload
        hits.append(SearchHit(
            text=pl.get("text", ""), url=pl.get("url", ""),
            title=pl.get("title", ""), category=pl.get("category", ""),
            effective_date=pl.get("effective_date"), score=p.score or 0.0,
        ))
    return hits
