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


def upsert_chunks(client: QdrantClient, name: str,
                  chunks: list[Chunk], embeddings: list[Embedding]):
    ensure_collection(client, name)
    points = [
        PointStruct(
            id=chunk.chunk_id,
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
