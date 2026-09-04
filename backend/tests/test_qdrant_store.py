from qdrant_client import QdrantClient, models as qm

from app.qdrant_store import SearchHit, hybrid_search


def _seed(client: QdrantClient, name: str):
    client.create_collection(name,
                             vectors_config={"dense": qm.VectorParams(
                                 size=4, distance=qm.Distance.COSINE)},
                             sparse_vectors_config={"sparse": qm.SparseVectorParams()})
    points = [
        qm.PointStruct(id=1, vector={"dense": [1.0, 0.0, 0.0, 0.0],
                                     "sparse": qm.SparseVector(indices=[1], values=[1.0])},
                       payload={"text": "第一条 学籍", "url": "https://x/a", "title": "规定",
                                "category": "教务政策", "effective_date": "2025-09-01",
                                "status": "active"}),
        qm.PointStruct(id=2, vector={"dense": [0.0, 1.0, 0.0, 0.0],
                                     "sparse": qm.SparseVector(indices=[2], values=[1.0])},
                       payload={"text": "旧政策", "url": "https://x/b", "title": "旧",
                                "category": "教务政策", "effective_date": "2020-01-01",
                                "status": "stale"}),
    ]
    client.upsert(name, points)


def test_hybrid_search_filters_stale():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    _seed(client, "campus_kb")
    # 稠密查询与点1相近、与点2正交 → RRF 后点1在前；点2 status=stale 应被过滤
    hits = hybrid_search(client, "campus_kb",
                         dense=[0.9, 0.1, 0.0, 0.0], sparse={1: 0.8},
                         top_k=10, limit=5)
    assert [h.url for h in hits] == ["https://x/a"]
    assert hits[0].effective_date == "2025-09-01"
    assert isinstance(hits[0], SearchHit)
