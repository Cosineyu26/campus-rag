from datetime import date, datetime

from pipeline.ingester import delete_by_url, ensure_collection, mark_stale_by_url, upsert_chunks
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
    assert [p.id for p in points] == ["hash-0000", "hash-0001"]
    assert points[0].payload["url"] == "https://x/a"
    assert points[0].payload["status"] == "active"
    assert points[0].payload["effective_date"] == "2025-09-01"
    assert points[0].vector["sparse"].indices == [3]
    assert points[0].vector["dense"] == [0.1, 0.2]


def test_delete_and_mark_stale_use_url_filter():
    client = FakeQdrant()
    delete_by_url(client, "campus_kb", "https://x/a")
    mark_stale_by_url(client, "campus_kb", "https://x/a")
    assert len(client.deletes) == 1 and len(client.payloads) == 1
    assert client.payloads[0][1] == {"status": "stale"}
