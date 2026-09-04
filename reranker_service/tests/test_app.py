import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeReranker:
    def compute_score(self, pairs, batch_size=None, max_length=None, normalize=None):
        return np.array([0.9, 0.1] * (len(pairs) // 2) or [0.9] * len(pairs))[:len(pairs)]


@pytest.fixture()
def client(monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_model", FakeReranker())
    return TestClient(app_module.app)


def test_rerank_contract(client):
    resp = client.post("/rerank", json={"query": "问题", "documents": ["甲", "乙"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["scores"]) == 2
    assert data["scores"][0] == 0.9


def test_rerank_empty_documents(client):
    resp = client.post("/rerank", json={"query": "q", "documents": []})
    assert resp.status_code == 200
    assert resp.json() == {"scores": []}


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
