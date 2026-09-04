import numpy as np
import pytest
from fastapi.testclient import TestClient


class FakeModel:
    def encode(self, texts, batch_size=None, return_dense=None, return_sparse=None, normalize_embeddings=None):
        return {
            "dense_vecs": np.array([[0.1] * 3] * len(texts)),
            "lexical_weights": [{0: 1.0, 5: 0.5}] * len(texts),
        }


@pytest.fixture()
def client(monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "_model", FakeModel())
    return TestClient(app_module.app)


def test_embed_contract(client):
    resp = client.post("/embed", json={"texts": ["你好", "世界"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["dense"]) == 2
    assert data["sparse"] == [{"0": 1.0, "5": 0.5}, {"0": 1.0, "5": 0.5}]


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
