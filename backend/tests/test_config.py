import os

from app.config import Settings


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("LLM_URL", "http://llm:9000")
    s = Settings()
    assert s.llm_url == "http://llm:9000"


def test_settings_defaults():
    s = Settings()
    assert s.llm_url == "http://localhost:8000"
    assert s.qdrant_url == "http://localhost:6333"
    assert s.embed_url == "http://localhost:8001"
    assert s.rerank_url == "http://localhost:8002"
    assert s.rerank_threshold == 0.3
    assert s.history_rounds == 6
    assert s.rewrite_rounds == 3
