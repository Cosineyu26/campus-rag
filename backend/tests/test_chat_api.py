# test_chat_api.py（Fake 全链注入；SSE 用 TestClient stream 解析）
import json

import pytest
from fastapi.testclient import TestClient


class FakeLLM:
    """同时支持 complete 与 stream（逐字吐 complete 结果）。"""
    def __init__(self, out="根据资料[1]，本科生学制四年。\n\n参考来源见上。", delay=0.0):
        self.out = out
        self.last_messages = None

    def complete(self, messages, max_tokens=512, temperature=0.7):
        self.last_messages = messages
        return self.out

    def stream(self, messages, max_tokens=2048, temperature=0.7):
        self.last_messages = messages
        for ch in self.out:
            yield ch


class FakeEmbed:
    def embed(self, texts):
        return [([0.1] * 4, {1: 1.0})] * len(texts)


class FakeRerank:
    def rerank(self, query, docs):
        return [0.9] * len(docs)


class FakeQdrant:
    def query_points(self, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(points=[
            SimpleNamespace(payload={"text": "本科生学制为四年，最长六年。",
                                     "url": "https://x/rule", "title": "学籍规定",
                                     "category": "教务政策",
                                     "effective_date": "2025-09-01"},
                            score=1.0)])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.chat_service as cs
    from app.api import chat as chat_mod
    from app.api import sessions as sess_mod
    from app.config import Settings

    db_url = f"sqlite:///{(tmp_path / 'chat.db').as_posix()}"
    settings = Settings(rerank_threshold=0.1)
    svc = cs.ChatService(settings=settings, db_url=db_url,
                         llm=FakeLLM(), embed=FakeEmbed(),
                         qdrant=FakeQdrant(), rerank=FakeRerank())
    # chat 与 sessions 两路由各自持有模块级 _get_service，须一并注入；
    # 端点以 _get_service(request) 调用，注入函数须接收 req 形参
    monkeypatch.setattr(chat_mod, "_get_service", lambda req: svc)
    monkeypatch.setattr(sess_mod, "_get_service", lambda req: svc)

    from app.main import create_app
    app = create_app(settings=settings)
    return TestClient(app)


def _collect_sse(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def test_chat_stream_full_flow(client):
    resp = client.post("/api/chat", json={"session_id": "s1", "message": "本科学制几年？"})
    assert resp.status_code == 200
    events = _collect_sse(resp)
    kinds = [e["type"] for e in events]
    assert kinds == ["delta", "sources", "done"] or "delta" in kinds
    text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert "[1]" in text
    sources = next(e for e in events if e["type"] == "sources")
    assert sources["sources"][0]["url"] == "https://x/rule"


def test_chat_no_hit_returns_fallback(client, monkeypatch):
    import app.chat_service as cs
    from app.api import chat as chat_mod
    from app.config import Settings

    class NoHitQdrant:
        def query_points(self, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(points=[])

    monkeypatch.setattr(chat_mod, "_get_service", lambda req: cs.ChatService(
        settings=Settings(rerank_threshold=0.1),
        db_url="sqlite://", llm=FakeLLM(), embed=FakeEmbed(),
        qdrant=NoHitQdrant(), rerank=FakeRerank()))
    resp = client.post("/api/chat", json={"session_id": "s2", "message": "火星怎么走？"})
    events = _collect_sse(resp)
    text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert "没有找到" in text
    assert all(e["type"] != "error" for e in events)


def test_sessions_history_and_delete(client):
    client.post("/api/chat", json={"session_id": "s3", "message": "学制几年？"})
    resp = client.get("/api/sessions/s3")
    assert resp.status_code == 200
    msgs = resp.json()["messages"]
    assert msgs[-1]["role"] == "assistant" and "学制" in msgs[-1]["content"]
    assert client.delete("/api/sessions/s3").status_code == 200
    assert client.get("/api/sessions/s3").json()["messages"] == []
