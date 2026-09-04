from datetime import date

from app.qdrant_store import SearchHit
from app.retrieval import RetrievePipeline, rewrite_query


def _hit(text="第一条 内容", date_s="2025-09-01", score=0.9):
    return SearchHit(text=text, url="https://x/a", title="t", category="教务政策",
                     effective_date=date_s, score=score)


class FakeLLM:
    def __init__(self, out="改写后的问题"):
        self.out = out
        self.calls = 0

    def complete(self, messages, max_tokens=512, temperature=0.7):
        self.calls += 1
        return self.out


def test_rewrite_query_falls_back_on_error():
    class Boom:
        def complete(self, **kw):
            raise RuntimeError("llm down")

    assert rewrite_query(Boom(), ["h1"], "q") == "q"


def test_pipeline_filters_expired_and_low_score():
    from app.config import Settings

    class FakeRerank:
        def rerank(self, query, docs):
            return [0.95, 0.2, 0.31, 0.9]  # 与 docs 等长

    # 过期（2020）与低分（0.2）被剔除；阈值 0.3
    emb = type("E", (), {"embed": lambda self, t: [([0.1] * 4, {1: 1.0})] * len(t)})()
    qd = type("Q", (), {"query_points": lambda self, **kw: type("R", (), {
        "points": [
            type("P", (), {"payload": {"text": "A", "url": "u1", "title": "t",
                                       "category": "c", "effective_date": "2025-09-01"},
                            "score": 1.0})(),
            type("P", (), {"payload": {"text": "B", "url": "u2", "title": "t",
                                       "category": "c", "effective_date": "2020-01-01"},
                            "score": 1.0})(),
            type("P", (), {"payload": {"text": "C", "url": "u3", "title": "t",
                                       "category": "c", "effective_date": None},
                            "score": 1.0})(),
            type("P", (), {"payload": {"text": "D", "url": "u4", "title": "t",
                                       "category": "c", "effective_date": "2030-01-01"},
                            "score": 1.0})(),
        ]})()})()

    settings = Settings(top_n_rerank=20, top_n_final=5, rerank_threshold=0.3)
    pipe = RetrievePipeline(settings=settings, embed=emb, qdrant=qd,
                            rerank=FakeRerank(), llm=FakeLLM())
    # retrieve 返回 (hits, used_query)（见检索接口契约），此处只取 hits
    hits, _used_query = pipe.retrieve("问题", [])
    # 过期(2020)与未来(2030)剔除；0.2 低分剔除；剩 A(0.95) C(0.31)
    assert [h.text for h in hits] == ["A", "C"]
