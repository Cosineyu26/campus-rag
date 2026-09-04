"""重排服务 HTTP 客户端（bge-reranker-v2-m3 跑在独立服务，端口 8002）。"""
import httpx


class RerankClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        resp = self._client.post(f"{self.base_url}/rerank",
                                 json={"query": query, "documents": docs})
        resp.raise_for_status()
        scores = resp.json()["scores"]
        if len(scores) != len(docs):
            raise ValueError(f"rerank 响应长度不匹配: {len(scores)} vs {len(docs)}")
        return scores
