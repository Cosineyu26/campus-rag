"""嵌入服务 HTTP 客户端——契约与 pipeline/embedder.py 一致（见数据管线计划），
后端独立复制以解耦（服务地址经 Settings.embed_url 注入）。"""
import httpx


class EmbedClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[tuple[list[float], dict[int, float]]]:
        resp = self._client.post(f"{self.base_url}/embed", json={"texts": texts})
        resp.raise_for_status()
        data = resp.json()
        dense, sparse = data["dense"], data["sparse"]
        if not (len(dense) == len(sparse) == len(texts)):
            raise ValueError(
                f"embed 响应长度不匹配: dense={len(dense)} sparse={len(sparse)} texts={len(texts)}")
        return [(d, {int(k): v for k, v in s.items()}) for d, s in zip(dense, sparse)]
