import httpx

from .models import Embedding


class EmbedderClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[Embedding]:
        resp = self._client.post(f"{self.base_url}/embed", json={"texts": texts})
        resp.raise_for_status()
        data = resp.json()
        dense, sparse = data["dense"], data["sparse"]
        if not (len(dense) == len(sparse) == len(texts)):
            raise ValueError(f"embed 响应长度不匹配: dense={len(dense)} sparse={len(sparse)} texts={len(texts)}")
        return [
            (dense_vec, {int(k): v for k, v in sparse_vec.items()})
            for dense_vec, sparse_vec in zip(dense, sparse)
        ]
