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
        return [
            (dense, {int(k): v for k, v in sparse.items()})
            for dense, sparse in zip(data["dense"], data["sparse"])
        ]
