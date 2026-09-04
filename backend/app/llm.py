"""llama.cpp / OpenAI 兼容 LLM 客户端（流式与非流式）。

llm.py 只依赖 HTTP 契约，可指向任何 OpenAI 兼容服务（llama.cpp server 已部署，
未来可无缝换 vLLM）。
"""
import json

import httpx


class LlmClient:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0):
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._model = model
        self._client = httpx.Client(timeout=timeout)

    def _payload(self, messages, max_tokens, temperature, stream):
        return {"model": self._model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature,
                "stream": stream}

    def complete(self, messages: list[dict], max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        resp = self._client.post(self._url, json=self._payload(
            messages, max_tokens, temperature, stream=False))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict], max_tokens: int = 2048,
               temperature: float = 0.7):
        with self._client.stream("POST", self._url, json=self._payload(
                messages, max_tokens, temperature, stream=True)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, json.JSONDecodeError):
                    delta = None
                if delta:
                    yield delta
