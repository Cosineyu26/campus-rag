"""Ollama 原生聊天客户端（流式与非流式）。

本实现为 Ollama 原生契约（POST {base}/api/chat，base 默认 http://localhost:11434）：
部署引擎已固定为 Ollama，且必须经原生端点调用——其 OpenAI 兼容端点不透传
think/num_ctx，qwen3 默认思考模式会导致 message.content 为空。

关键点：
- think: False 必须显式下传（qwen3 默认思考 → content 为空）；
- options.num_ctx 默认 8192（RAG 长上下文需要；ollama 容器默认 4096）；
- max_tokens 放请求体顶层。

换 OpenAI 兼容引擎（如 vLLM）时只需改本文件：端点路径、options 映射与响应
解析是全部差异所在（fetch 层思想，服务端对 llm.py 无感）。
"""
import json

import httpx


class LlmClient:
    """Ollama 原生聊天客户端。

    base_url 指向 Ollama 服务（本地默认为 http://localhost:11434，容器部署由
    Settings.llm_url 注入，如 http://ollama:11434）。
    """

    def __init__(self, base_url: str, model: str, num_ctx: int = 8192,
                 timeout: float = 300.0):
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._client = httpx.Client(timeout=timeout)
        self._num_ctx = num_ctx

    def _payload(self, messages, max_tokens, temperature, stream):
        return {"model": self._model,
                "messages": messages,
                "stream": stream,
                "think": False,
                "max_tokens": max_tokens,
                "options": {"num_ctx": self._num_ctx,
                            "temperature": temperature}}

    def complete(self, messages: list[dict], max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        resp = self._client.post(self._url, json=self._payload(
            messages, max_tokens, temperature, stream=False))
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def stream(self, messages: list[dict], max_tokens: int = 2048,
               temperature: float = 0.7):
        """逐行 yield content 增量片段。

        Ollama 原生 /api/chat 每行是独立 JSON（增量片段）；按行解析容错跳过
        无 content 的行（空片段/仅元数据行）。终止条件：data: [DONE]（SSE 封装，
        契约测试同款）或 done: true（Ollama 原生 NDJSON 行内标记）。
        """
        with self._client.stream("POST", self._url, json=self._payload(
                messages, max_tokens, temperature, stream=True)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                    if line == "[DONE]":
                        break
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk["message"]["content"]
                    done = chunk.get("done", False)
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                if done:
                    break
                if content:
                    yield content
