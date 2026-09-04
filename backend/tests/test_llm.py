import json

import httpx
import pytest

from app.llm import LlmClient


CHAT = {"message": {"role": "assistant", "content": "你好"}}
BASE = "http://localhost:11434"


def test_complete_ollama_contract(respx_mock):
    route = respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, json=CHAT))
    client = LlmClient(BASE, "qwen3:8b")
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp == "你好"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:8b"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is False
    assert body["think"] is False  # qwen3 默认思考 → content 为空，必须显式关闭
    assert body["max_tokens"] == 512
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.7}


def test_stream_parses_sse(respx_mock):
    chunks = [
        'data: {"message": {"content": "你好"}}\n\n',
        'data: {"message": {"content": "世界"}}\n\n',
        "data: [DONE]\n\n",
    ]
    respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, text="".join(chunks),
                                    headers={"content-type": "text/event-stream"}))
    client = LlmClient(BASE, "qwen3:8b")
    parts = [p for p in client.stream([{"role": "user", "content": "hi"}])]
    assert parts == ["你好", "世界"]


def test_http_error_propagates(respx_mock):
    respx_mock.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(500))
    client = LlmClient(BASE, "qwen3:8b")
    with pytest.raises(httpx.HTTPStatusError):
        client.complete([{"role": "user", "content": "hi"}])
