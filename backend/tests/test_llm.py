import httpx
import respx
import pytest

from app.llm import LlmClient


CHAT = {"choices": [{"message": {"role": "assistant", "content": "你好，我是校园助手。"}}]}


def test_complete(respx_mock):
    route = respx_mock.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT))
    client = LlmClient("http://llm:8000", "qwen3-8b")
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp == "你好，我是校园助手。"
    body = route.calls[0].request.content
    assert b'"model": "qwen3-8b"' in body or b'"model":"qwen3-8b"' in body


def test_stream_parses_sse(respx_mock):
    chunks = [
        'data: {"choices": [{"delta": {"content": "你好"}}]}\n\n',
        'data: {"choices": [{"delta": {"content": "世界"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    respx_mock.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="".join(chunks),
                                    headers={"content-type": "text/event-stream"}))
    client = LlmClient("http://llm:8000", "qwen3-8b")
    parts = [p for p in client.stream([{"role": "user", "content": "hi"}])]
    assert parts == ["你好", "世界"]


def test_http_error_propagates(respx_mock):
    respx_mock.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(500))
    client = LlmClient("http://llm:8000", "qwen3-8b")
    with pytest.raises(httpx.HTTPStatusError):
        client.complete([{"role": "user", "content": "hi"}])
