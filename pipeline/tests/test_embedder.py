import httpx
import respx

from pipeline.embedder import EmbedderClient


def test_embed_maps_response(respx_mock):
    route = respx_mock.post("http://embed:8001/embed").mock(return_value=httpx.Response(200, json={
        "dense": [[0.1, 0.2], [0.3, 0.4]],
        "sparse": [{"7": 1.5, "99": 0.8}, {"7": 0.9}],
    }))
    client = EmbedderClient("http://embed:8001")
    embs = client.embed(["问题一", "问题二"])
    assert route.called
    assert embs[0] == ([0.1, 0.2], {7: 1.5, 99: 0.8})
    assert embs[1] == ([0.3, 0.4], {7: 0.9})


def test_embed_propagates_http_error(respx_mock):
    respx_mock.post("http://embed:8001/embed").mock(return_value=httpx.Response(500))
    client = EmbedderClient("http://embed:8001")
    try:
        client.embed(["x"])
        assert False, "should raise"
    except httpx.HTTPStatusError:
        pass


def test_embed_rejects_length_mismatch(respx_mock):
    """dense/sparse/texts 长度不一致时抛 ValueError，避免下游 zip 静默截断。"""
    respx_mock.post("http://embed:8001/embed").mock(return_value=httpx.Response(200, json={
        "dense": [[0.1, 0.2]],
        "sparse": [{"7": 1.5}, {"8": 0.9}],
    }))
    client = EmbedderClient("http://embed:8001")
    try:
        client.embed(["问题一", "问题二"])
        assert False, "should raise"
    except ValueError as e:
        assert "长度不匹配" in str(e)
