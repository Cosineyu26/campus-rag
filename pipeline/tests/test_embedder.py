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
