import httpx
import respx

from app.embed_client import EmbedClient


def test_embed_contract(respx_mock):
    respx_mock.post("http://emb:8001/embed").mock(return_value=httpx.Response(200, json={
        "dense": [[0.1, 0.2], [0.3, 0.4]],
        "sparse": [{"7": 1.5}, {"7": 0.9}],
    }))
    client = EmbedClient("http://emb:8001")
    embs = client.embed(["问题", "文本"])
    assert embs == [([0.1, 0.2], {7: 1.5}), ([0.3, 0.4], {7: 0.9})]


def test_embed_length_mismatch_raises(respx_mock):
    respx_mock.post("http://emb:8001/embed").mock(return_value=httpx.Response(200, json={
        "dense": [[0.1]], "sparse": [{"7": 1.5}, {"8": 1.0}]}))
    client = EmbedClient("http://emb:8001")
    try:
        client.embed(["a", "b"])
        assert False, "should raise"
    except ValueError:
        pass
