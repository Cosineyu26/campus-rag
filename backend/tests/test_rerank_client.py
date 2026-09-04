import httpx
import respx

from app.rerank_client import RerankClient


def test_rerank_contract(respx_mock):
    respx_mock.post("http://rk:8002/rerank").mock(return_value=httpx.Response(200, json={
        "scores": [0.81, 0.12]}))
    client = RerankClient("http://rk:8002")
    scores = client.rerank("问题", ["文本A", "文本B"])
    assert scores == [0.81, 0.12]


def test_rerank_length_mismatch_raises(respx_mock):
    respx_mock.post("http://rk:8002/rerank").mock(return_value=httpx.Response(200, json={
        "scores": [0.5]}))
    client = RerankClient("http://rk:8002")
    try:
        client.rerank("q", ["a", "b"])
        assert False, "should raise"
    except ValueError:
        pass
