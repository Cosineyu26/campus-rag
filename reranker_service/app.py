import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = os.getenv("RERANKER_MODEL_DIR", "BAAI/bge-reranker-v2-m3")

app = FastAPI(title="reranker-service")
_model = None


def get_model():
    global _model
    if _model is None:
        from FlagEmbedding import FlagReranker

        _model = FlagReranker(MODEL_DIR, use_fp16=True)
    return _model


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
def rerank(req: RerankRequest):
    if not req.documents:
        return {"scores": []}
    model = get_model()
    pairs = [[req.query, d] for d in req.documents]
    scores = model.compute_score(pairs, normalize=True)
    return {"scores": [float(s) for s in scores]}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
