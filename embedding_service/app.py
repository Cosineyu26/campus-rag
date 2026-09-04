import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.getenv("BGE_M3_DIR", "BAAI/bge-m3")

app = FastAPI(title="embedding-service")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_DIR)
    return _model


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    model = get_model()
    enc = model.encode(
        req.texts, batch_size=32,
        return_dense=True, return_sparse=True, normalize_embeddings=True,
    )
    dense = enc["dense_vecs"].tolist()
    sparse = [{str(k): float(v) for k, v in row.items()} for row in enc["lexical_weights"]]
    return {"dense": dense, "sparse": sparse}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
