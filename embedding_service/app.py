import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = os.getenv("BGE_M3_DIR", "BAAI/bge-m3")

app = FastAPI(title="embedding-service")
_model = None


def get_model():
    """BGE-M3 须用官方 FlagEmbedding 的 BGEM3FlagModel——sentence-transformers
    的模型卡不含 sparse 输出模块，无法返回 lexical_weights。"""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(MODEL_DIR, use_fp16=True)
    return _model


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    model = get_model()
    enc = model.encode(req.texts, batch_size=32,
                       return_dense=True, return_sparse=True)
    dense = enc["dense_vecs"].tolist()
    sparse = [{str(k): float(v) for k, v in row.items()} for row in enc["lexical_weights"]]
    return {"dense": dense, "sparse": sparse}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
