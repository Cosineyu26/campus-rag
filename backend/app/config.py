import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class Settings:
    db_url: str = field(default_factory=lambda: _env("DB_URL", ""))
    llm_url: str = field(default_factory=lambda: _env("LLM_URL", "http://localhost:8000"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3-8b"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "campus_kb"))
    embed_url: str = field(default_factory=lambda: _env("EMBED_URL", "http://localhost:8001"))
    rerank_url: str = field(default_factory=lambda: _env("RERANK_URL", "http://localhost:8002"))
    rerank_threshold: float = field(default_factory=lambda: float(_env("RERANK_THRESHOLD", "0.3")))
    history_rounds: int = field(default_factory=lambda: int(_env("HISTORY_ROUNDS", "6")))
    rewrite_rounds: int = field(default_factory=lambda: int(_env("REWRITE_ROUNDS", "3")))
    top_k_recall: int = field(default_factory=lambda: int(_env("TOP_K_RECALL", "50")))
    top_n_rerank: int = field(default_factory=lambda: int(_env("TOP_N_RERANK", "20")))
    top_n_final: int = field(default_factory=lambda: int(_env("TOP_N_FINAL", "5")))


def load_settings() -> Settings:
    return Settings()
