import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import RawPage  # noqa: F401  确保包结构可导入


@dataclass
class SiteConfig:
    name: str
    category: str
    entry_urls: list[str]
    allowed_domain: str
    max_depth: int = 3
    delay_seconds: float = 2.5


@dataclass
class PipelineConfig:
    sites: list[SiteConfig]
    data_dir: Path = Path("data")
    chunk_size: int = 500
    chunk_overlap: int = 60
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = "campus_kb"
    mysql_url: str = field(default_factory=lambda: os.getenv(
        "MYSQL_URL", "mysql+pymysql://campus:campus@localhost:3306/campus_rag?charset=utf8mb4"))
    embedding_url: str = field(default_factory=lambda: os.getenv("EMBEDDING_URL", "http://localhost:8001"))


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sites = [SiteConfig(**s) for s in raw["sites"]]
    opts = raw.get("options", {})
    return PipelineConfig(sites=sites, **opts)
