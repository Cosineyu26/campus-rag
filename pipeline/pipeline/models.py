from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

Embedding = tuple[list[float], dict[int, float]]  # (dense, sparse{token_id: weight})


@dataclass
class RawPage:
    """爬虫落盘的原始页面：HTML 文件 + 元数据 + PDF 附件清单。"""
    url: str
    category: str
    html_path: Path
    title: str = ""
    published_at: Optional[date] = None
    pdf_files: list[tuple[str, Path]] = field(default_factory=list)  # [(原始URL, 本地路径)]
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class Document:
    """一篇可入库文档（HTML 正文或 PDF 全文）。"""
    url: str
    category: str
    title: str
    text: str
    content_hash: str          # SHA-256(text)，即 doc_id
    published_at: Optional[date] = None
    effective_date: Optional[date] = None
    source_path: Optional[Path] = None


@dataclass
class Chunk:
    """分块结果，chunk_id 即 Qdrant 点 ID。"""
    chunk_id: str              # f"{content_hash}-{index:04d}"
    doc_id: str                # content_hash
    text: str
    chunk_index: int
    title: str
    url: str
    category: str
    effective_date: Optional[date]
    crawled_at: datetime
