import re
from datetime import datetime

from .models import Chunk, Document

_CLAUSE_RE = re.compile(r"(?:第[一二三四五六七八九十百零〇\d]+条|[一二三四五六七八九十]+、)")
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def split_by_headings(text: str) -> list[str]:
    """按 Markdown 标题行切分（trafilatura 保留的标题结构）。无标题时返回原文单段。"""
    matches = list(_HEADING_LINE_RE.finditer(text))
    if not matches:
        return [text]
    parts, last = [], 0
    for m in matches:
        if m.start() > last:
            parts.append(text[last:m.start()])
        last = m.start()
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def split_by_clauses(text: str) -> list[str]:
    """按条款级边界切分（'第一条'、'一、'）。无条款标记时返回原文单段。"""
    matches = list(_CLAUSE_RE.finditer(text))
    if len(matches) < 2:
        return [text]
    parts, last = [], matches[0].start()
    for m in matches[1:]:
        parts.append(text[last:m.start()])
        last = m.start()
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def split_by_size(text: str, size: int, overlap: int) -> list[str]:
    """定长切分（中文按字符数近似 token 数），相邻块重叠 `overlap` 字符。"""
    if not text.strip():
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk(doc: Document, size: int = 500, overlap: int = 60) -> list[Chunk]:
    """三级切分：标题结构 → 条款级 → 定长兜底；chunk_id 确定性生成（幂等入库的基础）。"""
    out: list[Chunk] = []
    crawled_at = datetime.now()
    for section in split_by_headings(doc.text):
        for piece in split_by_clauses(section):
            for text in split_by_size(piece, size, overlap):
                idx = len(out)
                out.append(Chunk(
                    chunk_id=f"{doc.content_hash}-{idx:04d}",
                    doc_id=doc.content_hash,
                    text=text, chunk_index=idx, title=doc.title, url=doc.url,
                    category=doc.category, effective_date=doc.effective_date,
                    crawled_at=crawled_at,
                ))
    return out
