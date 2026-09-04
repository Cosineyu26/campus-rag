import hashlib
import re
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
import trafilatura

from .models import Document, RawPage

_EFFECTIVE_PATTERNS = [
    re.compile(r"自(\d{4})年(\d{1,2})月(\d{1,2})日起?(?:施行|执行|实施)"),
    re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(?:施行|执行|生效)"),
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """去除行首尾空白并合并连续空行。"""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_effective_date(text: str) -> date | None:
    for pat in _EFFECTIVE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _parse_html(raw: RawPage) -> Document:
    html = Path(raw.html_path).read_text(encoding="utf-8", errors="ignore")
    text = clean_text(trafilatura.extract(html, include_comments=False) or "")
    return Document(url=raw.url, category=raw.category, title=raw.title or "",
                    text=text, content_hash=sha256(text),
                    published_at=raw.published_at,
                    effective_date=extract_effective_date(text),
                    source_path=raw.html_path)


def _parse_pdf(pdf_url: str, pdf_path: Path, raw: RawPage) -> Document:
    with fitz.open(pdf_path) as doc:
        text = clean_text("\n".join(page.get_text() for page in doc))
    return Document(url=pdf_url, category=raw.category, title=raw.title or "",
                    text=text, content_hash=sha256(text),
                    published_at=raw.published_at,
                    effective_date=extract_effective_date(text),
                    source_path=pdf_path)


def parse(raw: RawPage) -> list[Document]:
    """一个 RawPage → 若干 Document（HTML 正文 + 每份 PDF 各一篇），空文本丢弃。"""
    docs = [_parse_html(raw)]
    docs += [_parse_pdf(url, path, raw) for url, path in raw.pdf_files]
    return [d for d in docs if d.text]
