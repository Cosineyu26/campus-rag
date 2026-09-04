from datetime import date

from pipeline.chunker import (chunk, split_by_clauses, split_by_headings,
                              split_by_size)
from pipeline.models import Document


def doc(text: str) -> Document:
    from pipeline.parser import sha256
    return Document(url="https://school.edu.cn/r", category="教务政策", title="某规定",
                    text=text, content_hash=sha256(text),
                    effective_date=date(2025, 9, 1))


def test_split_by_headings_splits_on_markdown():
    text = "# 第一章\n第一条 内容。\n# 第二章\n第二条 内容。"
    parts = split_by_headings(text)
    assert len(parts) == 2
    assert parts[0].startswith("# 第一章") and parts[1].startswith("# 第二章")


def test_split_by_clauses_uses_article_boundaries():
    text = "第一章 总则\n第一条 学生应当遵守纪律。\n第二条 考试作弊记零分。\n第三条 请假需审批。"
    parts = split_by_clauses(text)
    assert len(parts) == 3
    assert "第一条" in parts[0] and "第二条" in parts[1] and "第三条" in parts[2]


def test_split_by_clauses_single_when_no_clauses():
    text = "这是一段没有任何条款标记的普通文本。"
    assert split_by_clauses(text) == [text]


def test_split_by_size_overlaps():
    text = "字" * 1000
    parts = split_by_size(text, size=400, overlap=60)
    assert len(parts) == 3
    assert parts[0][-60:] == parts[1][:60]  # 重叠正确


def test_chunk_generates_deterministic_ids_and_metadata():
    d = doc("第一条 内容甲。" * 30)
    chunks = chunk(d, size=100, overlap=20)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"{d.content_hash}-{i:04d}"
        assert c.doc_id == d.content_hash
        assert c.url == d.url and c.category == d.category
        assert c.effective_date == d.effective_date
        assert c.chunk_index == i
