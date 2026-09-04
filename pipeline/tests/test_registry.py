from datetime import date

from pipeline.registry import DocumentRegistry, RegistryRecord


def rec(url, h="abc", status="active"):
    return RegistryRecord(url=url, title="t", category="c",
                          content_hash=h, status=status,
                          effective_date=date(2025, 9, 1))


def test_upsert_and_list(tmp_path):
    db = tmp_path / "t.db"
    reg = DocumentRegistry(f"sqlite:///{db.as_posix()}")  # as_posix: Windows 路径兼容
    reg.upsert(rec("https://x/a"))
    reg.upsert(rec("https://x/a", h="newhash"))  # 重复 upsert = 更新
    reg.upsert(rec("https://x/b"))
    rows = reg.list_all()
    assert set(rows) == {"https://x/a", "https://x/b"}
    assert rows["https://x/a"].content_hash == "newhash"


def test_mark_stale(tmp_path):
    db = tmp_path / "t.db"
    reg = DocumentRegistry(f"sqlite:///{db.as_posix()}")
    reg.upsert(rec("https://x/a"))
    reg.upsert(rec("https://x/b"))
    reg.mark_stale(["https://x/a"])
    rows = reg.list_all()
    assert rows["https://x/a"].status == "stale"
    assert rows["https://x/b"].status == "active"
