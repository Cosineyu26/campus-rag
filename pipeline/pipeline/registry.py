from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import Date, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="active")
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    published_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    last_crawled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


@dataclass
class RegistryRecord:
    url: str
    title: str
    category: str
    content_hash: str
    status: str = "active"
    effective_date: date | None = None
    published_at: date | None = None


def _to_record(row: DocumentRow) -> RegistryRecord:
    return RegistryRecord(url=row.url, title=row.title, category=row.category,
                          content_hash=row.content_hash, status=row.status,
                          effective_date=row.effective_date, published_at=row.published_at)


class DocumentRegistry:
    """文档注册表：增量比对的事实来源（测试用 SQLite，生产用 MySQL，同一套 ORM）。"""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)

    def list_all(self) -> dict[str, RegistryRecord]:
        with Session(self.engine) as s:
            rows = s.query(DocumentRow).all()
        return {r.url: _to_record(r) for r in rows}

    def upsert(self, rec: RegistryRecord):
        with Session(self.engine) as s:
            row = s.query(DocumentRow).filter_by(url=rec.url).one_or_none()
            if row is None:
                row = DocumentRow(url=rec.url)
                s.add(row)
            row.title = rec.title
            row.category = rec.category
            row.content_hash = rec.content_hash
            row.status = rec.status
            row.effective_date = rec.effective_date
            row.published_at = rec.published_at
            row.last_crawled_at = datetime.now()
            row.last_ingested_at = datetime.now()
            s.commit()

    def mark_stale(self, urls: list[str]):
        if not urls:
            return
        with Session(self.engine) as s:
            s.query(DocumentRow).filter(DocumentRow.url.in_(urls)).update(
                {"status": "stale"}, synchronize_session=False)
            s.commit()
