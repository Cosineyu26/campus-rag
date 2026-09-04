from contextlib import contextmanager
from datetime import datetime, timezone
from os import getenv

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def _engine(db_url: str | None = None):
    url = db_url or getenv("DB_URL", "sqlite:///campus_rag.db")
    kw = {"pool_pre_ping": True} if url.startswith("mysql") else {}
    return create_engine(url, **kw)


@contextmanager
def session_scope(db_url: str | None = None):
    eng = _engine(db_url)
    Base.metadata.create_all(eng)
    SessionLocal = sessionmaker(bind=eng, expire_on_commit=False)
    s: Session = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
        eng.dispose()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
