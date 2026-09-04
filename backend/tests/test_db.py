from datetime import datetime, timezone

from app.db import session_scope
from app.models import MessageRow, SessionRow


def test_session_and_message_crud(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 't.db').as_posix()}"
    with session_scope(db_url) as s:
        sess = SessionRow(session_id="abc-123", created_at=datetime.now(timezone.utc),
                          last_active_at=datetime.now(timezone.utc))
        s.add(sess)
        s.commit()
        sess_id = sess.id
    with session_scope(db_url) as s:
        m = MessageRow(session_id=sess_id, role="user", content="你好",
                       citations="[]", created_at=datetime.now(timezone.utc))
        s.add(m)
        s.commit()
    with session_scope(db_url) as s:
        rows = s.query(MessageRow).all()
        assert len(rows) == 1 and rows[0].content == "你好"
