from fastapi import APIRouter, Request

from ..chat_service import ChatService
from ..db import session_scope
from ..models import MessageRow, SessionRow

router = APIRouter(prefix="/api")


def _get_service(req: Request) -> ChatService:
    return req.app.state.chat_service


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    svc = _get_service(request)
    history = svc.load_history(session_id, 1000)
    return {"messages": history}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    svc = _get_service(request)
    with session_scope(svc.db_url) as s:
        sess = s.query(SessionRow).filter_by(session_id=session_id).one_or_none()
        if sess:
            s.query(MessageRow).filter_by(session_id=sess.id).delete()
            s.delete(sess)
    return {"deleted": True}
