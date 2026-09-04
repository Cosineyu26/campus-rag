import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..chat_service import ChatService

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    stream: bool = True


def _get_service(req: Request) -> ChatService:
    return req.app.state.chat_service


def _sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event, **data}, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    svc = _get_service(request)
    if not req.stream:
        text, citations, found = svc.answer(req.session_id, req.message, stream=False)
        return JSONResponse({"text": text, "sources": citations, "found": found})

    async def gen():
        try:
            text, citations, found = svc.answer(req.session_id, req.message, stream=True)
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            yield _sse("done", {})
            return
        yield _sse("delta", {"text": text})
        yield _sse("sources", {"sources": citations})
        yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream")
