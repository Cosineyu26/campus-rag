"""聊天编排：历史装载 → 检索 → 生成（流式/非流式）→ 引用提取 → 消息落库。

会话为匿名（session_id 由调用方给定）；上下文最近 history_rounds 轮。
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator

from .config import Settings
from .db import session_scope, utcnow
from .embed_client import EmbedClient
from .llm import LlmClient
from .models import MessageRow, SessionRow
from .prompts import FALLBACK_TEXT, build_user_message
from .qdrant_store import SearchHit
from .rerank_client import RerankClient
from .retrieval import RetrievePipeline

log = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, hits: list[SearchHit]) -> list[dict]:
    """提取回答中出现过的 [n] 引用（保序去重）→ 来源列表。"""
    idxs = []
    for m in _CITE_RE.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= len(hits) and i not in idxs:
            idxs.append(i)
    return [{"index": i, "title": hits[i - 1].title, "url": hits[i - 1].url}
            for i in idxs]


class ChatService:
    def __init__(self, settings: Settings, db_url: str,
                 llm: LlmClient, embed: EmbedClient,
                 qdrant, rerank: RerankClient):
        self.settings = settings
        self.db_url = db_url
        self.pipe = RetrievePipeline(settings, embed, qdrant, rerank, llm)
        self.llm = llm

    # ---- 会话 ----
    def ensure_session(self, session_key: str) -> int:
        with session_scope(self.db_url) as s:
            row = s.query(SessionRow).filter_by(session_id=session_key).one_or_none()
            now = utcnow()
            if row is None:
                row = SessionRow(session_id=session_key, created_at=now,
                                 last_active_at=now)
                s.add(row)
                s.flush()
            else:
                row.last_active_at = now
            return row.id

    def load_history(self, session_key: str, limit_rounds: int) -> list[dict]:
        with session_scope(self.db_url) as s:
            sess = s.query(SessionRow).filter_by(session_id=session_key).one_or_none()
            if sess is None:
                return []
            rows = (s.query(MessageRow).filter_by(session_id=sess.id)
                    .order_by(MessageRow.id.desc()).limit(limit_rounds * 2).all())
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    def _save_messages(self, session_key: str, user_msg: str,
                       answer: str, citations: list[dict]):
        sid = self.ensure_session(session_key)
        with session_scope(self.db_url) as s:
            s.add(MessageRow(session_id=sid, role="user", content=user_msg,
                             citations="[]", created_at=utcnow()))
            s.add(MessageRow(session_id=sid, role="assistant", content=answer,
                             citations=json.dumps(citations, ensure_ascii=False),
                             created_at=utcnow()))

    # ---- 问答 ----
    def _history_texts(self, history: list[dict]) -> list[str]:
        return [f"{'问' if m['role'] == 'user' else '答'}：{m['content']}" for m in history]

    def _retrieve_hits(self, question: str, history_texts: list[str]) -> list[SearchHit]:
        hits, _ = self.pipe.retrieve(question, history_texts)
        return hits

    def answer(self, session_key: str, question: str,
               stream: bool = False):
        """返回 (text, citations, found)；stream=True 时 text 为完整文本（内部流式拼装）。"""
        history = self.load_history(session_key, self.settings.history_rounds)
        hits = self._retrieve_hits(question, self._history_texts(history))
        if not hits:
            self._save_messages(session_key, question, FALLBACK_TEXT, [])
            return FALLBACK_TEXT, [], False
        user_msg = build_user_message(question, [h.text for h in hits])
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": user_msg})
        if stream:
            parts = [chunk for chunk in self.llm.stream(messages)]
            text = "".join(parts)
        else:
            text = self.llm.complete(messages, max_tokens=1024)
        citations = parse_citations(text, hits)
        self._save_messages(session_key, question, text, citations)
        return text, citations, True
