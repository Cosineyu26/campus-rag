"""检索编排：改写 → 嵌入 → Qdrant 混合检索 → 重排 → 阈值/过期过滤。

全部依赖注入（llm/embed/qdrant/rerank），单测用 Fake；真实组装在依赖注入容器层
（chat 端点按 Settings 构造）。
"""
import logging
from dataclasses import dataclass, field
from datetime import date

from .config import Settings
from .embed_client import EmbedClient
from .llm import LlmClient
from .prompts import rewrite_prompt
from .qdrant_store import SearchHit, build_qdrant_client, hybrid_search
from .rerank_client import RerankClient

log = logging.getLogger(__name__)


def rewrite_query(llm, history: list[str], question: str) -> str:
    """改写当前问题为独立检索语句；LLM 异常时降级用原文（不阻塞主流程）。"""
    try:
        return llm.complete([{"role": "user", "content": rewrite_prompt(history, question)}],
                            max_tokens=64, temperature=0.0).strip()
    except Exception as e:  # 改写失败不阻塞检索
        log.warning("rewrite failed, fallback to original: %s", e)
        return question


def _is_expired(effective_date: str | None, today: date | None = None) -> bool:
    """过期判定：effective_date 非空且 > 今天 → 过期（日期为字符串 ISO，Qdrant
    不支持字符串 range，故在应用层过滤；为空视作长期有效）。"""
    if not effective_date:
        return False
    today = today or date.today()
    try:
        return date.fromisoformat(effective_date) > today
    except ValueError:
        return False


@dataclass
class RetrievePipeline:
    settings: Settings
    embed: EmbedClient
    qdrant: object  # QdrantClient（测试注入 Fake）
    rerank: RerankClient
    llm: LlmClient

    def retrieve(self, question: str, history: list[str]) -> tuple[list[SearchHit], str]:
        s = self.settings
        used_query = rewrite_query(self.llm, history[-s.rewrite_rounds * 2:], question)
        [(dense, sparse)] = self.embed.embed([used_query])
        hits = hybrid_search(self.qdrant, s.qdrant_collection, dense, sparse,
                             top_k=s.top_k_recall, limit=s.top_n_rerank)
        if not hits:
            return [], used_query
        # 过期过滤（含 future-dated）在重排前做，省一次模型调用
        hits = [h for h in hits if not _is_expired(h.effective_date)]
        if not hits:
            return [], used_query
        scores = self.rerank.rerank(used_query, [h.text for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
        final = [(h, sc) for h, sc in ranked if sc >= s.rerank_threshold][:s.top_n_final]
        return [h for h, _ in final], used_query
