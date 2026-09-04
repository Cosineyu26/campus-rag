from fastapi import FastAPI

from .api import chat, sessions
from .chat_service import ChatService
from .config import Settings, load_settings


def build_service(settings: Settings) -> ChatService:
    from .embed_client import EmbedClient
    from .llm import LlmClient
    from .qdrant_store import build_qdrant_client
    from .rerank_client import RerankClient

    return ChatService(
        settings=settings, db_url=settings.db_url,
        llm=LlmClient(settings.llm_url, settings.llm_model),
        embed=EmbedClient(settings.embed_url),
        qdrant=build_qdrant_client(settings.qdrant_url),
        rerank=RerankClient(settings.rerank_url),
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="campus-rag-backend")
    app.state.settings = settings
    app.state.chat_service = build_service(settings)
    app.include_router(chat.router)
    app.include_router(sessions.router)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


app = create_app()
