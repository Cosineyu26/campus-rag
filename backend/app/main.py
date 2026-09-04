from fastapi import FastAPI

from .config import load_settings

settings = load_settings()


def create_app() -> FastAPI:
    app = FastAPI(title="campus-rag-backend")
    # 后续任务在此挂载路由：app.include_router(chat.router) 等

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


app = create_app()
