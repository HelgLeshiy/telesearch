"""FastAPI application factory wiring auth, workspaces, sources, jobs and search."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import ServerSettings, get_server_settings
from .db import get_session_factory, init_db
from .queue import Worker
from .routers import (
    auth,
    global_search,
    graph,
    guides,
    jobs,
    presets,
    search,
    sources,
    workspaces,
)

log = logging.getLogger("telesearch.server")

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or get_server_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings)
        if settings.is_dev_secret:
            log.warning(
                "TELESEARCH_SECRET_KEY is the insecure default; set a strong "
                "secret before exposing this service."
            )
        worker: Worker | None = None
        if settings.worker_inline:
            worker = Worker(get_session_factory(), settings)
            worker.start()
        app.state.worker = worker
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()

    app = FastAPI(title="telesearch", version="0.1.0", lifespan=lifespan)

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(auth.router, prefix="/api")
    app.include_router(workspaces.router, prefix="/api")
    app.include_router(sources.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(global_search.router, prefix="/api")
    app.include_router(graph.router, prefix="/api")
    app.include_router(presets.router, prefix="/api")
    app.include_router(guides.router, prefix="/api")

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    return app


app = create_app()
