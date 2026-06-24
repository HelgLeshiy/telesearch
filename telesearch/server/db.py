"""Database engine, session factory and base model.

SQLAlchemy 2.x with a portable default (SQLite). Schema is created with
``init_db`` for ease of running anywhere; production deployments should manage
migrations with Alembic (the models here are the source of truth for that).
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import ServerSettings, get_server_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine(settings: ServerSettings):
    url = settings.resolved_database_url
    connect_args = {}
    if url.startswith("sqlite"):
        # Allow use across the request thread and the background worker thread.
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args)


def get_engine(settings: ServerSettings | None = None):
    global _engine, _SessionLocal
    if _engine is None:
        settings = settings or get_server_settings()
        _engine = _build_engine(settings)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def init_db(settings: ServerSettings | None = None) -> None:
    """Create tables if they don't exist."""
    from . import models  # noqa: F401  (register mappers)

    engine = get_engine(settings)
    Base.metadata.create_all(engine)


def reset_engine() -> None:
    """Drop cached engine/session (used by tests to switch databases)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()
