"""Background jobs: a DB-backed queue with an in-process worker.

Jobs are persisted in the ``jobs`` table so progress survives restarts and is
visible to the API. The default worker runs in a thread inside the API process
(portable, no broker). For scale this is the seam to swap in Redis + Celery/Arq:
the handler functions stay the same; only the dispatch changes.
"""

from __future__ import annotations

import threading
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from ..service import RequestContext, index_source
from .config import ServerSettings, get_server_settings
from .models import Job, Source

JobHandler = Callable[[Session, Job, ServerSettings], None]
_HANDLERS: dict[str, JobHandler] = {}


def register_handler(job_type: str) -> Callable[[JobHandler], JobHandler]:
    def deco(fn: JobHandler) -> JobHandler:
        _HANDLERS[job_type] = fn
        return fn
    return deco


def enqueue(
    db: Session,
    *,
    workspace_id: str,
    job_type: str,
    source_id: str | None = None,
    params: dict | None = None,
) -> Job:
    job = Job(
        workspace_id=workspace_id,
        source_id=source_id,
        type=job_type,
        state="pending",
        params=params or {},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@register_handler("ingest")
def _handle_ingest(db: Session, job: Job, server_settings: ServerSettings) -> None:
    """Materialize an uploaded source and build/extend its index."""
    from .blobs import get_blob_store

    source = db.get(Source, job.source_id) if job.source_id else None
    if source is None:
        raise ValueError("ingest job has no source")

    source.status = "indexing"
    job.message = "parsing and indexing"
    job.progress = 0.1
    db.commit()

    root = get_blob_store(server_settings).materialize(source.blob_key)
    settings = get_settings()
    media = bool(job.params.get("index_media", server_settings.index_media_by_default))
    ctx = RequestContext(workspace_id=source.workspace_id, user_id=source.created_by)

    result = index_source(
        root,
        settings,
        ctx=ctx,
        kind=source.kind or None,
        collection_id=source.collection_id,
        chat_name=source.name or None,
        do_images=media,
        do_videos=media,
        do_audio=media,
        do_ocr=media,
        do_documents=True,
    )

    source.status = "ready"
    job.message = (
        f"parser={result.parser} messages={result.messages} chunks={result.chunks}"
    )
    job.progress = 1.0
    db.commit()


def run_job(
    session_factory: sessionmaker[Session],
    job_id: str,
    server_settings: ServerSettings | None = None,
) -> None:
    """Execute a single job by id, recording success/failure on the row."""
    server_settings = server_settings or get_server_settings()
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if job is None or job.state not in ("pending", "running"):
            return
        handler = _HANDLERS.get(job.type)
        job.state = "running"
        db.commit()
        if handler is None:
            raise ValueError(f"no handler for job type {job.type!r}")
        handler(db, job, server_settings)
        job.state = "completed"
        db.commit()
    except Exception as exc:  # record failure, don't crash the worker
        db.rollback()
        job = db.get(Job, job_id)
        if job is not None:
            job.state = "failed"
            job.error = str(exc)[:1999]
            if job.source_id:
                src = db.get(Source, job.source_id)
                if src is not None:
                    src.status = "failed"
                    src.error = str(exc)[:1999]
            db.commit()
    finally:
        db.close()


class Worker:
    """Polls the jobs table for pending work and runs it in a background thread."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        server_settings: ServerSettings | None = None,
    ):
        self.session_factory = session_factory
        self.server_settings = server_settings or get_server_settings()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _claim_pending(self) -> str | None:
        db = self.session_factory()
        try:
            job = db.scalars(
                select(Job).where(Job.state == "pending").order_by(Job.created_at)
            ).first()
            return job.id if job else None
        finally:
            db.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._claim_pending()
            if job_id is None:
                self._stop.wait(self.server_settings.worker_poll_seconds)
                continue
            run_job(self.session_factory, job_id, self.server_settings)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="telesearch-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
