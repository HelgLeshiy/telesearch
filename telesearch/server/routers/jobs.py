"""Job status endpoints."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db, get_session_factory
from ..deps import AccessScope, get_scope
from ..models import Job
from ..schemas import JobOut

router = APIRouter(prefix="/workspaces/{workspace_id}/jobs", tags=["jobs"])


def _to_out(j: Job) -> JobOut:
    return JobOut(
        id=j.id,
        workspace_id=j.workspace_id,
        source_id=j.source_id,
        type=j.type,
        state=j.state,
        progress=j.progress,
        message=j.message,
        error=j.error,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


@router.get("", response_model=list[JobOut])
def list_jobs(
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    rows = db.scalars(
        select(Job)
        .where(Job.workspace_id == scope.workspace.id)
        .order_by(Job.created_at.desc())
        .limit(100)
    ).all()
    return [_to_out(j) for j in rows]


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> JobOut:
    job = db.get(Job, job_id)
    if job is None or job.workspace_id != scope.workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return _to_out(job)


@router.get("/{job_id}/events")
def job_events(
    job_id: str,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Server-Sent Events stream of a job's progress until it finishes."""
    job = db.get(Job, job_id)
    if job is None or job.workspace_id != scope.workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    workspace_id = scope.workspace.id

    def stream():
        factory = get_session_factory()
        last = None
        for _ in range(600):  # ~5 min ceiling at 0.5s
            s = factory()
            try:
                j = s.get(Job, job_id)
            finally:
                s.close()
            if j is None or j.workspace_id != workspace_id:
                break
            snapshot = (j.state, round(j.progress, 3), j.message, j.error)
            if snapshot != last:
                payload = json.dumps({
                    "id": j.id, "state": j.state, "progress": j.progress,
                    "message": j.message, "error": j.error,
                })
                yield f"data: {payload}\n\n"
                last = snapshot
            if j.state in ("completed", "failed"):
                break
            time.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")
