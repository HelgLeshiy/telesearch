"""Source endpoints: upload (-> background indexing), list, delete, share."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...config import get_settings
from .. import audit
from ..blobs import get_blob_store
from ..config import ServerSettings, get_server_settings
from ..db import get_db
from ..deps import AccessScope, get_scope
from ..models import Job, Share, Source, User
from ..queue import enqueue, pending_job_count
from ..schemas import (
    CompleteRequest,
    PresignRequest,
    PresignResponse,
    ShareCreate,
    SourceOut,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/sources", tags=["sources"])


def _hash_upload(file: UploadFile) -> str:
    import hashlib

    h = hashlib.sha256()
    file.file.seek(0)
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        h.update(chunk)
    file.file.seek(0)
    return h.hexdigest()


def _to_out(s: Source) -> SourceOut:
    return SourceOut(
        id=s.id,
        workspace_id=s.workspace_id,
        collection_id=s.collection_id,
        kind=s.kind,
        name=s.name,
        status=s.status,
        bytes=s.bytes,
        error=s.error,
        created_at=s.created_at,
    )


@router.post("", response_model=SourceOut)
def upload_source(
    request: Request,
    file: UploadFile = File(...),
    kind: str = Form(""),
    name: str = Form(""),
    index_media: bool = Form(False),
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> SourceOut:
    scope.require("member")

    # Quota: cap the number of sources and in-flight jobs per workspace.
    if settings.max_sources_per_workspace:
        existing = db.scalar(
            select(func.count(Source.id)).where(Source.workspace_id == scope.workspace.id)
        )
        if existing >= settings.max_sources_per_workspace:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"source quota reached ({settings.max_sources_per_workspace})",
            )
    if settings.max_pending_jobs_per_workspace and pending_job_count(
        db, scope.workspace.id
    ) >= settings.max_pending_jobs_per_workspace:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many jobs in flight")

    # Content hash for dedup (read once, then rewind for saving).
    content_hash = _hash_upload(file)

    source = Source(
        workspace_id=scope.workspace.id,
        collection_id="",  # set to the row id below for guaranteed uniqueness
        kind=kind,
        name=name or (file.filename or "upload"),
        status="uploaded",
        content_hash=content_hash,
        created_by=scope.user.id,
    )
    db.add(source)
    db.flush()
    source.collection_id = source.id
    source.blob_key = f"{scope.workspace.id}/{source.id}"

    # Dedup: identical content already indexed in this workspace -> skip re-work.
    dup = db.scalars(
        select(Source).where(
            Source.workspace_id == scope.workspace.id,
            Source.content_hash == content_hash,
            Source.id != source.id,
            Source.status.in_(("uploaded", "indexing", "ready")),
        )
    ).first()
    if content_hash and dup is not None:
        source.status = "duplicate"
        db.commit()
        db.refresh(source)
        audit.record(
            db, action="source.duplicate", user_id=scope.user.id,
            workspace_id=scope.workspace.id, resource=source.id,
        )
        return _to_out(source)

    store = get_blob_store(settings)
    written = store.save(source.blob_key, file.filename or "upload", file.file)

    # Quota: reject oversized uploads (clean up what we wrote).
    if settings.max_upload_bytes and written > settings.max_upload_bytes:
        store.delete(source.blob_key)
        db.delete(source)
        db.commit()
        raise HTTPException(
            413, f"upload exceeds limit ({settings.max_upload_bytes} bytes)"
        )

    source.bytes = written
    db.commit()
    db.refresh(source)

    enqueue(
        db,
        workspace_id=scope.workspace.id,
        job_type="ingest",
        source_id=source.id,
        params={"index_media": bool(index_media)},
        lane="gpu" if index_media else "cpu",
    )
    audit.record(
        db, action="source.upload", user_id=scope.user.id,
        workspace_id=scope.workspace.id, resource=source.id,
        ip=request.client.host if request.client else "",
    )
    return _to_out(source)


def _quota_guard(db: Session, scope: AccessScope, settings: ServerSettings) -> None:
    if settings.max_sources_per_workspace:
        existing = db.scalar(
            select(func.count(Source.id)).where(Source.workspace_id == scope.workspace.id)
        )
        if existing >= settings.max_sources_per_workspace:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"source quota reached ({settings.max_sources_per_workspace})",
            )
    if settings.max_pending_jobs_per_workspace and pending_job_count(
        db, scope.workspace.id
    ) >= settings.max_pending_jobs_per_workspace:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many jobs in flight")


@router.post("/presign", response_model=PresignResponse)
def presign_source(
    body: PresignRequest,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> PresignResponse:
    """Create a pending source and return a direct (presigned) upload URL.

    Lets large files upload straight to storage (local signed endpoint or S3)
    instead of streaming through the JSON API; finish with ``/complete``.
    """
    scope.require("member")
    _quota_guard(db, scope, settings)

    source = Source(
        workspace_id=scope.workspace.id, collection_id="", kind=body.kind,
        name=body.name or body.filename, status="uploading", created_by=scope.user.id,
    )
    db.add(source)
    db.flush()
    source.collection_id = source.id
    source.blob_key = f"{scope.workspace.id}/{source.id}"
    db.commit()

    up = get_blob_store(settings).presign_put(source.blob_key, body.filename)
    return PresignResponse(
        source_id=source.id,
        upload=up,
        complete_url=f"/api/workspaces/{scope.workspace.id}/sources/{source.id}/complete",
    )


@router.post("/{source_id}/complete", response_model=SourceOut)
def complete_source(
    source_id: str,
    body: CompleteRequest,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> SourceOut:
    """Finalize a presigned upload: hash, dedup, and queue indexing."""
    scope.require("member")
    source = db.get(Source, source_id)
    if source is None or source.workspace_id != scope.workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "source not found")
    if source.status != "uploading":
        raise HTTPException(status.HTTP_409_CONFLICT, "source not awaiting completion")

    store = get_blob_store(settings)
    content_hash = store.content_hash(source.blob_key)
    source.content_hash = content_hash
    source.status = "uploaded"

    dup = db.scalars(
        select(Source).where(
            Source.workspace_id == scope.workspace.id,
            Source.content_hash == content_hash,
            Source.id != source.id,
            Source.status.in_(("uploaded", "indexing", "ready")),
        )
    ).first()
    if content_hash and dup is not None:
        source.status = "duplicate"
        db.commit()
        db.refresh(source)
        return _to_out(source)

    db.commit()
    enqueue(
        db, workspace_id=scope.workspace.id, job_type="ingest", source_id=source.id,
        params={"index_media": bool(body.index_media)},
        lane="gpu" if body.index_media else "cpu",
    )
    audit.record(
        db, action="source.upload", user_id=scope.user.id,
        workspace_id=scope.workspace.id, resource=source.id,
    )
    db.refresh(source)
    return _to_out(source)


@router.get("", response_model=list[SourceOut])
def list_sources(
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> list[SourceOut]:
    stmt = select(Source).where(Source.workspace_id == scope.workspace.id)
    if scope.collections is not None:
        stmt = stmt.where(Source.collection_id.in_(scope.collections))
    rows = db.scalars(stmt.order_by(Source.created_at)).all()
    return [_to_out(s) for s in rows]


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(
    source_id: str,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> None:
    scope.require("member")
    source = db.get(Source, source_id)
    if source is None or source.workspace_id != scope.workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "source not found")

    # Remove vectors (best-effort), blob, and metadata.
    from ...index.store import VectorStore

    engine_settings = get_settings()
    store = VectorStore(
        engine_settings.workspace_db_path(scope.workspace.id), create=False
    )
    store.delete_collection(source.collection_id)
    get_blob_store(settings).delete(source.blob_key)

    db.query(Job).filter(Job.source_id == source.id).delete()
    db.query(Share).filter(
        Share.resource_type == "source", Share.resource_id == source.id
    ).delete()
    db.delete(source)
    db.commit()
    audit.record(
        db, action="source.delete", user_id=scope.user.id,
        workspace_id=scope.workspace.id, resource=source_id,
    )


@router.post("/{source_id}/share", status_code=status.HTTP_204_NO_CONTENT)
def share_source(
    source_id: str,
    body: ShareCreate,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> None:
    scope.require("admin")
    source = db.get(Source, source_id)
    if source is None or source.workspace_id != scope.workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "source not found")
    target = db.scalars(select(User).where(User.email == body.user_email.lower())).first()
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")

    existing = db.scalars(
        select(Share).where(
            Share.resource_type == "source",
            Share.resource_id == source.id,
            Share.principal_type == "user",
            Share.principal_id == target.id,
        )
    ).first()
    if existing is None:
        db.add(
            Share(
                resource_type="source",
                resource_id=source.id,
                principal_type="user",
                principal_id=target.id,
                role=body.role,
            )
        )
        db.commit()
    audit.record(
        db, action="source.share", user_id=scope.user.id,
        workspace_id=scope.workspace.id, resource=f"{source.id}->{target.id}",
    )
