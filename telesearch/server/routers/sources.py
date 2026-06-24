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
from ..queue import enqueue
from ..schemas import ShareCreate, SourceOut

router = APIRouter(prefix="/workspaces/{workspace_id}/sources", tags=["sources"])


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

    # Quota: cap the number of sources per workspace.
    if settings.max_sources_per_workspace:
        existing = db.scalar(
            select(func.count(Source.id)).where(Source.workspace_id == scope.workspace.id)
        )
        if existing >= settings.max_sources_per_workspace:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"source quota reached ({settings.max_sources_per_workspace})",
            )

    source = Source(
        workspace_id=scope.workspace.id,
        collection_id="",  # set to the row id below for guaranteed uniqueness
        kind=kind,
        name=name or (file.filename or "upload"),
        status="uploaded",
        created_by=scope.user.id,
    )
    db.add(source)
    db.flush()
    source.collection_id = source.id
    source.blob_key = f"{scope.workspace.id}/{source.id}"

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
    )
    audit.record(
        db, action="source.upload", user_id=scope.user.id,
        workspace_id=scope.workspace.id, resource=source.id,
        ip=request.client.host if request.client else "",
    )
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
