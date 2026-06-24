"""Workspace management endpoints."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import get_settings
from .. import audit
from ..config import ServerSettings, get_server_settings
from ..db import get_db
from ..deps import AccessScope, get_current_user, get_scope
from ..models import AuditLog, GraphSnapshot, Job, Membership, Share, Source, User, Workspace
from ..schemas import AuditOut, WorkspaceCreate, WorkspaceOut

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceOut)
def create_workspace(
    body: WorkspaceCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkspaceOut:
    ws = Workspace(name=body.name, owner_user_id=user.id)
    db.add(ws)
    db.flush()
    db.add(Membership(workspace_id=ws.id, user_id=user.id, role="owner"))
    db.commit()
    return WorkspaceOut(id=ws.id, name=ws.name, role="owner")


@router.get("", response_model=list[WorkspaceOut])
def list_workspaces(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[WorkspaceOut]:
    rows = db.execute(
        select(Workspace, Membership.role)
        .join(Membership, Membership.workspace_id == Workspace.id)
        .where(Membership.user_id == user.id)
        .order_by(Workspace.created_at)
    ).all()
    return [WorkspaceOut(id=ws.id, name=ws.name, role=role) for ws, role in rows]


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(
    request: Request,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    server_settings: ServerSettings = Depends(get_server_settings),
) -> None:
    """Permanently delete a workspace and ALL its data (right to delete)."""
    scope.require("owner")
    ws_id = scope.workspace.id

    # 1) Physical data: the workspace's vector store and uploaded blobs.
    settings = get_settings()
    db_path = settings.workspace_db_path(ws_id)
    if ws_id != settings.default_workspace:
        shutil.rmtree(db_path.parent, ignore_errors=True)  # data_dir/workspaces/<id>
    else:  # default workspace shares the legacy path; only drop its table dir
        shutil.rmtree(db_path, ignore_errors=True)
    shutil.rmtree(server_settings.blob_dir / ws_id, ignore_errors=True)

    # 2) Relational rows not covered by cascade.
    source_ids = [s.id for s in db.scalars(
        select(Source).where(Source.workspace_id == ws_id)
    ).all()]
    if source_ids:
        db.query(Share).filter(
            Share.resource_type == "source", Share.resource_id.in_(source_ids)
        ).delete(synchronize_session=False)
    db.query(Job).filter(Job.workspace_id == ws_id).delete(synchronize_session=False)
    db.query(GraphSnapshot).filter(
        GraphSnapshot.workspace_id == ws_id
    ).delete(synchronize_session=False)
    db.delete(scope.workspace)  # cascades memberships + sources
    db.commit()

    audit.record(
        db, action="workspace.delete", user_id=scope.user.id, workspace_id=ws_id,
        ip=request.client.host if request.client else "",
    )


@router.get("/{workspace_id}/audit", response_model=list[AuditOut])
def list_audit(
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> list[AuditOut]:
    """Return the access/mutation audit trail for a workspace (admins+)."""
    scope.require("admin")
    rows = db.scalars(
        select(AuditLog)
        .where(AuditLog.workspace_id == scope.workspace.id)
        .order_by(AuditLog.ts.desc())
        .limit(200)
    ).all()
    return [
        AuditOut(
            id=r.id, action=r.action, user_id=r.user_id,
            workspace_id=r.workspace_id, resource=r.resource, ts=r.ts,
        )
        for r in rows
    ]
