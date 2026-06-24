"""Workspace management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Membership, User, Workspace
from ..schemas import WorkspaceCreate, WorkspaceOut

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
