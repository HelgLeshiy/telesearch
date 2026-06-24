"""FastAPI dependencies: authenticated user, workspace access and RBAC.

Access control is centralized here so route handlers receive an already-checked
:class:`AccessScope` (the user, the workspace, their role and the set of
collections they may read) instead of re-deriving permissions ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import verify_token
from .db import get_db
from .models import ROLE_ORDER, Membership, Share, Source, User, Workspace


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    uid = verify_token(token)
    if not uid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")
    user = db.get(User, uid)
    if user is None or user.status != "active":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown user")
    return user


@dataclass
class AccessScope:
    user: User
    workspace: Workspace
    role: str
    collections: list[str] | None  # readable collection ids; None = all in workspace

    def require(self, at_least: str) -> None:
        if ROLE_ORDER.get(self.role, -1) < ROLE_ORDER.get(at_least, 99):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires {at_least} role")


def _membership_role(db: Session, workspace_id: str, user_id: str) -> str | None:
    m = db.scalars(
        select(Membership).where(
            Membership.workspace_id == workspace_id, Membership.user_id == user_id
        )
    ).first()
    return m.role if m else None


def _shared_collections(db: Session, workspace_id: str, user_id: str) -> list[str]:
    """Collection ids in ``workspace_id`` shared directly to ``user_id``."""
    rows = db.scalars(
        select(Share).where(
            Share.principal_type == "user",
            Share.principal_id == user_id,
            Share.resource_type == "source",
        )
    ).all()
    if not rows:
        return []
    ids = {r.resource_id for r in rows}
    sources = db.scalars(
        select(Source).where(
            Source.workspace_id == workspace_id, Source.id.in_(ids)
        )
    ).all()
    return [s.collection_id for s in sources]


def get_scope(
    workspace_id: str = Path(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AccessScope:
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")

    role = _membership_role(db, workspace_id, user.id)
    if role is not None:
        # Members see every collection in the workspace.
        return AccessScope(user=user, workspace=workspace, role=role, collections=None)

    # Non-members may still have individual sources shared to them.
    shared = _shared_collections(db, workspace_id, user.id)
    if shared:
        return AccessScope(
            user=user, workspace=workspace, role="viewer", collections=shared
        )

    raise HTTPException(status.HTTP_403_FORBIDDEN, "no access to this workspace")
