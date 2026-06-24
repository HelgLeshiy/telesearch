"""ORM models: users, workspaces, membership/ACL, sources, jobs, audit.

These define the relational backbone of the multi-user service. Vector data
itself lives outside this database, physically partitioned per workspace by the
engine (``Settings.workspace_db_path``); a :class:`Source` row corresponds to a
``collection_id`` inside that workspace's vector store.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Roles, highest privilege first. Used by RBAC checks.
ROLE_ORDER = {"viewer": 0, "member": 1, "admin": 2, "owner": 3}


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[str] = mapped_column(String(256), default="")
    oidc_sub: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    plan: Mapped[str] = mapped_column(String(32), default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memberships: Mapped[list["Membership"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    sources: Mapped[list["Source"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id", name="uq_member"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), default="member")

    workspace: Mapped[Workspace] = relationship(back_populates="memberships")
    user: Mapped[User] = relationship(back_populates="memberships")


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    # Logical collection id inside the workspace's vector store.
    collection_id: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(64), default="")  # parser kind, "" = auto
    name: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    blob_key: Mapped[str] = mapped_column(String(512), default="")
    error: Mapped[str] = mapped_column(String(2000), default="")
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    workspace: Mapped[Workspace] = relationship(back_populates="sources")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("sources.id"), nullable=True, index=True
    )
    type: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(500), default="")
    error: Mapped[str] = mapped_column(String(2000), default="")
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Share(Base):
    __tablename__ = "shares"
    __table_args__ = (
        UniqueConstraint(
            "resource_type", "resource_id", "principal_type", "principal_id",
            name="uq_share",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    resource_type: Mapped[str] = mapped_column(String(32))  # workspace|source
    resource_id: Mapped[str] = mapped_column(String(32), index=True)
    principal_type: Mapped[str] = mapped_column(String(32))  # user
    principal_id: Mapped[str] = mapped_column(String(32), index=True)
    role: Mapped[str] = mapped_column(String(32), default="viewer")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SavedSearch(Base):
    """A reusable, named search preset (filters + selected contexts) per user."""

    __tablename__ = "saved_searches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(128))
    resource: Mapped[str] = mapped_column(String(256), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
