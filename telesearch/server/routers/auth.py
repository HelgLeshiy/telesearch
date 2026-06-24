"""Authentication endpoints: register, login, me."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import hash_password, issue_token, verify_password
from ..config import ServerSettings, get_server_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import Membership, User, Workspace
from ..schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


def _create_user_with_workspace(db: Session, email: str, name: str, password: str) -> User:
    user = User(email=email.lower(), name=name, password_hash=hash_password(password))
    db.add(user)
    db.flush()
    ws = Workspace(name="Personal", owner_user_id=user.id)
    db.add(ws)
    db.flush()
    db.add(Membership(workspace_id=ws.id, user_id=user.id, role="owner"))
    db.commit()
    db.refresh(user)
    return user


@router.post("/register", response_model=TokenResponse)
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> TokenResponse:
    if not settings.allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "registration disabled")
    if not _valid_email(body.email):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid email")
    existing = db.scalars(select(User).where(User.email == body.email.lower())).first()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = _create_user_with_workspace(db, body.email, body.name, body.password)
    return TokenResponse(access_token=issue_token(user.id, settings))


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> TokenResponse:
    user = db.scalars(select(User).where(User.email == body.email.lower())).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    return TokenResponse(access_token=issue_token(user.id, settings))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(id=user.id, email=user.email, name=user.name)
