"""OIDC single sign-on (optional).

A minimal, dependency-light Authorization-Code flow using httpx: discovery ->
authorization URL -> callback exchanges the code, fetches userinfo, and links or
creates a local user, then mints the same session token the rest of the app
uses. The three network steps are isolated functions so they can be stubbed in
tests and swapped for a library (e.g. authlib) later.
"""

from __future__ import annotations

from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import issue_token
from ..config import ServerSettings, get_server_settings
from ..db import get_db
from ..models import Membership, User, Workspace
from ..schemas import TokenResponse

router = APIRouter(prefix="/auth/oidc", tags=["auth"])


def _discover(issuer: str) -> dict:
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    return httpx.get(url, timeout=10).raise_for_status().json()


def _exchange_code(cfg: dict, code: str, settings: ServerSettings) -> dict:
    resp = httpx.post(
        cfg["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.oidc_redirect_uri,
            "client_id": settings.oidc_client_id,
            "client_secret": settings.oidc_client_secret,
        },
        timeout=10,
    )
    return resp.raise_for_status().json()


def _userinfo(cfg: dict, access_token: str) -> dict:
    resp = httpx.get(
        cfg["userinfo_endpoint"],
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    return resp.raise_for_status().json()


def _get_or_create_user(db: Session, sub: str, email: str, name: str) -> User:
    user = db.scalars(select(User).where(User.oidc_sub == sub)).first()
    if user is None and email:
        user = db.scalars(select(User).where(User.email == email.lower())).first()
    if user is None:
        user = User(email=(email or f"{sub}@oidc").lower(), name=name, oidc_sub=sub)
        db.add(user)
        db.flush()
        ws = Workspace(name="Personal", owner_user_id=user.id)
        db.add(ws)
        db.flush()
        db.add(Membership(workspace_id=ws.id, user_id=user.id, role="owner"))
    elif not user.oidc_sub:
        user.oidc_sub = sub
    db.commit()
    db.refresh(user)
    return user


def _require_enabled(settings: ServerSettings) -> None:
    if not settings.oidc_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "OIDC is not enabled")


@router.get("/login")
def oidc_login(
    state: str = "",
    settings: ServerSettings = Depends(get_server_settings),
) -> dict:
    _require_enabled(settings)
    cfg = _discover(settings.oidc_issuer)
    params = {
        "response_type": "code",
        "client_id": settings.oidc_client_id,
        "redirect_uri": settings.oidc_redirect_uri,
        "scope": settings.oidc_scopes,
        "state": state,
    }
    return {"authorization_url": cfg["authorization_endpoint"] + "?" + urlencode(params)}


@router.get("/callback", response_model=TokenResponse)
def oidc_callback(
    code: str,
    db: Session = Depends(get_db),
    settings: ServerSettings = Depends(get_server_settings),
) -> TokenResponse:
    _require_enabled(settings)
    cfg = _discover(settings.oidc_issuer)
    tokens = _exchange_code(cfg, code, settings)
    info = _userinfo(cfg, tokens.get("access_token", ""))
    sub = info.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OIDC userinfo missing 'sub'")
    user = _get_or_create_user(db, sub, info.get("email", ""), info.get("name", ""))
    return TokenResponse(access_token=issue_token(user.id, settings))
