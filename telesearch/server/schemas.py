"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8)
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    name: str


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class WorkspaceOut(BaseModel):
    id: str
    name: str
    role: str


class SourceOut(BaseModel):
    id: str
    workspace_id: str
    collection_id: str
    kind: str
    name: str
    status: str
    bytes: int
    error: str
    created_at: datetime


class JobOut(BaseModel):
    id: str
    workspace_id: str
    source_id: Optional[str]
    type: str
    state: str
    progress: float
    message: str
    error: str
    created_at: datetime
    updated_at: datetime


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    collections: Optional[list[str]] = None
    modalities: Optional[list[str]] = None
    senders: Optional[list[str]] = None
    source_kinds: Optional[list[str]] = None
    date_from: Optional[int] = None
    date_to: Optional[int] = None
    rerank: Optional[bool] = None


class SearchHit(BaseModel):
    chunk_id: str
    message_id: int
    sender: str
    date_str: str
    modality: str
    content: str
    media_path: Optional[str]
    score: float
    chat: str


class ShareCreate(BaseModel):
    user_email: str
    role: str = "viewer"
