"""Server configuration (separate from the engine's model/device Settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import get_settings


class ServerSettings(BaseSettings):
    """HTTP-service settings, read from ``TELESEARCH_*`` env vars / ``.env``.

    Defaults are deliberately portable so the service runs with no external
    dependencies; point ``database_url`` at Postgres and ``blob_backend`` at S3
    for production.
    """

    model_config = SettingsConfigDict(
        env_prefix="TELESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Persistence. SQLite by default; e.g. postgresql+psycopg://user:pw@host/db
    database_url: str = ""

    # Session signing. MUST be overridden in production (a stable random secret).
    secret_key: str = "dev-insecure-change-me"
    token_ttl_seconds: int = 7 * 24 * 3600

    # Blob storage for uploads: "local" (filesystem) or "s3".
    blob_backend: str = "local"
    s3_bucket: str = ""
    s3_endpoint_url: str = ""

    # Background worker: run jobs in an in-process thread by default.
    worker_inline: bool = True
    worker_poll_seconds: float = 1.0

    # Open self-service registration (set False to lock down to invites/OIDC).
    allow_registration: bool = True

    # CORS origins for a separately-hosted frontend ("*" for any; "" to disable).
    cors_origins: str = ""

    # Default media flags for upload-triggered indexing. Media (VLM/Whisper) is
    # off by default because it needs a GPU; the cheap text path runs anywhere.
    index_media_by_default: bool = False

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_file = get_settings().data_dir / "telesearch.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_file}"

    @property
    def blob_dir(self) -> Path:
        return get_settings().data_dir / "blobs"

    @property
    def is_dev_secret(self) -> bool:
        return self.secret_key == "dev-insecure-change-me"


@lru_cache
def get_server_settings() -> ServerSettings:
    return ServerSettings()
