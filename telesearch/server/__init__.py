"""Multi-user HTTP service for telesearch (Phase 1: service backbone).

Turns the single-user engine into a small web service: authentication, per-user
workspaces with isolated data, file uploads, background indexing jobs, and
scoped search — all behind a FastAPI app. Storage backends are pluggable with
portable defaults (SQLite + local blob storage + an in-process worker) so the
service runs anywhere; Postgres / S3 / external auth are enabled via config.
"""

from .config import ServerSettings, get_server_settings

__all__ = ["ServerSettings", "get_server_settings"]
