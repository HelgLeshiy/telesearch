"""Authentication primitives: password hashing and signed session tokens.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib, no extra dependency).
Session tokens are signed, expiring tokens (``itsdangerous``); they carry only
the user id and are verified on every request. This module is transport- and
framework-agnostic; FastAPI wiring lives in ``deps.py``.

An OIDC provider can be layered on top by minting the same session token after
an external login — see ``issue_token``; the rest of the app is unaffected.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import ServerSettings, get_server_settings

_PBKDF2_ROUNDS = 240_000
_SALT_BYTES = 16
_TOKEN_SALT = "telesearch.session"


def hash_password(password: str) -> str:
    """Return ``pbkdf2_sha256$rounds$salt_hex$hash_hex``."""
    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verification of a password against an encoded hash."""
    try:
        algo, rounds_s, salt_hex, hash_hex = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        rounds = int(rounds_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, rounds)
    return hmac.compare_digest(digest, expected)


def _serializer(settings: ServerSettings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key, salt=_TOKEN_SALT)


def issue_token(user_id: str, settings: ServerSettings | None = None) -> str:
    settings = settings or get_server_settings()
    return _serializer(settings).dumps({"uid": user_id})


def verify_token(token: str, settings: ServerSettings | None = None) -> str | None:
    """Return the user id encoded in ``token`` or ``None`` if invalid/expired."""
    settings = settings or get_server_settings()
    try:
        data = _serializer(settings).loads(token, max_age=settings.token_ttl_seconds)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict):
        return None
    return data.get("uid")
