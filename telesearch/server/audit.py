"""Lightweight audit logging for sensitive actions (data access/mutation).

Hosting other people's private conversations makes an access/mutation trail a
requirement, not a nicety (design §3.5). ``record`` appends a row; failures here
must never break the underlying request, so they are swallowed.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from .models import AuditLog


def record(
    db: Session,
    *,
    action: str,
    user_id: str | None = None,
    workspace_id: str | None = None,
    resource: str = "",
    ip: str = "",
) -> None:
    try:
        db.add(
            AuditLog(
                action=action,
                user_id=user_id,
                workspace_id=workspace_id,
                resource=resource,
                ip=ip,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
