"""Prometheus-style metrics endpoint (no external OTel/Prometheus deps).

Exposes basic service gauges scraped from the database. A full OpenTelemetry
setup (traces/metrics exporters) can be layered on without changing this; this
endpoint gives operators immediate visibility with zero infrastructure.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Job, Source, User, Workspace

router = APIRouter(tags=["meta"])


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(db: Session = Depends(get_db)) -> str:
    lines: list[str] = []

    def gauge(name: str, value, help_: str, labels: str = "") -> None:
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{{{labels}}} {value}" if labels else f"{name} {value}")

    gauge("telesearch_users_total", db.scalar(select(func.count(User.id))) or 0, "Users")
    gauge("telesearch_workspaces_total",
          db.scalar(select(func.count(Workspace.id))) or 0, "Workspaces")
    gauge("telesearch_sources_total",
          db.scalar(select(func.count(Source.id))) or 0, "Sources")

    by_state = db.execute(
        select(Job.state, func.count(Job.id)).group_by(Job.state)
    ).all()
    lines.append("# HELP telesearch_jobs Jobs by state")
    lines.append("# TYPE telesearch_jobs gauge")
    for state, count in by_state:
        lines.append(f'telesearch_jobs{{state="{state}"}} {count}')

    return "\n".join(lines) + "\n"
