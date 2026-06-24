"""Knowledge-graph endpoints: refresh (background), fetch, topic drill-down."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import ServerSettings, get_server_settings
from ..db import get_db
from ..deps import AccessScope, get_scope
from ..models import GraphSnapshot, Job
from ..queue import enqueue
from ..schemas import JobOut

router = APIRouter(prefix="/workspaces/{workspace_id}/graph", tags=["graph"])


def _job_out(j: Job) -> JobOut:
    return JobOut(
        id=j.id, workspace_id=j.workspace_id, source_id=j.source_id, type=j.type,
        state=j.state, progress=j.progress, message=j.message, error=j.error,
        created_at=j.created_at, updated_at=j.updated_at,
    )


@router.post("/refresh", response_model=JobOut)
def refresh_graph(
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> JobOut:
    """Queue a background rebuild of the workspace knowledge graph."""
    scope.require("member")
    collections = scope.collections  # None = whole workspace
    job = enqueue(
        db,
        workspace_id=scope.workspace.id,
        job_type="graph_refresh",
        params={"collections": collections} if collections else {},
    )
    return _job_out(job)


@router.get("")
def get_graph(
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> dict:
    """Return the latest cached graph snapshot (empty graph if none yet)."""
    snap = db.scalars(
        select(GraphSnapshot)
        .where(GraphSnapshot.workspace_id == scope.workspace.id)
        .order_by(GraphSnapshot.created_at.desc())
    ).first()
    if snap is None:
        return {"nodes": [], "edges": [], "meta": {"n_topics": 0, "n_chunks": 0}}
    return snap.data


@router.get("/topics/{topic_id}")
def get_topic(
    topic_id: int,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
) -> dict:
    """Return a topic node with its representative messages."""
    snap = db.scalars(
        select(GraphSnapshot)
        .where(GraphSnapshot.workspace_id == scope.workspace.id)
        .order_by(GraphSnapshot.created_at.desc())
    ).first()
    if snap is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no graph built yet")
    for node in snap.data.get("nodes", []):
        if node.get("id") == topic_id:
            return node
    raise HTTPException(status.HTTP_404_NOT_FOUND, "topic not found")


@router.post("/g3")
def build_g3(
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    server_settings: ServerSettings = Depends(get_server_settings),
) -> dict:
    """EXPERIMENTAL: build a GraphRAG-style fact graph (requires an LLM endpoint)."""
    scope.require("member")
    if not server_settings.graph_g3_enabled:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "G3 fact graph is experimental and disabled (set TELESEARCH_GRAPH_G3_ENABLED=true)",
        )
    from openai import OpenAI

    from ...config import get_settings
    from ...graph import build_fact_graph
    from ...index.store import VectorStore

    es = get_settings()
    store = VectorStore(es.workspace_db_path(scope.workspace.id), create=False)
    rows = store.fetch_all(scope.collections)
    client = OpenAI(base_url=es.llm_base_url, api_key=es.llm_api_key,
                    timeout=es.llm_request_timeout, max_retries=es.llm_max_retries)

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=es.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return build_fact_graph(rows, complete)
