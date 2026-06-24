"""Global, cross-context search: combine all sources a user can access.

Searches every workspace the user belongs to plus any individually-shared
sources that live in other workspaces, fused into one ranked list. This realizes
"combining/recombining contexts" from the design: pick none (everything),
specific workspaces (``workspace_ids``) or specific sources (``collections``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import Settings, get_settings
from ...index.store import VectorStore
from ...search.multi import SearchTarget, fan_out_search
from ...service import SearchQuery, build_where
from ..db import get_db
from ..deps import get_current_user
from ..models import Membership, Share, Source, User
from ..schemas import GlobalSearchRequest, SearchHit
from .search import get_shared_models

router = APIRouter(prefix="/search", tags=["search"])


def _lit(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _combine(*clauses: str | None) -> str | None:
    parts = [c for c in clauses if c]
    return " AND ".join(parts) if parts else None


def _base_where(body: GlobalSearchRequest) -> str | None:
    return build_where(
        SearchQuery(
            text=body.query,
            collections=body.collections,
            date_from=body.date_from,
            date_to=body.date_to,
            modalities=body.modalities,
            senders=body.senders,
            source_kinds=body.source_kinds,
        )
    )


@router.post("", response_model=list[SearchHit])
def global_search(
    body: GlobalSearchRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[SearchHit]:
    base_where = _base_where(body)
    ws_filter = set(body.workspace_ids) if body.workspace_ids else None

    member_ws = set(
        db.scalars(
            select(Membership.workspace_id).where(Membership.user_id == user.id)
        ).all()
    )

    targets: list[SearchTarget] = []
    used_ws = set()
    for ws_id in member_ws:
        if ws_filter and ws_id not in ws_filter:
            continue
        store = VectorStore(settings.workspace_db_path(ws_id), create=False)
        targets.append(SearchTarget(store=store, where=base_where))
        used_ws.add(ws_id)

    # Sources shared directly to this user that live in workspaces they are not a
    # member of (member workspaces are already fully covered above).
    shared_ids = set(
        db.scalars(
            select(Share.resource_id).where(
                Share.principal_type == "user",
                Share.principal_id == user.id,
                Share.resource_type == "source",
            )
        ).all()
    )
    if shared_ids:
        shared_sources = db.scalars(
            select(Source).where(Source.id.in_(shared_ids))
        ).all()
        by_ws: dict[str, list[str]] = {}
        for s in shared_sources:
            if s.workspace_id in member_ws:
                continue
            if ws_filter and s.workspace_id not in ws_filter:
                continue
            by_ws.setdefault(s.workspace_id, []).append(s.collection_id)
        for ws_id, coll_ids in by_ws.items():
            in_clause = f"collection_id IN ({', '.join(_lit(c) for c in coll_ids)})"
            store = VectorStore(settings.workspace_db_path(ws_id), create=False)
            targets.append(SearchTarget(store=store, where=_combine(base_where, in_clause)))

    if not targets:
        return []

    embedder, reranker = get_shared_models(settings)
    use_rerank = settings.use_reranker if body.rerank is None else body.rerank
    results = fan_out_search(
        embedder,
        reranker,
        targets,
        body.query,
        k=body.k,
        use_rerank=use_rerank,
    )
    return [
        SearchHit(
            chunk_id=r.chunk_id,
            message_id=r.message_id,
            sender=r.sender,
            date_str=r.date_str,
            modality=r.modality,
            content=r.content,
            media_path=r.media_path,
            score=r.score,
            chat=r.chat,
        )
        for r in results
    ]
