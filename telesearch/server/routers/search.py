"""Scoped search + RAG endpoints."""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...config import Settings, get_settings
from ...search.retriever import Retriever
from ...service import RequestContext, SearchQuery, SearchService
from ..db import get_db
from ..deps import AccessScope, get_scope
from ..schemas import SearchHit, SearchRequest

router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["search"])

# Share the heavy embedder/reranker across all requests (load once per process).
# The vector store is (re)opened per request so searches always see freshly
# indexed data written by the background worker on a separate connection.
_lock = threading.Lock()
_shared_embedder = None
_shared_reranker = None


def get_shared_models(settings: Settings):
    """Return the process-wide shared (embedder, reranker), loading once."""
    global _shared_embedder, _shared_reranker
    with _lock:
        if _shared_embedder is None:
            from ...index.embeddings import TextEmbedder
            from ...search.reranker import Reranker

            _shared_embedder = TextEmbedder(settings)
            _shared_reranker = Reranker(settings) if settings.use_reranker else None
    return _shared_embedder, _shared_reranker


def _get_retriever(settings: Settings, db_path: str) -> Retriever:
    embedder, reranker = get_shared_models(settings)
    return Retriever(settings, db_path=db_path, embedder=embedder, reranker=reranker)


def _scoped_service(scope: AccessScope, settings: Settings) -> SearchService:
    ctx = RequestContext(
        workspace_id=scope.workspace.id,
        user_id=scope.user.id,
        role=scope.role,
        collections=scope.collections,
    )
    db_path = str(settings.workspace_db_path(scope.workspace.id))
    retriever = _get_retriever(settings, db_path)
    return SearchService(settings, ctx, retriever=retriever)


@router.post("/search", response_model=list[SearchHit])
def search(
    body: SearchRequest,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[SearchHit]:
    service = _scoped_service(scope, settings)
    results = service.search(
        SearchQuery(
            text=body.query,
            k=body.k,
            collections=body.collections,
            modalities=body.modalities,
            senders=body.senders,
            source_kinds=body.source_kinds,
            date_from=body.date_from,
            date_to=body.date_to,
            rerank=body.rerank,
        )
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


@router.post("/ask")
def ask(
    body: SearchRequest,
    scope: AccessScope = Depends(get_scope),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    service = _scoped_service(scope, settings)
    try:
        answer, sources = service.ask(body.query, k=body.k)
    except Exception as exc:  # LLM server typically unavailable without a GPU
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"answer synthesis unavailable: {exc}",
        ) from exc
    return {
        "answer": answer,
        "sources": [
            {
                "message_id": s.message_id,
                "modality": s.modality,
                "sender": s.sender,
                "date_str": s.date_str,
                "content": s.content,
            }
            for s in sources
        ],
    }
