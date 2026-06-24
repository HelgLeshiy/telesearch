"""Search service: structured queries with filters over a workspace's index.

Builds a LanceDB prefilter expression from a :class:`SearchQuery` (date range,
modalities, senders, source kinds, collection scope) and runs hybrid retrieval +
rerank through :class:`telesearch.search.Retriever`, scoped to the caller's
workspace. Also exposes RAG (``ask``) over the same scoped retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..search.retriever import Retriever, SearchResult
from .context import RequestContext


@dataclass
class SearchQuery:
    """A structured search request."""

    text: str
    collections: Optional[list[str]] = None
    date_from: Optional[int] = None  # unix seconds, inclusive
    date_to: Optional[int] = None  # unix seconds, inclusive
    modalities: Optional[list[str]] = None
    senders: Optional[list[str]] = None
    source_kinds: Optional[list[str]] = None
    k: int = 10
    rerank: Optional[bool] = None
    extra_where: Optional[str] = None


def _lit(value: str) -> str:
    """Quote a string literal for a LanceDB/SQL filter (escape single quotes)."""
    return "'" + str(value).replace("'", "''") + "'"


def _in_clause(column: str, values: list[str]) -> str:
    return f"{column} IN ({', '.join(_lit(v) for v in values)})"


def build_where(query: SearchQuery, ctx: Optional[RequestContext] = None) -> Optional[str]:
    """Compose the prefilter for ``query`` within ``ctx``'s access scope.

    The effective collection scope is the intersection of what the query asks
    for and what the context allows. Returns ``None`` when there are no
    constraints (search everything in scope).
    """
    clauses: list[str] = []

    allowed = ctx.collections if ctx else None
    requested = query.collections
    if allowed is not None and requested is not None:
        scope = [c for c in requested if c in allowed]
    else:
        scope = requested if requested is not None else allowed
    if scope is not None:
        if not scope:
            # Asked for collections entirely outside the allowed set -> match none.
            return "collection_id IN ()"
        clauses.append(_in_clause("collection_id", scope))

    if query.date_from is not None:
        clauses.append(f"timestamp >= {int(query.date_from)}")
    if query.date_to is not None:
        clauses.append(f"timestamp <= {int(query.date_to)}")
    if query.modalities:
        clauses.append(_in_clause("modality", query.modalities))
    if query.senders:
        clauses.append(_in_clause("sender", query.senders))
    if query.source_kinds:
        clauses.append(_in_clause("source_kind", query.source_kinds))
    if query.extra_where:
        clauses.append(f"({query.extra_where})")

    if not clauses:
        return None
    return " AND ".join(clauses)


class SearchService:
    """Workspace-scoped search + RAG over a single shared retriever."""

    def __init__(
        self,
        settings: Settings,
        ctx: Optional[RequestContext] = None,
        *,
        retriever: Optional[Retriever] = None,
    ):
        self.settings = settings
        self.ctx = ctx or RequestContext.default()
        self.retriever = retriever or Retriever(
            settings, db_path=settings.workspace_db_path(self.ctx.workspace_id)
        )

    def search(self, query: SearchQuery) -> list[SearchResult]:
        where = build_where(query, self.ctx)
        return self.retriever.search(
            query.text, k=query.k, rerank=query.rerank, where=where
        )

    def ask(self, question: str, **kwargs):
        from ..search import answer_question

        return answer_question(
            question, self.settings, retriever=self.retriever, **kwargs
        )
