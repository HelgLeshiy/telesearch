"""Multi-store fan-out retrieval.

Combine results across several vector stores (e.g. several workspaces, or a
workspace plus individually-shared collections living in other workspaces): query
each target with one shared query embedding, union the candidate pools (chunk
ids are globally unique because they are namespaced by the source's collection
id), then run a single cross-encoder rerank over the union so the final ranking
is consistent regardless of how many sources were combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..index.embeddings import TextEmbedder
from ..index.store import VectorStore
from .reranker import Reranker
from .retriever import SearchResult


@dataclass
class SearchTarget:
    """One store to query, with an optional prefilter for that store."""

    store: VectorStore
    where: Optional[str] = None


def fan_out_search(
    embedder: TextEmbedder,
    reranker: Optional[Reranker],
    targets: list[SearchTarget],
    query: str,
    *,
    k: int = 10,
    candidates: int = 50,
    use_rerank: bool = True,
    rerank_query: Optional[str] = None,
) -> list[SearchResult]:
    """Search ``targets`` and return the merged, reranked top-``k`` results."""
    rerank_query = rerank_query or query
    query_vec = embedder.encode([query], is_query=True)[0]

    merged: dict[str, SearchResult] = {}
    for target in targets:
        if target.store is None or target.store.table is None:
            continue
        rows = target.store.hybrid_search(
            query, query_vec, k=candidates, where=target.where
        )
        for row in rows:
            sr = SearchResult.from_row(row)
            prev = merged.get(sr.chunk_id)
            if prev is None or sr.score > prev.score:
                merged[sr.chunk_id] = sr

    results = sorted(merged.values(), key=lambda r: r.score, reverse=True)

    if use_rerank and reranker is not None and results:
        ranked = reranker.rerank(rerank_query, [r.content for r in results], top_k=k)
        out: list[SearchResult] = []
        for idx, score in ranked:
            r = results[idx]
            r.score = score
            out.append(r)
        return out

    return results[:k]
