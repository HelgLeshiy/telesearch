"""Hybrid retrieval over the indexed conversation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..index.embeddings import TextEmbedder
from ..index.store import VectorStore
from .reranker import Reranker


@dataclass
class SearchResult:
    chunk_id: str
    message_id: int
    sender: str
    date_str: str
    modality: str
    content: str
    media_path: Optional[str]
    score: float
    chat: str = ""

    @classmethod
    def from_row(cls, row: dict) -> "SearchResult":
        return cls(
            chunk_id=row["chunk_id"],
            message_id=row["message_id"],
            sender=row.get("sender", ""),
            date_str=row.get("date_str", ""),
            modality=row.get("modality", ""),
            content=row.get("content", ""),
            media_path=row.get("media_path") or None,
            score=float(row.get("score", row.get("_relevance_score", 0.0) or 0.0)),
            chat=row.get("chat", ""),
        )


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = TextEmbedder(settings)
        self.store = VectorStore(settings.db_path, self.embedder.dim)
        self._reranker = Reranker(settings) if settings.use_reranker else None

    def search(
        self,
        query: str,
        k: int = 10,
        modality: Optional[str] = None,
        rerank: Optional[bool] = None,
        rerank_query: Optional[str] = None,
    ) -> list[SearchResult]:
        """Hybrid retrieve, then (optionally) cross-encoder rerank.

        Pipeline: bge-m3 dense + BM25 -> RRF candidates -> modality filter ->
        bge-reranker-v2-m3 -> top ``k``.

        ``rerank_query`` lets the caller retrieve candidates with one query
        (e.g. a HyDE-expanded one for better recall) but rerank them against a
        different, sharper query (e.g. the user's original question).
        """
        use_rerank = self.settings.use_reranker if rerank is None else rerank
        rerank_query = rerank_query or query

        query_vec = self.embedder.encode([query], is_query=True)[0]
        # Over-fetch a generous candidate pool for the reranker to sift.
        candidates = self.settings.rerank_candidates if use_rerank else max(k * 4, k)
        rows = self.store.hybrid_search(query, query_vec, k=candidates)
        results = [SearchResult.from_row(r) for r in rows]

        if modality:
            results = [r for r in results if r.modality == modality]

        if use_rerank and self._reranker is not None and results:
            ranked = self._reranker.rerank(
                rerank_query, [r.content for r in results], top_k=k
            )
            reranked: list[SearchResult] = []
            for idx, score in ranked:
                r = results[idx]
                r.score = score
                reranked.append(r)
            return reranked

        return results[:k]
