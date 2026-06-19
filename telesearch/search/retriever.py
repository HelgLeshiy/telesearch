"""Hybrid retrieval over the indexed conversation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..index.embeddings import TextEmbedder
from ..index.store import VectorStore


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
        )


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = TextEmbedder(settings)
        self.store = VectorStore(settings.db_path, self.embedder.dim)

    def search(
        self,
        query: str,
        k: int = 10,
        modality: Optional[str] = None,
    ) -> list[SearchResult]:
        query_vec = self.embedder.encode([query], is_query=True)[0]
        # Over-fetch so modality filtering still returns enough results.
        fetch_k = k * 4 if modality else k
        rows = self.store.hybrid_search(query, query_vec, k=fetch_k)
        results = [SearchResult.from_row(r) for r in rows]
        if modality:
            results = [r for r in results if r.modality == modality]
        return results[:k]
