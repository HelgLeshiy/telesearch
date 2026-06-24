"""Vector store interface.

Retrieval and indexing depend on this protocol rather than on LanceDB directly,
so the storage engine can be swapped (e.g. Qdrant / pgvector / managed Lance)
as scale demands without touching the pipeline or search code. The current
implementation is :class:`telesearch.index.store.VectorStore`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal contract every vector store implementation must satisfy."""

    def add(self, rows: list[dict[str, Any]], vectors: np.ndarray) -> None: ...

    def build_fts(self) -> None: ...

    def count(self) -> int: ...

    def existing_message_ids(self) -> set[int]: ...

    def drop(self) -> None: ...

    def delete_modalities(self, modalities: list[str] | tuple[str, ...]) -> int: ...

    def hybrid_search(
        self,
        query_text: str,
        query_vec: np.ndarray,
        k: int = 10,
        candidates: int = 50,
        rrf_k: int = 60,
        where: str | None = None,
    ) -> list[dict]: ...

    def fetch_around(
        self,
        chat: str,
        message_ids: list[int],
        before: int,
        after: int,
        modalities: tuple[str, ...] | None = None,
        limit: int = 2000,
    ) -> list[dict]: ...
