"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

A bi-encoder (bge-m3) embeds the query and documents independently, which is
fast but approximate. A cross-encoder reads ``(query, document)`` *together* and
scores their relevance directly, which is far more precise — but too expensive
to run over the whole index. So we use it the standard way: retrieve a generous
candidate set with the bi-encoder + BM25, then rerank only those candidates.
"""

from __future__ import annotations

from functools import cached_property

from ..config import Settings


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            self.settings.reranker_model,
            device=self.settings.device,
            trust_remote_code=True,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` pairs sorted best-first.

        Indices refer to positions in ``documents`` so the caller can map back
        to its own records.
        """
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            range(len(documents)), key=lambda i: float(scores[i]), reverse=True
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(i, float(scores[i])) for i in ranked]
