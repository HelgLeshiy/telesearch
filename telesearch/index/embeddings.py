"""Text embedding via sentence-transformers, running locally on the GPU.

We embed all searchable text (messages, image captions, video summaries and
transcripts) into a single shared vector space. Keeping everything in one text
space means a query like "the receipt photo from the restaurant" matches an
image caption just as well as it matches a typed message.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from ..config import Settings


class TextEmbedder:
    """Wrapper around a sentence-transformers embedding model."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.settings.text_embed_model,
            device=self.settings.device,
            trust_remote_code=True,
        )

    @cached_property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Return L2-normalized embeddings for ``texts``.

        ``is_query`` lets instruction-tuned models (e.g. bge-m3) use the
        appropriate prompt for queries vs. documents when available.
        """
        prompt_name = "query" if is_query else None
        kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        # Not every model defines named prompts; fall back gracefully.
        try:
            return self._model.encode(texts, prompt_name=prompt_name, **kwargs)
        except (ValueError, KeyError):
            return self._model.encode(texts, **kwargs)
