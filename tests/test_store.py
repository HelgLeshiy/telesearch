"""Tests for the LanceDB hybrid store (no GPU/models required)."""

import numpy as np

from telesearch.index.store import VectorStore
from telesearch.models import Chunk


def _chunk(cid, mid, sender, modality, content):
    return Chunk(
        chunk_id=cid,
        message_id=mid,
        chat="c",
        sender=sender,
        timestamp=mid * 100,
        date_str="2024-01-01",
        modality=modality,
        content=content,
    )


def test_hybrid_search(tmp_path):
    dim = 8
    store = VectorStore(tmp_path / "db", dim)
    rng = np.random.default_rng(0)

    chunks = [
        _chunk("1:text", 1, "Alice", "text", "sunset over the ocean beach"),
        _chunk("2:image", 2, "Bob", "image", "a photo of snowy mountains"),
        _chunk("3:text", 3, "Alice", "text", "dinner receipt from the restaurant"),
    ]
    vecs = rng.standard_normal((3, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    store.add([c.to_row() for c in chunks], vecs)
    store.build_fts()

    assert store.count() == 3

    # Keyword + vector both point at the sunset chunk.
    results = store.hybrid_search("sunset beach", vecs[0], k=3)
    assert results[0]["chunk_id"] == "1:text"
    assert "score" in results[0]
