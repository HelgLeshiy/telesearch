"""Tests for multi-store fan-out retrieval (model-free with fakes)."""

import numpy as np

from telesearch.index.store import VectorStore
from telesearch.models import Chunk
from telesearch.search.multi import SearchTarget, fan_out_search


class FakeEmbedder:
    dim = 8

    def encode(self, texts, *, is_query=False, batch_size=64):
        return np.ones((len(texts), self.dim), dtype="float32")


class FakeReranker:
    """Rank by number of shared lowercase words with the query (desc)."""

    def rerank(self, query, documents, top_k=None):
        qwords = set(query.lower().split())
        scored = sorted(
            range(len(documents)),
            key=lambda i: len(qwords & set(documents[i].lower().split())),
            reverse=True,
        )
        if top_k is not None:
            scored = scored[:top_k]
        return [(i, float(len(qwords & set(documents[i].lower().split())))) for i in scored]


def _store(tmp_path, name, chunks):
    store = VectorStore(tmp_path / name, 8)
    vecs = np.ones((len(chunks), 8), dtype="float32")
    store.add([c.to_row() for c in chunks], vecs)
    store.build_fts()
    return store


def _chunk(cid, content, collection_id):
    return Chunk(
        chunk_id=cid, message_id=1, chat="c", sender="s", timestamp=0,
        date_str="", modality="text", content=content, collection_id=collection_id,
    )


def test_fan_out_merges_across_stores(tmp_path):
    a = _store(tmp_path, "a", [_chunk("A:1", "sunset over the beach", "A")])
    b = _store(tmp_path, "b", [_chunk("B:1", "sunset beach cocktail", "B")])

    results = fan_out_search(
        FakeEmbedder(), FakeReranker(),
        [SearchTarget(a), SearchTarget(b)],
        "sunset beach", k=10,
    )
    cids = {r.chunk_id for r in results}
    assert cids == {"A:1", "B:1"}  # results from both stores combined


def test_fan_out_respects_per_target_where(tmp_path):
    a = _store(
        tmp_path, "a",
        [_chunk("A:1", "shared note", "shared"), _chunk("A:2", "private note", "priv")],
    )
    results = fan_out_search(
        FakeEmbedder(), FakeReranker(),
        [SearchTarget(a, where="collection_id = 'shared'")],
        "note", k=10,
    )
    assert {r.chunk_id for r in results} == {"A:1"}  # private collection excluded


def test_fan_out_skips_missing_store(tmp_path):
    missing = VectorStore(tmp_path / "empty", create=False)  # no table
    a = _store(tmp_path, "a", [_chunk("A:1", "hello world", "A")])
    results = fan_out_search(
        FakeEmbedder(), FakeReranker(),
        [SearchTarget(missing), SearchTarget(a)],
        "hello", k=10,
    )
    assert {r.chunk_id for r in results} == {"A:1"}


def test_fan_out_without_rerank_uses_store_score(tmp_path):
    a = _store(tmp_path, "a", [_chunk("A:1", "alpha", "A")])
    results = fan_out_search(
        FakeEmbedder(), None, [SearchTarget(a)], "alpha", k=10, use_rerank=False
    )
    assert results and results[0].chunk_id == "A:1"
