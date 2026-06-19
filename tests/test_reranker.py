"""Test reranker ordering logic without loading the actual model."""

from telesearch.config import Settings
from telesearch.search.reranker import Reranker


class _DummyModel:
    """Returns a score = length of the document, for deterministic ordering."""

    def predict(self, pairs, show_progress_bar=False):
        return [float(len(doc)) for _q, doc in pairs]


def test_rerank_orders_and_truncates():
    reranker = Reranker(Settings())
    # Inject the dummy model in place of the cached_property's lazy load.
    reranker.__dict__["_model"] = _DummyModel()

    docs = ["short", "a much longer document", "medium one"]
    ranked = reranker.rerank("q", docs, top_k=2)

    assert [i for i, _ in ranked] == [1, 2]  # longest, then medium
    assert ranked[0][1] >= ranked[1][1]


def test_rerank_empty():
    reranker = Reranker(Settings())
    assert reranker.rerank("q", []) == []
