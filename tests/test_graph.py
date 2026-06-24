"""Tests for the knowledge-graph builder (numpy + sklearn, no embedding models)."""

import numpy as np

from telesearch.graph import GraphParams, build_graph


def _rows(n, axis, words, dim=8, start=0, seed=0):
    """Make n chunk rows clustered along one axis with given keywords."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        v = rng.normal(0, 0.05, dim).astype("float32")
        v[axis] += 1.0
        rows.append({
            "chunk_id": f"c{start + i}",
            "content": f"{words} message number {start + i}",
            "timestamp": 1700000000 + (start + i) * 60,
            "vector": v.tolist(),
        })
    return rows


def test_build_graph_finds_two_topics():
    rows = _rows(5, 0, "alpha apple orchard", start=0, seed=1)
    rows += _rows(5, 1, "beta banana jungle", start=5, seed=2)

    graph = build_graph(rows, params=GraphParams(min_cluster_size=3))
    assert graph["meta"]["n_chunks"] == 10
    assert graph["meta"]["n_topics"] >= 2

    # Every node has a size, coordinates and keywords.
    for node in graph["nodes"]:
        assert node["size"] >= 1
        assert "x" in node and "y" in node
        assert isinstance(node["keywords"], list)

    # The two distinctive vocabularies should each surface in some topic.
    all_kw = " ".join(kw for node in graph["nodes"] for kw in node["keywords"])
    assert "alpha" in all_kw or "apple" in all_kw or "orchard" in all_kw
    assert "beta" in all_kw or "banana" in all_kw or "jungle" in all_kw


def test_build_graph_empty():
    g = build_graph([], params=GraphParams())
    assert g["nodes"] == [] and g["edges"] == [] and g["meta"]["n_chunks"] == 0


def test_build_graph_single_chunk():
    rows = [{"chunk_id": "x", "content": "lonely note", "timestamp": 1, "vector": [1, 0, 0, 0]}]
    g = build_graph(rows, params=GraphParams())
    assert g["meta"]["n_topics"] == 1
    assert g["nodes"][0]["size"] == 1


def test_build_graph_edges_between_related_topics():
    # Three groups; with a low threshold we expect at least one edge.
    rows = _rows(4, 0, "alpha", start=0, seed=1)
    rows += _rows(4, 1, "beta", start=4, seed=2)
    rows += _rows(4, 2, "gamma", start=8, seed=3)
    g = build_graph(rows, params=GraphParams(min_cluster_size=3, edge_threshold=-1.0))
    assert len(g["edges"]) >= 1
    for e in g["edges"]:
        assert "source" in e and "target" in e and "weight" in e


def test_params_hash_is_stable_and_collection_sensitive():
    p = GraphParams()
    assert p.hash(["a", "b"]) == p.hash(["b", "a"])  # order-independent
    assert p.hash(["a"]) != p.hash(["a", "b"])
