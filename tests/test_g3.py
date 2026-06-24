"""Tests for the experimental G3 fact-graph builder (stubbed LLM)."""

from telesearch.graph import build_fact_graph


def _rows(texts):
    return [{"chunk_id": f"c{i}", "content": t} for i, t in enumerate(texts)]


def test_build_fact_graph_aggregates_entities_and_relations():
    calls = {"n": 0}

    def fake_complete(prompt):
        calls["n"] += 1
        return (
            '{"entities": ["Alice", "Rome", "Ritz Hotel"], '
            '"relations": [{"source": "Alice", "target": "Ritz Hotel", "type": "booked"}, '
            '{"source": "Ritz Hotel", "target": "Rome", "type": "located_in"}]}'
        )

    rows = _rows(["Alice booked the Ritz in Rome", "check-in at 3pm"])
    g = build_fact_graph(rows, fake_complete, batch_chars=10)

    labels = {n["label"] for n in g["nodes"]}
    assert {"Alice", "Rome", "Ritz Hotel"} <= labels
    assert g["meta"]["experimental"] is True
    assert len(g["edges"]) >= 2
    # Edge endpoints reference valid node ids.
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_build_fact_graph_handles_bad_json():
    g = build_fact_graph(_rows(["hi"]), lambda p: "not json at all")
    assert g["nodes"] == [] and g["edges"] == []


def test_build_fact_graph_empty():
    g = build_fact_graph([], lambda p: "{}")
    assert g["meta"]["n_chunks"] == 0
