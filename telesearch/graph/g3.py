"""Experimental G3 fact graph (GraphRAG-style entity/relation extraction).

EXPERIMENTAL (design §7.5 / Phase 4). Unlike the G1+G2 topic graph, this asks an
LLM to pull entities and relations out of chunks, then aggregates them into a
fact graph (entity nodes sized by mentions, edges = extracted relations). It is
costly (per-chunk LLM calls) and noisy on casual chat, so it is off by default
and intended for measured evaluation, not as a promised feature.

The extractor is injected as ``complete(prompt) -> str`` so the engine is
testable with a stub and independent of any specific LLM client.
"""

from __future__ import annotations

import json
import re
from typing import Callable

_PROMPT = (
    "Extract the key entities and the relations between them from the chat "
    "excerpt below. Respond with STRICT JSON of the form "
    '{"entities": ["..."], "relations": [{"source": "...", "target": "...", '
    '"type": "..."}]}. Use short canonical entity names. Excerpt:\n\n'
)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return {}
        return {}


def build_fact_graph(
    rows: list[dict],
    complete: Callable[[str], str],
    *,
    max_chunks: int = 40,
    batch_chars: int = 1500,
) -> dict:
    """Build an entity/relation graph from chunk ``rows`` using ``complete``."""
    items = [r for r in rows if (r.get("content") or "").strip()][:max_chunks]
    meta = {"experimental": True, "n_chunks": len(items)}
    if not items:
        return {"nodes": [], "edges": [], "meta": meta}

    # Batch excerpts to bound the number of LLM calls.
    batches: list[str] = []
    buf = ""
    for r in items:
        piece = r["content"].strip()
        if len(buf) + len(piece) > batch_chars and buf:
            batches.append(buf)
            buf = ""
        buf += piece + "\n"
    if buf:
        batches.append(buf)

    mentions: dict[str, int] = {}
    rel_counts: dict[tuple[str, str, str], int] = {}
    for batch in batches:
        raw = complete(_PROMPT + batch)
        data = _parse_json(raw)
        for ent in data.get("entities", []) or []:
            name = str(ent).strip()
            if name:
                mentions[name] = mentions.get(name, 0) + 1
        for rel in data.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            s = str(rel.get("source", "")).strip()
            t = str(rel.get("target", "")).strip()
            ty = str(rel.get("type", "related")).strip() or "related"
            if s and t:
                mentions.setdefault(s, 1)
                mentions.setdefault(t, 1)
                rel_counts[(s, t, ty)] = rel_counts.get((s, t, ty), 0) + 1

    names = sorted(mentions, key=lambda n: mentions[n], reverse=True)
    idx = {n: i for i, n in enumerate(names)}
    nodes = [
        {"id": idx[n], "label": n, "size": mentions[n]} for n in names
    ]
    edges = [
        {"source": idx[s], "target": idx[t], "type": ty, "weight": c}
        for (s, t, ty), c in rel_counts.items()
        if s in idx and t in idx
    ]
    meta["n_entities"] = len(nodes)
    return {"nodes": nodes, "edges": edges, "meta": meta}
