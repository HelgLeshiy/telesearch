"""Tests for conversation-window chunking, reply stitching, neighbour fetch
and RAG context expansion (no GPU/models/LLM required)."""

from pathlib import Path

import numpy as np

from telesearch.index.build import (
    _build_conversation_chunks,
    _conversation_line,
    _message_to_chunks,
)
from telesearch.index.store import VectorStore
from telesearch.models import Chunk, Message
from telesearch.search.rag import _format_context, _gather_context
from telesearch.search.retriever import SearchResult


def _msg(i, text="", *, ts=None, sender="Alice", media_type=None, file_name=None,
         reply_to=None):
    return Message(
        id=i,
        chat="c",
        sender=sender,
        timestamp=ts if ts is not None else i * 60,
        date_str=f"2024-01-01 00:{i:02d}",
        text=text,
        reply_to=reply_to,
        media_type=media_type,
        file_name=file_name,
    )


# --------------------------------------------------------------------------- #
# Conversation-window chunking
# --------------------------------------------------------------------------- #
def test_conversation_line_formats_text_and_media():
    assert _conversation_line(_msg(1, "hello")) == "[msg 1] Alice: hello"
    assert _conversation_line(_msg(2, media_type="photo")) == "[msg 2] Alice: [photo]"
    line = _conversation_line(_msg(3, media_type="file", file_name="report.pdf"))
    assert line == "[msg 3] Alice: [file: report.pdf]"
    # Nothing to say -> no line.
    assert _conversation_line(_msg(4)) is None


def test_build_conversation_chunks_windows_and_overlap():
    messages = [_msg(i, f"m{i}") for i in range(1, 7)]
    chunks = _build_conversation_chunks(
        messages, window_size=3, stride=2, max_gap=3600
    )
    assert chunks, "expected at least one window"
    assert all(c.modality == "conversation" for c in chunks)

    first = chunks[0]
    assert first.chunk_id == "1:conversation"
    assert first.message_id == 1
    # Window content carries multiple per-message lines with citable ids.
    assert first.content.splitlines()[0] == "[msg 1] Alice: m1"
    assert len(first.content.splitlines()) == 3
    assert first.extra["message_ids"] == [1, 2, 3]

    # Overlapping windows advance by the stride.
    starts = [c.message_id for c in chunks]
    assert starts[0] == 1 and starts[1] == 3


def test_build_conversation_chunks_splits_on_time_gap():
    # Two bursts separated by a gap larger than max_gap must not be glued.
    messages = [
        _msg(1, "a", ts=0),
        _msg(2, "b", ts=10),
        _msg(3, "c", ts=10_000),  # big gap -> new session
        _msg(4, "d", ts=10_010),
    ]
    chunks = _build_conversation_chunks(
        messages, window_size=5, stride=5, max_gap=3600
    )
    # One window per burst, neither spanning the gap.
    assert len(chunks) == 2
    assert chunks[0].extra["message_ids"] == [1, 2]
    assert chunks[1].extra["message_ids"] == [3, 4]


def test_build_conversation_chunks_disabled_for_size_one():
    assert _build_conversation_chunks(
        [_msg(1, "a"), _msg(2, "b")], window_size=1, stride=1, max_gap=0
    ) == []


# --------------------------------------------------------------------------- #
# Reply stitching
# --------------------------------------------------------------------------- #
def test_reply_parent_is_prepended():
    parent = _msg(1, "should we book the Ritz in Rome?", sender="Bob")
    reply = _msg(2, "yes, book it", reply_to=1)
    lookup = {1: parent, 2: reply}

    chunks = _message_to_chunks(
        reply,
        Path("/nonexistent"),
        captioner=None,
        transcriber=None,
        num_frames=0,
        reply_lookup=lookup,
    )
    text = next(c for c in chunks if c.modality == "text").content
    assert "replying to Bob" in text
    assert "Ritz in Rome" in text
    assert "yes, book it" in text


def test_no_reply_lookup_leaves_text_plain():
    chunks = _message_to_chunks(
        _msg(5, "standalone"),
        Path("/nonexistent"),
        captioner=None,
        transcriber=None,
        num_frames=0,
    )
    assert chunks[0].content == "standalone"


# --------------------------------------------------------------------------- #
# Store neighbour fetch
# --------------------------------------------------------------------------- #
def _chunk(mid, modality="text", content=None):
    return Chunk(
        chunk_id=f"{mid}:{modality}",
        message_id=mid,
        chat="c",
        sender="Alice",
        timestamp=mid * 100,
        date_str="2024-01-01",
        modality=modality,
        content=content or f"message {mid}",
    )


def test_fetch_around_returns_sorted_neighbors(tmp_path):
    dim = 8
    store = VectorStore(tmp_path / "db", dim)
    chunks = [_chunk(i) for i in range(1, 11)]
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((len(chunks), dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add([c.to_row() for c in chunks], vecs)

    rows = store.fetch_around("c", [5], before=2, after=2)
    ids = [r["message_id"] for r in rows]
    assert ids == [3, 4, 5, 6, 7]
    assert all("vector" not in r for r in rows)


def test_fetch_around_filters_modality(tmp_path):
    dim = 8
    store = VectorStore(tmp_path / "db", dim)
    chunks = [_chunk(5, "text"), _chunk(5, "document"), _chunk(6, "text")]
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((len(chunks), dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add([c.to_row() for c in chunks], vecs)

    rows = store.fetch_around("c", [5], before=1, after=1, modalities=("text",))
    assert {r["modality"] for r in rows} == {"text"}
    assert sorted(r["message_id"] for r in rows) == [5, 6]


def test_fetch_around_no_neighbors_disabled(tmp_path):
    store = VectorStore(tmp_path / "db", 8)
    assert store.fetch_around("c", [5], before=0, after=0) == []
    assert store.fetch_around("c", [], before=2, after=2) == []


# --------------------------------------------------------------------------- #
# RAG context expansion
# --------------------------------------------------------------------------- #
class _FakeStore:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def fetch_around(self, chat, message_ids, before, after, modalities=None):
        self.calls.append((chat, tuple(message_ids), before, after))
        return self._rows


def _result(mid, content):
    return SearchResult(
        chunk_id=f"{mid}:text",
        message_id=mid,
        sender="Alice",
        date_str="2024-01-01",
        modality="text",
        content=content,
        media_path=None,
        score=1.0,
        chat="c",
    )


def test_gather_context_merges_and_marks_hits():
    hit = _result(5, "the matched line")
    neighbor_rows = [
        {"chunk_id": "4:text", "message_id": 4, "chat": "c", "sender": "Bob",
         "date_str": "d", "modality": "text", "content": "before"},
        {"chunk_id": "6:text", "message_id": 6, "chat": "c", "sender": "Bob",
         "date_str": "d", "modality": "text", "content": "after"},
        # Duplicate of the hit must not override it.
        {"chunk_id": "5:text", "message_id": 5, "chat": "c", "sender": "Alice",
         "date_str": "d", "modality": "text", "content": "dup"},
    ]
    store = _FakeStore(neighbor_rows)

    merged, hit_ids = _gather_context([hit], store, neighbors=2)

    assert [r.message_id for r in merged] == [4, 5, 6]  # chronological
    assert hit_ids == {"5:text"}
    assert store.calls == [("c", (5,), 2, 2)]
    # The original hit object is preserved (not replaced by the duplicate row).
    assert next(r for r in merged if r.chunk_id == "5:text").content == "the matched line"


def test_gather_context_without_neighbors():
    hit = _result(5, "only this")
    store = _FakeStore([])
    merged, hit_ids = _gather_context([hit], store, neighbors=0)
    assert [r.chunk_id for r in merged] == ["5:text"]
    assert store.calls == []  # neighbour fetch skipped entirely


def test_format_context_includes_message_tags():
    out = _format_context([_result(7, "hello there")])
    assert "[msg 7 | text | Alice | 2024-01-01]" in out
    assert "hello there" in out
