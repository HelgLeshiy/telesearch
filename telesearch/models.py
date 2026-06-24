"""Shared data structures used across ingestion, indexing and search."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Message:
    """A single normalized message from any supported source.

    The fields below are source-agnostic: a parser for Telegram, WhatsApp, a
    generic text file, etc. all emit the same :class:`Message`, and everything
    downstream (chunking, embedding, indexing, search) is written against this
    shape rather than any one export format. New sources are added by writing a
    parser that yields these objects (see ``telesearch.ingest.base.Parser``).
    """

    id: int
    chat: str
    sender: str
    timestamp: int  # unix seconds
    date_str: str
    text: str = ""
    reply_to: Optional[int] = None
    media_type: Optional[str] = None  # "photo" | "video" | "voice" | "file" | None
    media_path: Optional[str] = None  # path relative to the source root
    mime_type: Optional[str] = None
    file_name: Optional[str] = None
    # Provenance (source-agnostic indexing). ``source_kind`` records which parser
    # produced this message ("telegram", "whatsapp", "file", ...); ``external_id``
    # keeps the original id in the source system; ``thread`` names a sub-channel
    # / thread within the source when one exists.
    source_kind: str = "telegram"
    external_id: Optional[str] = None
    thread: Optional[str] = None


@dataclass
class Chunk:
    """A retrievable unit stored in the vector index.

    A chunk may represent a piece of text, an image caption, a video summary,
    or a transcript. ``modality`` records which one it is so the UI can render
    results appropriately and so searches can be filtered by type.
    """

    chunk_id: str
    message_id: int
    chat: str
    sender: str
    timestamp: int
    date_str: str
    modality: str  # "text" | "image" | "video" | "audio"
    content: str  # the searchable text (message text, caption or transcript)
    media_path: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Multi-tenant / multi-source fields used for scoping and filtering at search
    # time. ``collection_id`` groups all chunks of one ingested source so a query
    # can be restricted to (or combined across) chosen sources; ``source_kind``
    # mirrors :attr:`Message.source_kind`; ``doc_id`` groups the chunks of a
    # single document/chat; ``lang`` is an optional ISO language hint.
    collection_id: str = ""
    source_kind: str = "telegram"
    doc_id: str = ""
    lang: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        # LanceDB stores scalars/strings cleanly; flatten ``extra`` to a string.
        import json

        row["extra"] = json.dumps(self.extra, ensure_ascii=False)
        return row
