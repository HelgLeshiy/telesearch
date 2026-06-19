"""Shared data structures used across ingestion, indexing and search."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Message:
    """A single normalized Telegram message."""

    id: int
    chat: str
    sender: str
    timestamp: int  # unix seconds
    date_str: str
    text: str = ""
    reply_to: Optional[int] = None
    media_type: Optional[str] = None  # "photo" | "video" | "voice" | "file" | None
    media_path: Optional[str] = None  # path relative to the export root
    mime_type: Optional[str] = None
    file_name: Optional[str] = None


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

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        # LanceDB stores scalars/strings cleanly; flatten ``extra`` to a string.
        import json

        row["extra"] = json.dumps(self.extra, ensure_ascii=False)
        return row
