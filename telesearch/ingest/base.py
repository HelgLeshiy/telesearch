"""Source-agnostic ingestion interface.

Indexing and search are written against :class:`telesearch.models.Message`, not
against any one export format. A :class:`Parser` turns an uploaded source (a
folder, an archive that was unpacked, a single file) into a stream of
``Message`` objects. New input formats — WhatsApp, VK, Slack, a plain text file,
etc. — are added by implementing this protocol and registering it
(``telesearch.ingest.registry``); nothing downstream needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

from ..models import Message


@dataclass
class SourceContext:
    """Everything a parser needs about one ingested source.

    ``root`` is the on-disk location of the (already unpacked) upload — a
    directory or a single file. ``collection_id`` is the logical id under which
    every produced chunk is grouped so search can be scoped to this source.
    ``declared_kind`` is an optional caller hint ("telegram", "whatsapp", ...)
    that, when present, pins parser selection instead of relying on sniffing.
    """

    root: Path
    collection_id: str = "default"
    workspace_id: str = "default"
    declared_kind: Optional[str] = None
    chat_name: Optional[str] = None  # override the display name when known

    @property
    def media_root(self) -> Path:
        """Directory used to resolve relative media paths emitted by the parser."""
        return self.root if self.root.is_dir() else self.root.parent


@runtime_checkable
class Parser(Protocol):
    """Turns an ingested source into a stream of normalized messages."""

    name: str

    def sniff(self, ctx: SourceContext) -> float:
        """Confidence in ``0.0..1.0`` that this parser handles ``ctx``.

        A generic fallback parser returns a small positive value so it wins
        only when nothing more specific matches. Return ``0.0`` to abstain.
        """
        ...

    def parse(self, ctx: SourceContext) -> Iterator[Message]:
        """Yield normalized :class:`Message` objects for ``ctx``."""
        ...
