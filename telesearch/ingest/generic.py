"""Generic fallback parsers: index arbitrary text and files.

These prove the source-agnostic ingestion seam and give every upload a sensible
default: if no specialized parser recognizes the source, we still extract its
text and make it searchable.

* :class:`GenericTextParser` — plain-text/markdown/code/CSV style files. Each
  file becomes one message whose ``text`` is the file contents; the indexing
  pipeline then chunks long text as usual.
* :class:`GenericFileParser` — any other single file. It is emitted as a
  ``file`` message so the existing document-extraction step (PDF/Office/...) in
  the build pipeline turns it into searchable text. Genuinely binary files with
  no extractable text simply yield nothing downstream.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterator

from ..models import Message
from .base import SourceContext

# Extensions we treat as directly-readable text (no extraction step needed).
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json",
    ".yaml", ".yml", ".ini", ".cfg", ".toml", ".py", ".js", ".ts", ".java",
    ".c", ".h", ".cpp", ".go", ".rs", ".rb", ".sh", ".sql", ".html", ".xml",
    ".srt", ".vtt",
}


def _iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


class GenericTextParser:
    """Index readable text files (the default for unknown text uploads)."""

    name = "generic_text"

    def sniff(self, ctx: SourceContext) -> float:
        for path in _iter_files(ctx.root):
            if path.suffix.lower() in _TEXT_SUFFIXES:
                return 0.2  # weak: only wins when nothing specific matches
        return 0.0

    def parse(self, ctx: SourceContext) -> Iterator[Message]:
        idx = 0
        for path in _iter_files(ctx.root):
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if not text:
                continue
            idx += 1
            ts = int(path.stat().st_mtime)
            yield Message(
                id=idx,
                chat=ctx.chat_name or path.name,
                sender=path.name,
                timestamp=ts,
                date_str="",
                text=text,
                source_kind="file",
                external_id=str(path.relative_to(ctx.media_root)),
            )


class GenericFileParser:
    """Index any single file via the document-extraction pipeline."""

    name = "generic_file"

    def sniff(self, ctx: SourceContext) -> float:
        for _ in _iter_files(ctx.root):
            return 0.1  # absolute last resort
        return 0.0

    def parse(self, ctx: SourceContext) -> Iterator[Message]:
        idx = 0
        for path in _iter_files(ctx.root):
            idx += 1
            rel = str(path.relative_to(ctx.media_root))
            mime, _ = mimetypes.guess_type(path.name)
            yield Message(
                id=idx,
                chat=ctx.chat_name or path.name,
                sender=path.name,
                timestamp=int(path.stat().st_mtime),
                date_str="",
                media_type="file",
                media_path=rel,
                mime_type=mime,
                file_name=path.name,
                source_kind="file",
                external_id=rel,
            )
