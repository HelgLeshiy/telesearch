"""Indexing service: ingest a source and build its searchable index.

Selects the right parser for an upload (or honors an explicit kind), normalizes
it to messages, and writes chunks into the workspace's vector store. Used by the
CLI today and by background workers / the API later — all through one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..ingest import SourceContext, get_parser, select_parser
from .context import RequestContext


@dataclass
class IndexResult:
    parser: str
    collection_id: str
    messages: int
    chunks: int
    db_path: str


def _default_collection_id(root: Path) -> str:
    """Derive a stable collection id from the source path."""
    name = root.stem if root.is_file() else root.name
    return name or "default"


def index_source(
    path: str | Path,
    settings: Settings,
    *,
    ctx: Optional[RequestContext] = None,
    kind: Optional[str] = None,
    collection_id: Optional[str] = None,
    chat_name: Optional[str] = None,
    **build_flags,
) -> IndexResult:
    """Parse a source and build/extend its index. Returns an :class:`IndexResult`.

    ``kind`` pins the parser (e.g. ``"telegram"``); when omitted the parser is
    auto-selected by sniffing. ``collection_id`` groups this source's chunks for
    scoped search; it defaults to the source's name. ``build_flags`` are passed
    through to :func:`telesearch.index.build.build_index` (``do_images`` etc.).
    """
    from ..index.build import build_index

    ctx = ctx or RequestContext.default()
    root = Path(path)
    collection_id = collection_id or _default_collection_id(root)

    source_ctx = SourceContext(
        root=root,
        collection_id=collection_id,
        workspace_id=ctx.workspace_id,
        declared_kind=kind,
        chat_name=chat_name,
    )
    parser = get_parser(kind) if kind else select_parser(source_ctx)
    messages = list(parser.parse(source_ctx))

    db_path = settings.workspace_db_path(ctx.workspace_id)
    count = build_index(
        messages,
        source_ctx.media_root,
        settings,
        db_path=db_path,
        collection_id=collection_id,
        **build_flags,
    )
    return IndexResult(
        parser=parser.name,
        collection_id=collection_id,
        messages=len(messages),
        chunks=count,
        db_path=str(db_path),
    )


def reindex_source_text(
    path: str | Path,
    settings: Settings,
    *,
    ctx: Optional[RequestContext] = None,
    kind: Optional[str] = None,
    collection_id: Optional[str] = None,
    chat_name: Optional[str] = None,
    do_conversation_windows: bool = True,
) -> IndexResult:
    """Refresh only text + conversation chunks for a source (no media re-process)."""
    from ..index.build import reindex_text

    ctx = ctx or RequestContext.default()
    root = Path(path)
    collection_id = collection_id or _default_collection_id(root)

    source_ctx = SourceContext(
        root=root,
        collection_id=collection_id,
        workspace_id=ctx.workspace_id,
        declared_kind=kind,
        chat_name=chat_name,
    )
    parser = get_parser(kind) if kind else select_parser(source_ctx)
    messages = list(parser.parse(source_ctx))

    db_path = settings.workspace_db_path(ctx.workspace_id)
    count = reindex_text(
        messages,
        source_ctx.media_root,
        settings,
        do_conversation_windows=do_conversation_windows,
        db_path=db_path,
        collection_id=collection_id,
    )
    return IndexResult(
        parser=parser.name,
        collection_id=collection_id,
        messages=len(messages),
        chunks=count,
        db_path=str(db_path),
    )
