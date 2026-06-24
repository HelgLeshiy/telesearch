"""Reusable service layer shared by the CLI, (future) HTTP API and workers.

The indexing and search logic used to live inside the Typer CLI. It now lives
here as plain callables so the same code paths can be driven from the command
line, an API request handler, or a background job without duplication.
"""

from .context import DEFAULT_WORKSPACE, RequestContext
from .indexing import IndexResult, index_source, reindex_source_text
from .search import SearchQuery, SearchService, build_where

__all__ = [
    "RequestContext",
    "DEFAULT_WORKSPACE",
    "index_source",
    "reindex_source_text",
    "IndexResult",
    "SearchQuery",
    "SearchService",
    "build_where",
]
