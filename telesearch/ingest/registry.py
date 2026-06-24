"""Registry and auto-selection for source parsers.

Built-in parsers register themselves at import time (see
``telesearch.ingest.__init__``). Callers either pin a parser by name (when the
user tells us the format) or let :func:`select_parser` pick the best match by
sniffing the upload.
"""

from __future__ import annotations

from .base import Parser, SourceContext

_PARSERS: list[Parser] = []


def register(parser: Parser) -> None:
    """Register a parser instance. Re-registering the same name replaces it."""
    global _PARSERS
    _PARSERS = [p for p in _PARSERS if p.name != parser.name]
    _PARSERS.append(parser)


def available() -> list[str]:
    """Names of all registered parsers."""
    return [p.name for p in _PARSERS]


def get_parser(name: str) -> Parser:
    """Return the parser registered under ``name`` (raises if unknown)."""
    for p in _PARSERS:
        if p.name == name:
            return p
    raise KeyError(
        f"unknown parser {name!r}; available: {', '.join(available()) or '(none)'}"
    )


def select_parser(ctx: SourceContext) -> Parser:
    """Pick the parser for ``ctx``.

    If ``ctx.declared_kind`` is set, it is honored. Otherwise the parser with
    the highest ``sniff`` confidence wins; a generic fallback should always
    return a small positive score so selection never fails for a non-empty
    upload.
    """
    if not _PARSERS:
        raise RuntimeError("no parsers registered")
    if ctx.declared_kind:
        return get_parser(ctx.declared_kind)
    ranked = sorted(_PARSERS, key=lambda p: p.sniff(ctx), reverse=True)
    best = ranked[0]
    if best.sniff(ctx) <= 0.0:
        raise ValueError(
            f"no parser recognized the source at {ctx.root}; "
            f"try passing an explicit kind ({', '.join(available())})"
        )
    return best
