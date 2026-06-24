"""Source-agnostic ingestion: parsers that normalize uploads into messages.

Built-in parsers are registered here at import time. Add a new input format by
implementing :class:`telesearch.ingest.base.Parser` and registering it.
"""

from .base import Parser, SourceContext
from .generic import GenericFileParser, GenericTextParser
from .registry import available, get_parser, register, select_parser
from .telegram_parser import TelegramParser, parse_export

# Register built-ins (most specific first; generic parsers are weak fallbacks).
register(TelegramParser())
register(GenericTextParser())
register(GenericFileParser())

__all__ = [
    "Parser",
    "SourceContext",
    "TelegramParser",
    "GenericTextParser",
    "GenericFileParser",
    "parse_export",
    "register",
    "available",
    "get_parser",
    "select_parser",
]
