"""Source-agnostic ingestion: parsers that normalize uploads into messages.

Built-in parsers are registered here at import time. Add a new input format by
implementing :class:`telesearch.ingest.base.Parser` and registering it.
"""

from .base import Parser, SourceContext
from .generic import GenericFileParser, GenericTextParser
from .json_chat import GenericJSONChatParser
from .registry import available, get_parser, register, select_parser
from .telegram_parser import TelegramParser, parse_export
from .whatsapp import WhatsAppParser

# Register built-ins (most specific first; generic parsers are weak fallbacks).
# sniff confidences: telegram .95 > whatsapp .85 > json_chat .6 > text .2 > file .1
register(TelegramParser())
register(WhatsAppParser())
register(GenericJSONChatParser())
register(GenericTextParser())
register(GenericFileParser())

__all__ = [
    "Parser",
    "SourceContext",
    "TelegramParser",
    "WhatsAppParser",
    "GenericJSONChatParser",
    "GenericTextParser",
    "GenericFileParser",
    "parse_export",
    "register",
    "available",
    "get_parser",
    "select_parser",
]
