"""Parser for WhatsApp "Export chat" text files.

WhatsApp exports a single ``_chat.txt`` (or ``WhatsApp Chat with X.txt``) where
each message starts with a date/time prefix; messages can span multiple lines.
The exact prefix format varies by locale and platform, so this parser is
deliberately tolerant and treats unrecognized leading lines as continuations.
This is best-effort by nature — see the design doc's note on messenger exports.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..models import Message
from .base import SourceContext

# Two common shapes:
#   [2024-01-31, 10:05:12] Alice: hello          (bracketed, iOS-style)
#   31/01/2024, 10:05 - Alice: hello             (dash, Android-style)
_BRACKET = re.compile(
    r"^\[(?P<date>[^\],]+),?\s+(?P<time>[^\]]+)\]\s*(?P<rest>.*)$"
)
_DASH = re.compile(
    r"^(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APMapm]{2})?)\s+-\s+(?P<rest>.*)$"
)
# Strip iOS bidi/format marker that can prefix lines.
_LRM = "\u200e"

_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%d/%m/%y %H:%M", "%m/%d/%y %H:%M",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d/%m/%Y %I:%M %p", "%m/%d/%y %I:%M %p", "%d/%m/%y %I:%M %p",
)


def _parse_ts(date: str, time: str) -> tuple[int, str]:
    raw = f"{date.strip()} {time.strip()}".replace("\u202f", " ")
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return int(dt.timestamp()), raw
        except ValueError:
            continue
    return 0, raw


def _match(line: str):
    line = line.lstrip(_LRM)
    m = _BRACKET.match(line)
    if m:
        return m
    return _DASH.match(line)


def _chat_file(root: Path) -> Path | None:
    if root.is_file():
        return root if root.suffix.lower() == ".txt" else None
    txts = sorted(root.rglob("*.txt"))
    # Prefer a file that looks like a WhatsApp export.
    for p in txts:
        if "chat" in p.name.lower() or "whatsapp" in p.name.lower():
            return p
    return txts[0] if txts else None


class WhatsAppParser:
    """:class:`~telesearch.ingest.base.Parser` for WhatsApp chat exports."""

    name = "whatsapp"

    def sniff(self, ctx: SourceContext) -> float:
        path = _chat_file(ctx.root)
        if path is None or not path.exists():
            return 0.0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                hits = 0
                for _ in range(40):
                    line = fh.readline()
                    if not line:
                        break
                    if _match(line):
                        hits += 1
        except OSError:
            return 0.0
        # Several timestamped lines in the head is a strong signal.
        return 0.85 if hits >= 3 else 0.0

    def parse(self, ctx: SourceContext) -> Iterator[Message]:
        path = _chat_file(ctx.root)
        if path is None or not path.exists():
            return
        chat = ctx.chat_name or path.stem

        cur: dict | None = None
        idx = 0

        def emit(rec: dict) -> Message:
            nonlocal idx
            idx += 1
            ts, date_str = _parse_ts(rec["date"], rec["time"])
            return Message(
                id=idx,
                chat=chat,
                sender=rec["sender"],
                timestamp=ts,
                date_str=date_str,
                text=rec["text"].strip(),
                source_kind="whatsapp",
            )

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                m = _match(line)
                if m:
                    if cur is not None and cur["text"].strip():
                        yield emit(cur)
                    rest = m.group("rest").lstrip(_LRM)
                    if ": " in rest:
                        sender, text = rest.split(": ", 1)
                    else:
                        # System line (no sender), skip — not user content.
                        cur = None
                        continue
                    cur = {
                        "date": m.group("date"),
                        "time": m.group("time"),
                        "sender": sender.strip(),
                        "text": text,
                    }
                elif cur is not None:
                    cur["text"] += "\n" + line  # continuation of a multi-line message
        if cur is not None and cur["text"].strip():
            yield emit(cur)
