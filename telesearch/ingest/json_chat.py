"""Best-effort parser for generic JSON chat exports (VK / Slack / Discord, ...).

There is no single standard for messenger JSON, so this parser accepts the
common shapes — a top-level list of messages, or an object with a ``messages``
(or ``items`` / ``conversations``) array — and maps flexible field names for the
sender, text and timestamp. It is intentionally lenient ("best effort"): it
won't perfectly model every tool, but it makes most JSON exports searchable
without a bespoke parser. Telegram's own ``result.json`` is handled by the more
specific :class:`TelegramParser`, which outranks this one.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..models import Message
from .base import SourceContext

_SENDER_KEYS = ("sender", "from", "author", "user", "name", "from_name", "nick")
_TEXT_KEYS = ("text", "content", "message", "body", "msg")
_TS_KEYS = ("timestamp", "date", "ts", "time", "datetime", "created_at", "unixtime")
_MESSAGES_KEYS = ("messages", "items", "conversations", "data")


def _json_file(root: Path) -> Path | None:
    if root.is_file():
        return root if root.suffix.lower() == ".json" else None
    jsons = sorted(root.rglob("*.json"))
    return jsons[0] if jsons else None


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def _messages(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if isinstance(data, dict):
        for key in _MESSAGES_KEYS:
            val = data.get(key)
            if isinstance(val, list):
                return [m for m in val if isinstance(m, dict)]
    return []


def _looks_like_telegram(data: Any) -> bool:
    return isinstance(data, dict) and "messages" in data and (
        "type" in data or "name" in data
    ) and any(
        isinstance(m, dict) and "date_unixtime" in m
        for m in data.get("messages", [])[:5]
    )


def _first(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(_first(item, _TEXT_KEYS) or ""))
        return "".join(parts)
    if isinstance(value, dict):
        return str(_first(value, _TEXT_KEYS) or "")
    return ""


def _parse_ts(value: Any) -> tuple[int, str]:
    if value is None:
        return 0, ""
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e11:  # milliseconds
            v /= 1000.0
        return int(v), ""
    s = str(value)
    if s.isdigit():
        return int(s), s
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return int(datetime.strptime(s[:26], fmt).timestamp()), s
        except ValueError:
            continue
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()), s
    except ValueError:
        return 0, s


def _sender_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(_first(value, _SENDER_KEYS) or "unknown")
    return str(value) if value not in (None, "") else "unknown"


class GenericJSONChatParser:
    """Tolerant parser for assorted JSON chat exports (best-effort)."""

    name = "json_chat"

    def sniff(self, ctx: SourceContext) -> float:
        path = _json_file(ctx.root)
        if path is None or not path.exists():
            return 0.0
        try:
            data = _load(path)
        except (json.JSONDecodeError, OSError):
            return 0.0
        if _looks_like_telegram(data):
            return 0.0  # defer to the specific Telegram parser
        msgs = _messages(data)
        if not msgs:
            return 0.0
        sample = msgs[0]
        has_text = _first(sample, _TEXT_KEYS) is not None
        has_sender = _first(sample, _SENDER_KEYS) is not None
        return 0.6 if (has_text and has_sender) else 0.0

    def parse(self, ctx: SourceContext) -> Iterator[Message]:
        path = _json_file(ctx.root)
        if path is None or not path.exists():
            return
        try:
            data = _load(path)
        except (json.JSONDecodeError, OSError):
            return
        chat = ctx.chat_name or path.stem
        for idx, raw in enumerate(_messages(data), start=1):
            text = _flatten_text(_first(raw, _TEXT_KEYS))
            if not text.strip():
                continue
            ts, date_str = _parse_ts(_first(raw, _TS_KEYS))
            yield Message(
                id=int(raw.get("id", idx)) if str(raw.get("id", idx)).isdigit() else idx,
                chat=chat,
                sender=_sender_name(_first(raw, _SENDER_KEYS)),
                timestamp=ts,
                date_str=date_str,
                text=text,
                source_kind="json_chat",
                external_id=str(raw.get("id")) if raw.get("id") is not None else None,
            )
