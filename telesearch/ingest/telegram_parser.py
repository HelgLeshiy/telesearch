"""Parse a Telegram Desktop ``result.json`` export into normalized messages.

To produce the export: Telegram Desktop -> hamburger menu -> Settings ->
Advanced -> Export Telegram data -> select the chat, enable Photos / Video
files / Voice messages, and choose **"Machine-readable JSON"** as the format.
This yields a folder containing ``result.json`` plus media sub-folders
(``photos/``, ``video_files/``, ``voice_messages/`` ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..models import Message


def _flatten_text(text: Any) -> str:
    """Telegram's ``text`` field is either a string or a list of entities."""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts: list[str] = []
        for item in text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


# Telegram writes this placeholder when the media itself was not exported.
_NOT_INCLUDED = "(File not included"


def _is_real_path(path: Any) -> bool:
    return isinstance(path, str) and bool(path) and not path.startswith(_NOT_INCLUDED)


def _media_for(msg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(media_type, relative_path)`` for a message, if any.

    Normalizes Telegram's many media shapes into four handler types:
    ``photo`` / ``video`` / ``voice`` / ``file`` (plus ``sticker``, which the
    pipeline currently skips). Files sent as plain attachments are routed by
    MIME type so that e.g. an image dropped into the ``files/`` folder is still
    captioned like a photo. Media that was not downloaded (placeholder path) is
    treated as having no media.
    """
    if _is_real_path(msg.get("photo")):
        return "photo", msg["photo"]

    media_type = msg.get("media_type")
    path = msg.get("file")
    if not _is_real_path(path):
        return None, None

    known = {
        "video_file": "video",
        "video_message": "video",
        "animation": "video",
        "voice_message": "voice",
        "audio_file": "voice",
    }
    if media_type in known:
        return known[media_type], path
    if media_type == "sticker":
        return "sticker", path

    # Generic attachment: route by MIME so images/videos/audio sent as files
    # are processed by the right handler instead of treated as documents.
    mime = (msg.get("mime_type") or "").lower()
    if mime.startswith("image/") and not mime.endswith("webp"):
        return "photo", path
    if mime.startswith("video/"):
        return "video", path
    if mime.startswith("audio/"):
        return "voice", path
    return "file", path


def parse_export(export_path: str | Path) -> Iterator[Message]:
    """Yield :class:`Message` objects from a Telegram export.

    ``export_path`` may point at the export directory or directly at
    ``result.json``. Media paths are returned relative to the export root so
    they can be resolved later regardless of where the export is mounted.
    """
    export_path = Path(export_path)
    if export_path.is_dir():
        result_file = export_path / "result.json"
    else:
        result_file = export_path

    if not result_file.exists():
        raise FileNotFoundError(f"Could not find result.json at {result_file}")

    with result_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    chat_name = data.get("name") or "telegram_chat"
    messages = data.get("messages", [])

    for msg in messages:
        if msg.get("type") != "message":
            continue  # skip service messages (joins, calls, pins, ...)

        text = _flatten_text(msg.get("text", ""))
        media_type, media_path = _media_for(msg)

        try:
            timestamp = int(msg.get("date_unixtime", 0))
        except (TypeError, ValueError):
            timestamp = 0

        yield Message(
            id=int(msg.get("id", 0)),
            chat=chat_name,
            sender=msg.get("from") or msg.get("from_id") or "unknown",
            timestamp=timestamp,
            date_str=str(msg.get("date", "")),
            text=text,
            reply_to=msg.get("reply_to_message_id"),
            media_type=media_type,
            media_path=media_path,
            mime_type=msg.get("mime_type"),
            file_name=msg.get("file_name"),
        )
