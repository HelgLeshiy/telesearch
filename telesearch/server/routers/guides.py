"""Static export guides: how to get data out of common apps for upload."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/guides", tags=["meta"])

_GUIDES = [
    {
        "kind": "telegram",
        "title": "Telegram",
        "steps": [
            "Open Telegram Desktop.",
            "Settings → Advanced → Export Telegram data.",
            "Select the chat; enable Photos / Video / Voice messages as needed.",
            "Choose format: Machine-readable JSON.",
            "Upload the resulting folder (or a zip of it, containing result.json).",
        ],
    },
    {
        "kind": "whatsapp",
        "title": "WhatsApp",
        "steps": [
            "Open the chat in WhatsApp on your phone.",
            "Tap the chat name → Export chat.",
            "Choose 'Without media' (or 'Include media').",
            "Upload the exported .txt (or the zip).",
        ],
    },
    {
        "kind": "json_chat",
        "title": "VK / Slack / Discord (JSON)",
        "steps": [
            "Export your messages as JSON using the app's export tool or a "
            "community exporter (e.g. DiscordChatExporter).",
            "Upload the .json file. Parsing is best-effort across formats.",
        ],
    },
    {
        "kind": "generic",
        "title": "Any file or text",
        "steps": [
            "Just upload any document (PDF, Office, text/code/CSV) or .txt note.",
            "Its text is extracted and indexed automatically.",
        ],
    },
]


@router.get("")
def list_guides() -> list[dict]:
    return _GUIDES
