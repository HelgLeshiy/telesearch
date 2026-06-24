"""Tests for the Phase 2 messenger parsers (WhatsApp, generic JSON chat)."""

import json

from telesearch.ingest import SourceContext, get_parser, select_parser


def test_whatsapp_dash_format_and_multiline(tmp_path):
    chat = (
        "01/02/2024, 10:00 - Alice: hey there\n"
        "01/02/2024, 10:01 - Bob: hi\nthis continues on a second line\n"
        "01/02/2024, 10:02 - Messages are end-to-end encrypted\n"
    )
    (tmp_path / "WhatsApp Chat with Bob.txt").write_text(chat, encoding="utf-8")

    parser = select_parser(SourceContext(root=tmp_path))
    assert parser.name == "whatsapp"  # beats generic_text fallback

    msgs = list(parser.parse(SourceContext(root=tmp_path)))
    assert [m.sender for m in msgs] == ["Alice", "Bob"]  # system line dropped
    assert "continues on a second line" in msgs[1].text
    assert all(m.source_kind == "whatsapp" for m in msgs)
    assert msgs[0].timestamp > 0


def test_whatsapp_bracket_format(tmp_path):
    chat = "[2024-01-31, 10:05:12] Alice: bracketed hello\n"
    (tmp_path / "_chat.txt").write_text(chat, encoding="utf-8")
    msgs = list(get_parser("whatsapp").parse(SourceContext(root=tmp_path)))
    assert len(msgs) == 1
    assert msgs[0].sender == "Alice"
    assert msgs[0].text == "bracketed hello"


def test_json_chat_list_of_messages(tmp_path):
    data = [
        {"author": "Alice", "content": "hello from json", "timestamp": "2024-01-01T10:00:00"},
        {"author": "Bob", "content": "reply", "timestamp": 1704103260},
        {"author": "Bob", "content": ""},  # empty -> skipped
    ]
    (tmp_path / "export.json").write_text(json.dumps(data), encoding="utf-8")

    parser = select_parser(SourceContext(root=tmp_path))
    assert parser.name == "json_chat"
    msgs = list(parser.parse(SourceContext(root=tmp_path)))
    assert [m.text for m in msgs] == ["hello from json", "reply"]
    assert msgs[0].sender == "Alice"
    assert msgs[0].timestamp > 0
    assert all(m.source_kind == "json_chat" for m in msgs)


def test_json_chat_messages_envelope_and_flexible_fields(tmp_path):
    data = {"messages": [{"from": {"name": "Carol"}, "text": "nested sender"}]}
    (tmp_path / "vk.json").write_text(json.dumps(data), encoding="utf-8")
    msgs = list(get_parser("json_chat").parse(SourceContext(root=tmp_path)))
    assert len(msgs) == 1
    assert msgs[0].sender == "Carol"
    assert msgs[0].text == "nested sender"


def test_json_chat_defers_to_telegram(tmp_path):
    data = {
        "name": "Chat",
        "type": "personal_chat",
        "messages": [
            {"id": 1, "type": "message", "date_unixtime": "1", "from": "A", "text": "x"}
        ],
    }
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    # Telegram must win over the generic JSON parser.
    assert select_parser(SourceContext(root=tmp_path)).name == "telegram"
