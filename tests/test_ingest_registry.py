"""Tests for the source-agnostic ingestion layer (parsers + registry)."""

import json

import pytest

from telesearch.ingest import (
    SourceContext,
    available,
    get_parser,
    select_parser,
)


def _write_telegram(tmp_path):
    data = {
        "name": "Alice",
        "type": "personal_chat",
        "messages": [
            {
                "id": 2,
                "type": "message",
                "date": "2024-01-01T10:00:00",
                "date_unixtime": "1704103200",
                "from": "Alice",
                "text": "hello world",
            }
        ],
    }
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")


def test_builtin_parsers_registered():
    names = available()
    assert "telegram" in names
    assert "generic_text" in names
    assert "generic_file" in names


def test_select_parser_detects_telegram(tmp_path):
    _write_telegram(tmp_path)
    ctx = SourceContext(root=tmp_path)
    parser = select_parser(ctx)
    assert parser.name == "telegram"


def test_telegram_parser_sets_source_kind(tmp_path):
    _write_telegram(tmp_path)
    parser = get_parser("telegram")
    messages = list(parser.parse(SourceContext(root=tmp_path)))
    assert messages and all(m.source_kind == "telegram" for m in messages)


def test_chat_name_override(tmp_path):
    _write_telegram(tmp_path)
    parser = get_parser("telegram")
    messages = list(parser.parse(SourceContext(root=tmp_path, chat_name="Renamed")))
    assert all(m.chat == "Renamed" for m in messages)


def test_generic_text_is_fallback_for_plain_text(tmp_path):
    (tmp_path / "notes.txt").write_text("some free-form notes here", encoding="utf-8")
    parser = select_parser(SourceContext(root=tmp_path))
    # No telegram export present -> generic text wins over generic file.
    assert parser.name == "generic_text"

    messages = list(parser.parse(SourceContext(root=tmp_path)))
    assert len(messages) == 1
    assert messages[0].text == "some free-form notes here"
    assert messages[0].source_kind == "file"


def test_declared_kind_overrides_detection(tmp_path):
    _write_telegram(tmp_path)
    # Force the generic file parser even though a telegram export is present.
    ctx = SourceContext(root=tmp_path, declared_kind="generic_file")
    parser = select_parser(ctx)
    assert parser.name == "generic_file"


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        get_parser("nope")
