"""Tests for the Telegram export parser."""

import json

from telesearch.ingest import parse_export


def _write_export(tmp_path):
    data = {
        "name": "Alice",
        "type": "personal_chat",
        "id": 1,
        "messages": [
            {"id": 1, "type": "service", "action": "create_group"},
            {
                "id": 2,
                "type": "message",
                "date": "2024-01-01T10:00:00",
                "date_unixtime": "1704103200",
                "from": "Alice",
                "from_id": "user1",
                "text": "hello world",
            },
            {
                "id": 3,
                "type": "message",
                "date": "2024-01-01T10:01:00",
                "date_unixtime": "1704103260",
                "from": "Bob",
                "from_id": "user2",
                "text": [{"type": "bold", "text": "Wow"}, " ok ", {"type": "link", "text": "http://x"}],
                "photo": "photos/p.jpg",
            },
            {
                "id": 4,
                "type": "message",
                "date": "2024-01-01T10:02:00",
                "date_unixtime": "1704103320",
                "from": "Bob",
                "media_type": "voice_message",
                "file": "voice/v.ogg",
            },
            {
                "id": 5,
                "type": "message",
                "date": "2024-01-01T10:03:00",
                "date_unixtime": "1704103380",
                "from": "Alice",
                "media_type": "video_file",
                "file": "video_files/c.mp4",
                "text": "trip",
            },
        ],
    }
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")


def test_parse_export_basic(tmp_path):
    _write_export(tmp_path)
    messages = list(parse_export(tmp_path))

    # service message skipped
    assert [m.id for m in messages] == [2, 3, 4, 5]

    by_id = {m.id: m for m in messages}
    assert by_id[2].text == "hello world"
    assert by_id[2].media_type is None
    # text entities flattened
    assert by_id[3].text == "Wow ok http://x"
    assert by_id[3].media_type == "photo"
    assert by_id[3].media_path == "photos/p.jpg"
    assert by_id[4].media_type == "voice"
    assert by_id[5].media_type == "video"
    assert by_id[2].timestamp == 1704103200
