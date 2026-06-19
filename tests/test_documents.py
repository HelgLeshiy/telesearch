"""Tests for document text extraction and chunk splitting."""

from telesearch.media.documents import (
    extract_document_text,
    is_supported,
    split_text,
)


def test_extract_plain_text(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Quarterly budget meeting notes: revenue up 20%.", encoding="utf-8")
    text = extract_document_text(f, "text/plain", "notes.txt")
    assert "Quarterly budget" in text


def test_extract_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,amount\nAlice,100\nBob,200\n", encoding="utf-8")
    text = extract_document_text(f, "text/csv", "data.csv")
    assert "Alice" in text and "200" in text


def test_extract_html_strips_tags(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body><script>ignore()</script><p>Hello &amp; world</p></body></html>",
        encoding="utf-8",
    )
    text = extract_document_text(f, "text/html", "page.html")
    assert "Hello & world" in text
    assert "ignore" not in text


def test_binary_file_is_skipped(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00\x01\x02\x03binarystuff\x00\xff")
    assert extract_document_text(f, "application/octet-stream", "blob.bin") == ""


def test_is_supported():
    assert is_supported("report.pdf", None)
    assert is_supported("sheet.xlsx", None)
    assert is_supported(None, "text/plain")
    assert not is_supported("archive.zip", "application/zip")


def test_split_text_overlap():
    text = "word " * 1000  # ~5000 chars
    chunks = split_text(text, chunk_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # Reassembled content covers the whole document.
    assert "word" in chunks[0] and "word" in chunks[-1]


def test_split_text_short():
    assert split_text("tiny", chunk_chars=1000) == ["tiny"]
    assert split_text("   ") == []
