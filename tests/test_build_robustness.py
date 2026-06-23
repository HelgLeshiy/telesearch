"""Tests for indexing robustness: per-message timeout/skip and image guards."""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import io

from PIL import Image

from telesearch.index.build import _process_block
from telesearch.media.captioner import _path_to_data_url


class _NullBar:
    """Minimal stand-in for a tqdm bar."""

    def __init__(self):
        self.n = 0

    def update(self, k):
        self.n += k


def _msg(i):
    return SimpleNamespace(id=i)


def test_process_block_skips_hung_message():
    """A message that exceeds the timeout is skipped, not allowed to stall."""

    def process(msg):
        if msg.id == 2:
            time.sleep(5)  # would hang the block without a timeout
            return ["should-never-be-used"]
        return [f"ok-{msg.id}"]

    block = [_msg(1), _msg(2), _msg(3)]
    bar = _NullBar()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = _process_block(block, process, pool, bar, item_timeout=0.3)

    # Every message advances the progress bar, even the skipped one.
    assert bar.n == 3
    assert results[0] == ["ok-1"]
    assert results[1] == []  # timed out -> empty, skipped
    assert results[2] == ["ok-3"]


def test_process_block_handles_exceptions():
    """A message that raises is logged and skipped without failing the block."""

    def process(msg):
        if msg.id == 2:
            raise RuntimeError("boom")
        return [f"ok-{msg.id}"]

    block = [_msg(1), _msg(2), _msg(3)]
    bar = _NullBar()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = _process_block(block, process, pool, bar, item_timeout=5)

    assert bar.n == 3
    assert results == [["ok-1"], [], ["ok-3"]]


def test_process_block_without_pool():
    """The serial path (pool=None) still bounds errors per message."""

    def process(msg):
        if msg.id == 2:
            raise ValueError("nope")
        return [msg.id]

    bar = _NullBar()
    results = _process_block([_msg(1), _msg(2)], process, None, bar, item_timeout=0)
    assert bar.n == 2
    assert results == [[1], []]


def test_path_to_data_url_reads_image(tmp_path):
    f = tmp_path / "pic.jpg"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(f, format="JPEG")
    url = _path_to_data_url(f, max_megapixels=50)
    assert url.startswith("data:image/jpeg;base64,")


def test_path_to_data_url_rejects_oversized_image(tmp_path):
    f = tmp_path / "big.png"
    Image.new("RGB", (1000, 1000), (255, 0, 0)).save(f, format="PNG")
    # 1 MP image with a 0.1 MP cap -> rejected instead of decoded.
    try:
        _path_to_data_url(f, max_megapixels=0.1)
    except ValueError as exc:
        assert "too large" in str(exc)
    else:  # pragma: no cover - guard must trip
        raise AssertionError("expected ValueError for oversized image")


def test_path_to_data_url_downscales_large_image(tmp_path):
    f = tmp_path / "wide.jpg"
    Image.new("RGB", (4096, 64), (0, 128, 255)).save(f, format="JPEG")
    url = _path_to_data_url(f, max_side=1024, max_megapixels=50)
    raw = url.split(",", 1)[1]
    import base64

    decoded = Image.open(io.BytesIO(base64.b64decode(raw)))
    assert max(decoded.size) <= 1024
