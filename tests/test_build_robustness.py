"""Tests for indexing robustness: per-message timeout/skip and image guards."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import io

from PIL import Image

from telesearch.index.build import _HangWatchdog, _describe, _process_block
from telesearch.media.captioner import _path_to_data_url
from telesearch.models import Message


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


def test_describe_message():
    photo = Message(
        id=1, chat="c", sender="s", timestamp=0, date_str="",
        media_type="photo", media_path="photos/x.jpg",
    )
    assert "photo" in _describe(photo)
    assert "photos/x.jpg" in _describe(photo)

    text = Message(id=2, chat="c", sender="s", timestamp=0, date_str="", text="hi")
    assert _describe(text) == "text-only"


def test_hang_watchdog_fires_and_resets():
    fired = threading.Event()
    inflight = {42: "photo, photos/stuck.jpg"}

    wd = _HangWatchdog(timeout=0.1, inflight=inflight)
    # Patch the dump so the test doesn't spam stderr; just record that it fired.
    wd._fire = lambda: (fired.set(), wd.cancel())  # type: ignore[assignment]
    wd.reset()
    assert fired.wait(2.0), "watchdog should fire when no progress is made"
    wd.cancel()


def test_hang_watchdog_disabled_when_zero():
    wd = _HangWatchdog(timeout=0, inflight={})
    wd.reset()
    assert wd._timer is None
    wd.cancel()


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


def _media_msg(media_type, media_path):
    return Message(
        id=1, chat="c", sender="s", timestamp=0, date_str="",
        media_type=media_type, media_path=media_path,
    )


def test_no_videos_skips_video_processing(tmp_path):
    """--no-videos must not extract/caption videos even when images are on."""
    from telesearch.index.build import _message_to_chunks

    (tmp_path / "v.mp4").write_bytes(b"\x00\x00")
    calls = []

    class FakeCaptioner:
        def caption_frames(self, frames):
            calls.append("caption_frames")
            return "x"

        def caption_frame_data_urls(self, urls):
            calls.append("caption_frame_data_urls")
            return "x"

    class FakeTranscriber:
        def transcribe(self, path):
            calls.append("transcribe")
            return "x"

    chunks = _message_to_chunks(
        _media_msg("video", "v.mp4"),
        tmp_path,
        captioner=FakeCaptioner(),
        transcriber=FakeTranscriber(),
        decoder=None,
        num_frames=4,
        do_images=True,
        do_videos=False,
        do_audio=False,
    )
    assert calls == []
    assert chunks == []


def test_no_audio_skips_voice(tmp_path):
    from telesearch.index.build import _message_to_chunks

    (tmp_path / "a.ogg").write_bytes(b"\x00")
    calls = []

    class FakeTranscriber:
        def transcribe(self, path):
            calls.append("transcribe")
            return "x"

    chunks = _message_to_chunks(
        _media_msg("voice", "a.ogg"),
        tmp_path,
        captioner=None,
        transcriber=FakeTranscriber(),
        decoder=None,
        num_frames=4,
        do_images=False,
        do_videos=False,
        do_audio=False,
    )
    assert calls == []
    assert chunks == []


def test_videos_enabled_uses_killable_frame_decode(tmp_path):
    """With videos on, frames are extracted via the (killable) decoder pool."""
    from telesearch.index.build import _message_to_chunks

    (tmp_path / "v.mp4").write_bytes(b"\x00\x00")
    calls = []

    class FakeDecoder:
        def video_frame_data_urls(self, path, num_frames):
            calls.append(("frames", num_frames))
            return ["u1", "u2"]

    class FakeCaptioner:
        def caption_frame_data_urls(self, urls):
            calls.append(("caption", tuple(urls)))
            return "a short summary"

    chunks = _message_to_chunks(
        _media_msg("video", "v.mp4"),
        tmp_path,
        captioner=FakeCaptioner(),
        transcriber=None,
        decoder=FakeDecoder(),
        num_frames=3,
        do_images=False,
        do_videos=True,
        do_audio=False,
    )
    assert ("frames", 3) in calls
    assert any(c[0] == "caption" for c in calls)
    assert len(chunks) == 1 and chunks[0].modality == "video"


def test_decode_pool_decodes_image(tmp_path):
    from telesearch.media.decode_pool import DecodePool

    f = tmp_path / "p.jpg"
    Image.new("RGB", (50, 50), (1, 2, 3)).save(f, format="JPEG")
    pool = DecodePool(max_workers=1, timeout=30, max_image_megapixels=50)
    try:
        url = pool.image_data_url(f)
        assert url.startswith("data:image/jpeg;base64,")
    finally:
        pool.close()


def test_decode_pool_kills_hung_task_and_recovers():
    """A task exceeding the timeout is killed; the pool keeps working after."""
    import telesearch.media.decode_pool as dp

    pool = dp.DecodePool(max_workers=1, timeout=0.5, max_image_megapixels=50)
    if not pool.isolated:
        pool.close()
        return  # pebble unavailable -> isolation not active; nothing to assert

    # Monkeypatch the run target with a top-level hanging function via the pool
    # internals: schedule a sleep that exceeds the timeout.
    import time as _time
    from concurrent.futures import TimeoutError as FuturesTimeout

    fut = pool._pool.schedule(_time.sleep, args=[10], timeout=0.5)
    try:
        try:
            fut.result()
            raise AssertionError("expected timeout")
        except FuturesTimeout:
            pass
        # Pool still usable after killing the hung worker.
        fut2 = pool._pool.schedule(_time.sleep, args=[0], timeout=5)
        fut2.result()
    finally:
        pool.close()
