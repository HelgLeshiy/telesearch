"""Extract evenly spaced frames from a video using PyAV (ffmpeg bindings)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def extract_frames(video_path: str | Path, num_frames: int = 4) -> list[Image.Image]:
    """Return up to ``num_frames`` evenly spaced frames as PIL images.

    Requires the optional ``av`` dependency (``pip install av``) and a system
    ffmpeg installation. Returns an empty list if the video cannot be read.
    """
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Video frame extraction needs PyAV. Install with: pip install av"
        ) from exc

    video_path = str(video_path)
    frames: list[Image.Image] = []

    try:
        container = av.open(video_path)
    except Exception:
        return frames

    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        total = stream.frames or 0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0

        if duration > 0:
            # Seek to evenly spaced timestamps; robust for long videos.
            for i in range(num_frames):
                target = duration * (i + 0.5) / num_frames
                container.seek(int(target / stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    frames.append(frame.to_image())
                    break
        else:
            # Fallback: decode sequentially and sample by frame count.
            step = max(1, total // num_frames) if total else 30
            for idx, frame in enumerate(container.decode(stream)):
                if idx % step == 0:
                    frames.append(frame.to_image())
                if len(frames) >= num_frames:
                    break
    finally:
        container.close()

    return frames[:num_frames]
