"""Transcribe audio (voice messages and video sound tracks) with Whisper.

Uses faster-whisper (CTranslate2) which runs Whisper large-v3 comfortably on
the GPU. Transcripts become searchable text in the index.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..config import Settings


class Transcriber:
    """Lazy wrapper around a faster-whisper model.

    The underlying CTranslate2 model is shared in-process and contends on the
    GPU, so transcription is serialized with a lock. This lets the rest of the
    pipeline (remote VLM calls) run concurrently while ASR stays safe.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Audio transcription needs faster-whisper. "
                    "Install with: pip install faster-whisper"
                ) from exc

            device = "cuda" if self.settings.device.startswith("cuda") else "cpu"
            self._model = WhisperModel(
                self.settings.whisper_model,
                device=device,
                compute_type=self.settings.whisper_compute,
            )
        return self._model

    def transcribe(self, media_path: str | Path) -> str:
        """Return the transcript text for an audio or video file."""
        with self._lock:
            model = self._ensure_model()
            segments, _info = model.transcribe(str(media_path), vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
