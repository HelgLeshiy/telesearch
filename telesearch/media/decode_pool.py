"""Run risky, CPU-bound media decoding in killable worker processes.

Image decoding (PIL) and document parsing (pypdf/openpyxl/...) occasionally
hang in native code on a malformed/huge input. A thread can't be interrupted,
so such a hang permanently wedges a worker. Running these pure functions in a
process pool lets us enforce a per-call timeout and **terminate** the stuck
worker, so the indexing run keeps making progress instead of dying.

If ``pebble`` is unavailable we fall back to running in-thread (no isolation),
preserving behaviour rather than failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .captioner import _path_to_data_url
from .documents import extract_document_text


class DecodeTimeout(Exception):
    """Raised when a decode/extract call exceeds its time budget."""


class DecodePool:
    """Killable process pool for image decode + document text extraction."""

    def __init__(self, max_workers: int, timeout: float, max_image_megapixels: float):
        self.timeout = timeout if timeout and timeout > 0 else None
        self.max_image_megapixels = max_image_megapixels
        self._pool = None
        self._lock = None
        try:
            import multiprocessing as mp
            import threading

            from pebble import ProcessPool  # type: ignore

            self._lock = threading.Lock()
            self._max_workers = max(1, max_workers)
            # "spawn" gives clean worker processes that never inherit a CUDA
            # context from the parent (which fork would, and which can crash).
            self._pool = ProcessPool(
                max_workers=self._max_workers, context=mp.get_context("spawn")
            )
        except Exception:
            # pebble missing or pool init failed -> run in-thread (no isolation).
            self._pool = None

    @property
    def isolated(self) -> bool:
        return self._pool is not None

    def _run(self, fn, args):
        if self._pool is None:
            return fn(*args)
        from concurrent.futures import TimeoutError as FuturesTimeout

        with self._lock:
            pool = self._pool
        future = pool.schedule(fn, args=list(args), timeout=self.timeout)
        try:
            return future.result()
        except FuturesTimeout as exc:
            raise DecodeTimeout(
                f"{getattr(fn, '__name__', fn)} exceeded {self.timeout:.0f}s"
            ) from exc

    def image_data_url(self, path: str | Path) -> str:
        """Decode an image file to a JPEG data URL (killable)."""
        return self._run(
            _path_to_data_url, (str(path), 1024, self.max_image_megapixels)
        )

    def document_text(
        self,
        path: str | Path,
        mime_type: Optional[str],
        file_name: Optional[str],
        max_chars: int,
    ) -> str:
        """Extract text from a document attachment (killable)."""
        return self._run(
            extract_document_text, (str(path), mime_type, file_name, max_chars)
        )

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join(timeout=5)
            except Exception:
                try:
                    self._pool.stop()
                except Exception:
                    pass
            self._pool = None
