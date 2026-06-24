"""End-to-end indexing pipeline: export -> chunks -> embeddings -> LanceDB."""

from __future__ import annotations

import faulthandler
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from ..config import Settings
from ..models import Chunk, Message
from .embeddings import TextEmbedder
from .store import VectorStore


def _resolve_media(export_root: Path, media_path: Optional[str]) -> Optional[Path]:
    if not media_path:
        return None
    candidate = export_root / media_path
    return candidate if candidate.exists() else None


def _message_to_chunks(
    msg: Message,
    export_root: Path,
    *,
    captioner,
    transcriber,
    num_frames: int,
    do_ocr: bool = False,
    do_documents: bool = False,
    doc_chunk_chars: int = 1200,
    doc_chunk_overlap: int = 150,
    doc_max_chars: int = 400_000,
    reply_lookup: Optional[dict[int, Message]] = None,
) -> list[Chunk]:
    """Turn one message into one or more searchable chunks."""
    chunks: list[Chunk] = []

    def base(
        modality: str,
        content: str,
        media_path: Optional[str],
        extra=None,
        suffix: str = "",
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{msg.id}:{modality}{suffix}",
            message_id=msg.id,
            chat=msg.chat,
            sender=msg.sender,
            timestamp=msg.timestamp,
            date_str=msg.date_str,
            modality=modality,
            content=content.strip(),
            media_path=media_path,
            extra=extra or {},
        )

    if msg.text.strip():
        # If this message is a reply, prepend a short snippet of the message it
        # replies to. A bare "yes, book it" is meaningless on its own; with the
        # quoted parent it becomes searchable and answerable in context.
        content = msg.text
        if reply_lookup and msg.reply_to:
            parent = reply_lookup.get(msg.reply_to)
            parent_text = (parent.text or "").strip() if parent else ""
            if parent_text:
                snippet = parent_text[:200]
                content = f"(replying to {parent.sender}: {snippet})\n{msg.text}"
        chunks.append(base("text", content, None))

    resolved = _resolve_media(export_root, msg.media_path)

    if msg.media_type == "photo" and resolved and captioner is not None:
        try:
            caption = captioner.caption_image(resolved)
            if caption:
                content = caption
                if msg.text.strip():
                    content = f"{caption}\nCaption: {msg.text.strip()}"
                chunks.append(base("image", content, msg.media_path, {"caption": caption}))
        except Exception as exc:  # pragma: no cover - robustness for big exports
            tqdm.write(f"[warn] image caption failed for msg {msg.id}: {exc}")

        if do_ocr:
            try:
                ocr_text = captioner.ocr_image(resolved)
                if ocr_text:
                    # Separate chunk so on-image text is retrievable on its own
                    # (great for screenshots, receipts, documents, memes).
                    chunks.append(
                        base("ocr", ocr_text, msg.media_path, {"ocr_text": ocr_text})
                    )
            except Exception as exc:  # pragma: no cover
                tqdm.write(f"[warn] image OCR failed for msg {msg.id}: {exc}")

    if msg.media_type == "video" and resolved:
        parts: list[str] = []
        extra: dict = {}
        if captioner is not None:
            try:
                from ..media.video import extract_frames

                frames = extract_frames(resolved, num_frames)
                summary = captioner.caption_frames(frames)
                if summary:
                    parts.append(summary)
                    extra["summary"] = summary
            except Exception as exc:  # pragma: no cover
                tqdm.write(f"[warn] video caption failed for msg {msg.id}: {exc}")
        if transcriber is not None:
            try:
                transcript = transcriber.transcribe(resolved)
                if transcript:
                    parts.append(f"Transcript: {transcript}")
                    extra["transcript"] = transcript
            except Exception as exc:  # pragma: no cover
                tqdm.write(f"[warn] video transcribe failed for msg {msg.id}: {exc}")
        if parts:
            chunks.append(base("video", "\n".join(parts), msg.media_path, extra))

    if msg.media_type == "file" and resolved and do_documents:
        from ..media.documents import extract_document_text, split_text

        try:
            text = extract_document_text(
                resolved, msg.mime_type, msg.file_name, max_chars=doc_max_chars
            )
        except Exception as exc:  # pragma: no cover - robustness for big exports
            tqdm.write(f"[warn] document extract failed for msg {msg.id}: {exc}")
            text = ""
        if text:
            label = msg.file_name or Path(msg.media_path).name
            pieces = split_text(text, doc_chunk_chars, doc_chunk_overlap)
            for i, piece in enumerate(pieces):
                # Prefix the file name so retrieval/answers know the source.
                content = f"{label}\n{piece}" if i == 0 else piece
                chunks.append(
                    base(
                        "document",
                        content,
                        msg.media_path,
                        {
                            "file_name": label,
                            "mime_type": msg.mime_type,
                            "part": i,
                            "n_parts": len(pieces),
                        },
                        suffix=f":{i}",
                    )
                )

    if msg.media_type == "voice" and resolved and transcriber is not None:
        try:
            transcript = transcriber.transcribe(resolved)
            if transcript:
                chunks.append(
                    base("audio", transcript, msg.media_path, {"transcript": transcript})
                )
        except Exception as exc:  # pragma: no cover
            tqdm.write(f"[warn] voice transcribe failed for msg {msg.id}: {exc}")

    return chunks


def _iter_blocks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


_MEDIA_MARKERS = {
    "photo": "[photo]",
    "video": "[video]",
    "voice": "[voice message]",
    "sticker": "",
}


def _conversation_line(msg: Message) -> Optional[str]:
    """Render a message as one ``[msg id] Sender: text`` line for a window.

    Media-only messages contribute a short marker (``[photo]`` etc.) so the flow
    of the conversation is preserved even where there is no typed text. The
    per-line message id lets the answer model cite a specific message.
    """
    text = (msg.text or "").strip()
    if msg.media_type == "file":
        marker = f"[file: {msg.file_name}]" if msg.file_name else "[file]"
    else:
        marker = _MEDIA_MARKERS.get(msg.media_type or "", "")
    body = " ".join(p for p in (text, marker) if p).strip()
    if not body:
        return None
    return f"[msg {msg.id}] {msg.sender}: {body}"


def _build_conversation_chunks(
    messages: list[Message],
    *,
    window_size: int,
    stride: int,
    max_gap: int,
) -> list[Chunk]:
    """Group consecutive messages into overlapping conversation-window chunks.

    Messages are ordered by time and split into sessions wherever the gap
    between two messages exceeds ``max_gap`` (so unrelated conversations are not
    glued together). Each session is then covered by sliding windows of up to
    ``window_size`` messages advancing by ``stride`` (so windows overlap),
    giving retrieval the surrounding context that a single short message lacks.
    """
    if window_size <= 1:
        return []
    stride = max(1, stride)
    ordered = sorted(messages, key=lambda m: (m.timestamp, m.id))

    sessions: list[list[Message]] = []
    current: list[Message] = []
    prev_ts: Optional[int] = None
    for m in ordered:
        if (
            prev_ts is not None
            and max_gap
            and (m.timestamp - prev_ts) > max_gap
        ):
            if current:
                sessions.append(current)
            current = []
        current.append(m)
        prev_ts = m.timestamp
    if current:
        sessions.append(current)

    chunks: list[Chunk] = []
    seen_cids: set[str] = set()
    for session in sessions:
        n = len(session)
        for start in range(0, n, stride):
            window = session[start : start + window_size]
            if len(window) < 2:
                continue  # a lone message is already covered by its own chunk
            lines = [ln for ln in (_conversation_line(m) for m in window) if ln]
            if len(lines) < 2:
                continue
            first = window[0]
            cid = f"{first.id}:conversation"
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            ids = [m.id for m in window]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    message_id=first.id,
                    chat=first.chat,
                    sender="conversation",
                    timestamp=first.timestamp,
                    date_str=first.date_str,
                    modality="conversation",
                    content="\n".join(lines),
                    media_path=None,
                    extra={
                        "message_ids": ids,
                        "start_id": ids[0],
                        "end_id": ids[-1],
                    },
                )
            )
            if start + window_size >= n:
                break  # last window reached the end of the session
    return chunks


def _describe(msg: Message) -> str:
    """Short human-readable description of what a message will process."""
    media_type = getattr(msg, "media_type", None)
    media_path = getattr(msg, "media_path", None)
    file_name = getattr(msg, "file_name", None)
    bits = []
    if media_type:
        bits.append(media_type)
    if media_path:
        bits.append(media_path)
    if file_name and file_name != media_path:
        bits.append(file_name)
    return ", ".join(bits) if bits else "text-only"


class _HangWatchdog:
    """Dump in-flight messages + all thread stacks if progress stalls.

    The build runs media decoding/captioning across a thread pool. A single
    item stuck in an uninterruptible C call (e.g. a corrupt image in PIL) can't
    be force-killed, so when nothing completes for ``timeout`` seconds we print
    exactly what is in flight and where every thread is parked — turning an
    opaque freeze into an actionable report.
    """

    def __init__(self, timeout: float, inflight: dict[int, str]):
        self.timeout = timeout
        self.inflight = inflight
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _fire(self) -> None:
        tqdm.write(
            f"\n[hang] no message completed for {self.timeout:.0f}s. "
            f"Likely one item is stuck. Currently in flight:"
        )
        for mid, desc in list(self.inflight.items()):
            tqdm.write(f"[hang]   msg {mid}: {desc}")
        tqdm.write("[hang] thread stacks follow (look for the worker thread):")
        faulthandler.dump_traceback(file=sys.stderr)
        self.reset()  # keep reporting while still stuck

    def reset(self) -> None:
        if not self.timeout or self.timeout <= 0:
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.timeout, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def _process_block(block, process, pool, bar, item_timeout: float, watchdog=None):
    """Run ``process`` over a block, bounding each message by ``item_timeout``.

    A message that hangs (e.g. a corrupt image whose decode never returns, or a
    stuck remote call) is logged and skipped instead of freezing the whole run.
    Progress is reported per message so a slow item is visible immediately.
    """
    from concurrent.futures import TimeoutError as FuturesTimeout

    timeout = item_timeout if item_timeout and item_timeout > 0 else None
    results: list = []

    def _advance() -> None:
        bar.update(1)
        if watchdog is not None:
            watchdog.reset()

    if pool is None:
        for msg in block:
            try:
                results.append(process(msg))
            except Exception as exc:  # pragma: no cover - robustness for big exports
                tqdm.write(f"[warn] msg {msg.id} failed: {exc}")
                results.append([])
            _advance()
        return results

    futures = [(msg, pool.submit(process, msg)) for msg in block]
    for msg, fut in futures:
        try:
            results.append(fut.result(timeout=timeout))
        except FuturesTimeout:
            tqdm.write(
                f"[warn] msg {msg.id} timed out after {timeout:.0f}s ({_describe(msg)}); "
                f"skipping. (raise TELESEARCH_MEDIA_ITEM_TIMEOUT if this is legit media)"
            )
            fut.cancel()
            results.append([])
        except Exception as exc:  # pragma: no cover - robustness for big exports
            tqdm.write(f"[warn] msg {msg.id} failed: {exc}")
            results.append([])
        _advance()
    return results


def build_index(
    messages: Iterable[Message],
    export_root: str | Path,
    settings: Settings,
    *,
    do_images: bool = True,
    do_videos: bool = True,
    do_audio: bool = True,
    do_ocr: bool = True,
    do_documents: bool = True,
    do_conversation_windows: bool = True,
    resume: bool = True,
    rebuild: bool = False,
    workers: int | None = None,
    embed_batch: int | None = None,
) -> int:
    """Build the LanceDB index from parsed messages. Returns new chunk count.

    The build is **resumable**: messages already present in the index are
    skipped (unless ``rebuild=True``), and each block is persisted as it is
    processed, so a long run over thousands of media files can be safely
    interrupted and restarted. Media understanding (remote VLM captioning/OCR,
    frame extraction) runs **concurrently** across ``workers`` threads to keep
    the GPU server busy; local Whisper transcription is serialized internally.
    """
    from concurrent.futures import ThreadPoolExecutor

    export_root = Path(export_root)
    workers = workers or settings.media_workers
    if embed_batch is None:
        embed_batch = settings.embed_batch_size

    captioner = None
    transcriber = None
    if do_images or do_videos:
        from ..media.captioner import VLMCaptioner

        captioner = VLMCaptioner(settings)
    if do_audio or do_videos:
        from ..media.asr import Transcriber

        transcriber = Transcriber(settings)

    embedder = TextEmbedder(settings)
    store = VectorStore(settings.db_path, embedder.dim)
    # Surface the effective embedding memory knobs so a stale image (built before
    # these were added) is obvious: if you don't see this line, rebuild the image.
    tqdm.write(
        f"[embed] {settings.text_embed_model} on {settings.device} | "
        f"batch_size={embed_batch} | max_seq_length={embedder.max_seq_length}"
    )

    # Materialize the full set first so replies can be linked to their parent
    # (the parent may already be indexed and thus filtered out below).
    all_messages = list(messages)
    reply_lookup = {m.id: m for m in all_messages}

    if rebuild:
        store.drop()
        seen: set[int] = set()
    else:
        seen = store.existing_message_ids() if resume else set()

    messages = [m for m in all_messages if m.id not in seen]
    if seen:
        tqdm.write(f"[resume] skipping {len(seen)} already-indexed messages")

    # Track what each worker is currently chewing on so the hang watchdog can
    # name the exact message/file if the build stalls.
    inflight: dict[int, str] = {}

    def process(msg: Message) -> list[Chunk]:
        inflight[msg.id] = _describe(msg)
        try:
            return _message_to_chunks(
                msg,
                export_root,
                captioner=captioner if (do_images or do_videos) else None,
                transcriber=transcriber if (do_audio or do_videos) else None,
                num_frames=settings.video_frames,
                do_ocr=do_ocr and do_images,
                do_documents=do_documents,
                doc_chunk_chars=settings.doc_chunk_chars,
                doc_chunk_overlap=settings.doc_chunk_overlap,
                doc_max_chars=settings.doc_max_chars,
                reply_lookup=reply_lookup,
            )
        finally:
            inflight.pop(msg.id, None)

    total = 0
    # Block size balances throughput (concurrency window) with how often we
    # persist progress for resumability.
    block_size = max(embed_batch, workers * 4)
    # Always use a pool so the per-message timeout guard applies even at
    # workers=1 (a single hung file would otherwise stall the whole build).
    pool = ThreadPoolExecutor(max_workers=max(workers, 1))
    watchdog = _HangWatchdog(settings.hang_traceback_seconds, inflight)

    try:
        with tqdm(total=len(messages), desc="indexing", unit="msg") as bar:
            watchdog.reset()
            for block in _iter_blocks(messages, block_size):
                results = _process_block(
                    block, process, pool, bar, settings.media_item_timeout, watchdog
                )

                chunks = [c for group in results for c in group]
                if chunks:
                    vectors = embedder.encode(
                        [c.content for c in chunks], batch_size=embed_batch
                    )
                    store.add([c.to_row() for c in chunks], vectors)
                    total += len(chunks)
    finally:
        watchdog.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    if do_conversation_windows and settings.enable_conversation_windows:
        conv_chunks = _build_conversation_chunks(
            messages,
            window_size=settings.conversation_window_size,
            stride=settings.conversation_window_stride,
            max_gap=settings.conversation_window_max_gap,
        )
        if conv_chunks:
            tqdm.write(
                f"[windows] indexing {len(conv_chunks)} conversation-window chunks "
                f"(size={settings.conversation_window_size}, "
                f"stride={settings.conversation_window_stride})"
            )
            for block in _iter_blocks(conv_chunks, embed_batch):
                vectors = embedder.encode(
                    [c.content for c in block], batch_size=embed_batch
                )
                store.add([c.to_row() for c in block], vectors)
                total += len(block)

    store.build_fts()
    return total


def reindex_text(
    messages: Iterable[Message],
    export_root: str | Path,
    settings: Settings,
    *,
    do_conversation_windows: bool = True,
) -> int:
    """Refresh only the *text-derived* chunks over an existing index.

    Rebuilds ``text`` (with reply stitching) and ``conversation`` window chunks
    from the export, leaving the expensive media chunks (image captions, video
    summaries, voice transcripts, OCR, document text) untouched. This lets you
    adopt conversation-window context on a large, already-indexed export
    **without** re-running the VLM/Whisper over thousands of media files — so it
    needs no vLLM server (run with ``--no-deps``). Returns the new chunk count.
    """
    export_root = Path(export_root)
    all_messages = list(messages)
    reply_lookup = {m.id: m for m in all_messages}

    embedder = TextEmbedder(settings)
    store = VectorStore(settings.db_path, embedder.dim)
    embed_batch = settings.embed_batch_size

    removed = store.delete_modalities(["text", "conversation"])
    tqdm.write(f"[reindex-text] removed {removed} existing text/conversation chunks")

    # Text-only pass: captioner/transcriber/documents disabled, so only the
    # per-message text chunk (with reply context) is produced.
    chunks: list[Chunk] = []
    for msg in all_messages:
        chunks.extend(
            _message_to_chunks(
                msg,
                export_root,
                captioner=None,
                transcriber=None,
                num_frames=0,
                do_ocr=False,
                do_documents=False,
                reply_lookup=reply_lookup,
            )
        )

    if do_conversation_windows and settings.enable_conversation_windows:
        conv_chunks = _build_conversation_chunks(
            all_messages,
            window_size=settings.conversation_window_size,
            stride=settings.conversation_window_stride,
            max_gap=settings.conversation_window_max_gap,
        )
        chunks.extend(conv_chunks)
        tqdm.write(
            f"[reindex-text] adding {len(conv_chunks)} conversation-window chunks "
            f"(size={settings.conversation_window_size}, "
            f"stride={settings.conversation_window_stride})"
        )

    total = 0
    with tqdm(total=len(chunks), desc="reindex-text", unit="chunk") as bar:
        for block in _iter_blocks(chunks, embed_batch):
            vectors = embedder.encode([c.content for c in block], batch_size=embed_batch)
            store.add([c.to_row() for c in block], vectors)
            total += len(block)
            bar.update(len(block))

    store.build_fts()
    return total
