"""End-to-end indexing pipeline: export -> chunks -> embeddings -> LanceDB."""

from __future__ import annotations

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
        chunks.append(base("text", msg.text, None))

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


def _process_block(block, process, pool, bar, item_timeout: float):
    """Run ``process`` over a block, bounding each message by ``item_timeout``.

    A message that hangs (e.g. a corrupt image whose decode never returns, or a
    stuck remote call) is logged and skipped instead of freezing the whole run.
    Progress is reported per message so a slow item is visible immediately.
    """
    from concurrent.futures import TimeoutError as FuturesTimeout

    timeout = item_timeout if item_timeout and item_timeout > 0 else None
    results: list = []

    if pool is None:
        for msg in block:
            try:
                results.append(process(msg))
            except Exception as exc:  # pragma: no cover - robustness for big exports
                tqdm.write(f"[warn] msg {msg.id} failed: {exc}")
                results.append([])
            bar.update(1)
        return results

    futures = [(msg, pool.submit(process, msg)) for msg in block]
    for msg, fut in futures:
        try:
            results.append(fut.result(timeout=timeout))
        except FuturesTimeout:
            tqdm.write(
                f"[warn] msg {msg.id} timed out after {timeout:.0f}s; skipping. "
                f"(stuck media/document — raise TELESEARCH_MEDIA_ITEM_TIMEOUT if legit)"
            )
            fut.cancel()
            results.append([])
        except Exception as exc:  # pragma: no cover - robustness for big exports
            tqdm.write(f"[warn] msg {msg.id} failed: {exc}")
            results.append([])
        bar.update(1)
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

    if rebuild:
        store.drop()
        seen: set[int] = set()
    else:
        seen = store.existing_message_ids() if resume else set()

    messages = [m for m in messages if m.id not in seen]
    if seen:
        tqdm.write(f"[resume] skipping {len(seen)} already-indexed messages")

    def process(msg: Message) -> list[Chunk]:
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
        )

    total = 0
    # Block size balances throughput (concurrency window) with how often we
    # persist progress for resumability.
    block_size = max(embed_batch, workers * 4)
    # Always use a pool so the per-message timeout guard applies even at
    # workers=1 (a single hung file would otherwise stall the whole build).
    pool = ThreadPoolExecutor(max_workers=max(workers, 1))

    try:
        with tqdm(total=len(messages), desc="indexing", unit="msg") as bar:
            for block in _iter_blocks(messages, block_size):
                results = _process_block(
                    block, process, pool, bar, settings.media_item_timeout
                )

                chunks = [c for group in results for c in group]
                if chunks:
                    vectors = embedder.encode(
                        [c.content for c in chunks], batch_size=embed_batch
                    )
                    store.add([c.to_row() for c in chunks], vectors)
                    total += len(chunks)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    store.build_fts()
    return total
