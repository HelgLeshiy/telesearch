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
) -> list[Chunk]:
    """Turn one message into one or more searchable chunks."""
    chunks: list[Chunk] = []

    def base(modality: str, content: str, media_path: Optional[str], extra=None) -> Chunk:
        return Chunk(
            chunk_id=f"{msg.id}:{modality}",
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


def build_index(
    messages: Iterable[Message],
    export_root: str | Path,
    settings: Settings,
    *,
    do_images: bool = True,
    do_videos: bool = True,
    do_audio: bool = True,
    do_ocr: bool = True,
    embed_batch: int = 256,
) -> int:
    """Build the LanceDB index from parsed messages. Returns chunk count."""
    export_root = Path(export_root)

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

    buffer: list[Chunk] = []
    total = 0

    def flush() -> None:
        nonlocal total
        if not buffer:
            return
        vectors = embedder.encode([c.content for c in buffer], batch_size=embed_batch)
        store.add([c.to_row() for c in buffer], vectors)
        total += len(buffer)
        buffer.clear()

    for msg in tqdm(messages, desc="indexing", unit="msg"):
        buffer.extend(
            _message_to_chunks(
                msg,
                export_root,
                captioner=captioner if (do_images or do_videos) else None,
                transcriber=transcriber if (do_audio or do_videos) else None,
                num_frames=settings.video_frames,
                do_ocr=do_ocr and do_images,
            )
        )
        if len(buffer) >= embed_batch:
            flush()

    flush()
    store.build_fts()
    return total
