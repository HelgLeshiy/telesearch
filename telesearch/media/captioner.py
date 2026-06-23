"""Caption images and video frames with an open-weight vision-language model.

The model is reached through an OpenAI-compatible endpoint (vLLM, SGLang,
Ollama, ...), so any served VLM that speaks that protocol works. Captions are
plain text that we later embed and index, which makes pictures and video
content searchable with ordinary natural-language queries.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageFile

from ..config import Settings

# Telegram exports occasionally contain truncated/partially-downloaded images.
# Be lenient so a damaged file yields a (possibly partial) image instead of
# hanging or raising deep inside the decoder.
ImageFile.LOAD_TRUNCATED_IMAGES = True

_IMAGE_PROMPT = (
    "You are indexing photos from a personal chat so they can be found later "
    "by text search. Describe this image in 1-3 detailed sentences. Mention "
    "visible objects, people and their actions, setting, text/signs, and any "
    "notable details. Be concrete and specific. Do not add commentary."
)

_VIDEO_PROMPT = (
    "These frames are sampled from a single short video shared in a personal "
    "chat. Summarize what happens in the video in 1-4 sentences so it can be "
    "found later by text search. Mention people, actions, objects, setting and "
    "any visible text. Do not describe the frames individually."
)

_OCR_PROMPT = (
    "Transcribe ALL readable text visible in this image verbatim, preserving "
    "the original language. Include signs, documents, screenshots, captions and "
    "handwriting. Output only the transcribed text, with no commentary. If there "
    "is no readable text, output exactly: NONE"
)


def _image_to_data_url(image: Image.Image, max_side: int = 1024) -> str:
    """Downscale and JPEG-encode an in-memory image as a base64 data URL."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _path_to_data_url(
    image_path: str | Path,
    max_side: int = 1024,
    max_megapixels: float = 50.0,
) -> str:
    """Open an image file safely and return it as a JPEG data URL.

    Uses the JPEG decoder's ``draft`` mode to downscale *before* a full decode,
    which keeps very large photos cheap to process, and rejects images above
    ``max_megapixels`` so a decompression bomb can't hang or OOM the worker.
    """
    with Image.open(image_path) as image:
        # `draft` lets the JPEG decoder load at a reduced resolution directly,
        # avoiding a multi-hundred-megapixel full decode for huge photos.
        try:
            image.draft("RGB", (max_side, max_side))
        except (OSError, ValueError):
            pass

        if max_megapixels and max_megapixels > 0:
            megapixels = (image.width * image.height) / 1_000_000
            if megapixels > max_megapixels:
                raise ValueError(
                    f"image too large to process: {megapixels:.0f} MP "
                    f"({image.width}x{image.height}) exceeds {max_megapixels:.0f} MP cap"
                )

        image.load()
        return _image_to_data_url(image, max_side=max_side)


class VLMCaptioner:
    """Generate text captions for images and groups of video frames."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
        )
        self.model = settings.vlm_model

    def _caption(self, data_urls: list[str], prompt: str) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=256,
        )
        return (resp.choices[0].message.content or "").strip()

    def _data_url(self, image_path: str | Path) -> str:
        return _path_to_data_url(
            image_path, max_megapixels=self.settings.max_image_megapixels
        )

    def caption_image(self, image_path: str | Path) -> str:
        return self._caption([self._data_url(image_path)], _IMAGE_PROMPT)

    def ocr_image(self, image_path: str | Path) -> str:
        """Return verbatim on-image text, or "" if none is detected."""
        text = self._caption([self._data_url(image_path)], _OCR_PROMPT)
        cleaned = text.strip()
        if not cleaned or cleaned.strip().upper() == "NONE":
            return ""
        return cleaned

    def caption_frames(self, frames: list[Image.Image]) -> str:
        if not frames:
            return ""
        urls = [_image_to_data_url(f) for f in frames]
        return self._caption(urls, _VIDEO_PROMPT)
