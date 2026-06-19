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
from PIL import Image

from ..config import Settings

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
    """Downscale and JPEG-encode an image as a base64 data URL."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class VLMCaptioner:
    """Generate text captions for images and groups of video frames."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
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

    def caption_image(self, image_path: str | Path) -> str:
        image = Image.open(image_path)
        return self._caption([_image_to_data_url(image)], _IMAGE_PROMPT)

    def ocr_image(self, image_path: str | Path) -> str:
        """Return verbatim on-image text, or "" if none is detected."""
        image = Image.open(image_path)
        text = self._caption([_image_to_data_url(image)], _OCR_PROMPT)
        cleaned = text.strip()
        if not cleaned or cleaned.strip().upper() == "NONE":
            return ""
        return cleaned

    def caption_frames(self, frames: list[Image.Image]) -> str:
        if not frames:
            return ""
        urls = [_image_to_data_url(f) for f in frames]
        return self._caption(urls, _VIDEO_PROMPT)
