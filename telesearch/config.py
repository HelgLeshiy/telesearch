"""Central configuration, loaded from environment variables / a .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for telesearch.

    Values are read from environment variables prefixed with ``TELESEARCH_``
    (see ``.env.example``). Everything has a default so the package is usable
    out of the box once the local model servers are running.
    """

    model_config = SettingsConfigDict(
        env_prefix="TELESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")

    # Local embedding models.
    text_embed_model: str = "BAAI/bge-m3"
    image_embed_model: str = "jinaai/jina-clip-v2"
    device: str = "cuda"

    # Cross-encoder reranker (re-scores the top candidates for precision).
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True
    rerank_candidates: int = 50

    # OCR: extract verbatim on-image text as its own searchable field.
    enable_ocr: bool = True

    # Document attachments (PDF, Office, text/code/CSV, ...).
    enable_documents: bool = True
    doc_chunk_chars: int = 1200
    doc_chunk_overlap: int = 150
    doc_max_chars: int = 400_000

    # OpenAI-compatible endpoint for the VLM / chat model (vLLM, SGLang, ...).
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    vlm_model: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    chat_model: str = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Audio transcription.
    whisper_model: str = "large-v3"
    whisper_compute: str = "float16"
    video_frames: int = 4

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def media_cache_dir(self) -> Path:
        return self.data_dir / "media_cache"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
