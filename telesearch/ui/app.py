"""Minimal Streamlit UI.

Run with:
    streamlit run telesearch/ui/app.py

The export root is read from the TELESEARCH_EXPORT_ROOT env var (so media
thumbnails can be displayed). Falls back to not showing images if unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from telesearch.config import get_settings
from telesearch.search import Retriever, answer_question

st.set_page_config(page_title="telesearch", layout="wide")
settings = get_settings()
export_root = os.environ.get("TELESEARCH_EXPORT_ROOT")

# All searchable modalities (must match what the indexer produces).
MODALITIES = [
    "any",
    "text",
    "conversation",
    "image",
    "video",
    "audio",
    "ocr",
    "document",
]

# Don't load whole huge files into memory just to offer a download button.
_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(settings)


def _render_content(content: str) -> None:
    """Render result text preserving line breaks.

    ``st.write`` treats strings as Markdown, where single newlines collapse —
    which mangles multi-line conversation windows and transcripts. ``st.text``
    keeps the original line breaks and won't interpret user text as Markdown.
    """
    st.text(content)


def _render_media(r) -> None:
    """Show the underlying media for a result, by modality."""
    if not (r.media_path and export_root):
        return
    media_file = Path(export_root) / r.media_path
    if not media_file.exists():
        return
    if r.modality == "image":
        st.image(str(media_file), width=320)
    elif r.modality == "video":
        st.video(str(media_file))
    elif r.modality == "audio":
        st.audio(str(media_file))
    elif r.modality == "document":
        try:
            if media_file.stat().st_size <= _MAX_DOWNLOAD_BYTES:
                st.download_button(
                    "Download file",
                    media_file.read_bytes(),
                    file_name=media_file.name,
                    key=f"dl-{r.chunk_id}",
                )
            else:
                st.caption(f"File: {media_file.name}")
        except Exception:
            st.caption(f"File: {media_file.name}")


st.title("telesearch — chat search")

mode = st.radio("Mode", ["Search", "Ask (RAG)"], horizontal=True)
query = st.text_input("Query")

col_k, col_type = st.columns(2)
with col_k:
    k = st.slider("Results", 3, 30, 10)
with col_type:
    if mode == "Search":
        modality = st.selectbox("Type", MODALITIES)
    else:
        modality = "any"
        st.caption("Ask searches across all message types.")

# Retrieval controls map directly onto the backend (HyDE / neighbour expansion
# for Ask, cross-encoder rerank for Search).
use_hyde = settings.enable_hyde
neighbors = settings.context_neighbors
rerank = settings.use_reranker
with st.expander("Advanced"):
    if mode == "Ask (RAG)":
        use_hyde = st.checkbox(
            "HyDE — draft a hypothetical answer to improve recall",
            value=settings.enable_hyde,
        )
        neighbors = st.slider(
            "Surrounding messages for context (each side of a hit)",
            0,
            12,
            settings.context_neighbors,
        )
    else:
        rerank = st.checkbox(
            "Cross-encoder rerank (sharper precision)", value=settings.use_reranker
        )

if query:
    retriever = get_retriever()
    if mode == "Ask (RAG)":
        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                query,
                settings,
                k=k,
                retriever=retriever,
                use_hyde=use_hyde,
                neighbors=neighbors,
            )
        st.markdown(answer)
        st.subheader("Sources / citations")
        results = sources
    else:
        mod = None if modality == "any" else modality
        with st.spinner("Searching..."):
            results = retriever.search(query, k=k, modality=mod, rerank=rerank)
        st.subheader("Results")

    if not results:
        st.info("No results.")
    for r in results:
        with st.container(border=True):
            st.caption(
                f"msg {r.message_id} · {r.modality} · {r.sender} · "
                f"{r.date_str} · score {r.score:.3f}"
            )
            _render_content(r.content)
            _render_media(r)
