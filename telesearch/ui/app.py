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


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(settings)


st.title("telesearch — chat search")

mode = st.radio("Mode", ["Search", "Ask (RAG)"], horizontal=True)
query = st.text_input("Query")
modality = st.selectbox("Type", ["any", "text", "image", "video", "audio"])
k = st.slider("Results", 3, 30, 10)

if query:
    retriever = get_retriever()
    if mode == "Ask (RAG)":
        with st.spinner("Thinking..."):
            answer, sources = answer_question(query, settings, k=k, retriever=retriever)
        st.markdown(answer)
        results = sources
    else:
        mod = None if modality == "any" else modality
        results = retriever.search(query, k=k, modality=mod)

    st.subheader("Results")
    for r in results:
        with st.container(border=True):
            st.caption(f"{r.modality} · {r.sender} · {r.date_str} · score {r.score:.3f}")
            st.write(r.content)
            if r.media_path and export_root:
                media_file = Path(export_root) / r.media_path
                if media_file.exists() and r.modality == "image":
                    st.image(str(media_file), width=320)
                elif media_file.exists() and r.modality == "video":
                    st.video(str(media_file))
