"""Answer natural-language questions over the conversation (RAG).

Retrieves the most relevant chunks, then asks a local open-weight chat model to
synthesize an answer grounded in that context, citing message ids.

Two things make this work well on *chat* data (as opposed to documents):

* **HyDE** — a question ("what hotel did we book in Rome?") and the message that
  answers it ("booked the Ritz, check-in 3pm") share almost no words, so a raw
  question embeds poorly. We first have the model draft a hypothetical answer
  and retrieve with that too, which bridges the gap (recall).
* **Neighbour expansion** — a single matched message is rarely enough to answer
  from; we also pull the messages just before/after each hit so the model sees
  the surrounding conversation (context).
"""

from __future__ import annotations

from collections import defaultdict

from openai import OpenAI

from ..config import Settings
from .retriever import Retriever, SearchResult

_SYSTEM_PROMPT = (
    "You answer questions about a personal Telegram conversation using only the "
    "provided excerpts. Excerpts may be typed messages, photo descriptions, "
    "video summaries or voice transcripts, and are shown in chronological order "
    "with surrounding messages included for context. Cite the message ids you "
    "rely on like [msg 1234]. If the excerpts do not contain the answer, say so "
    "plainly."
)

_HYDE_SYSTEM_PROMPT = (
    "You help a search engine find messages in a personal chat. Given a "
    "question, write a short, plausible hypothetical chat message (1-3 "
    "sentences) that would *contain the answer*. Be concrete and specific; "
    "invent plausible details. Output only the hypothetical message text, with "
    "no preamble or quotation marks."
)

# Modalities worth pulling in as surrounding context (skip verbose document /
# OCR chunks, which are already retrieved directly when relevant).
_NEIGHBOR_MODALITIES = ("text", "conversation", "image", "audio", "video")

# Hard cap on how many chunks we feed the model, to bound the prompt size.
_MAX_CONTEXT_CHUNKS = 60


def _hyde_query(question: str, client: OpenAI, settings: Settings) -> str:
    """Draft a hypothetical answering message to retrieve with (HyDE)."""
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
        temperature=0.3,
        max_tokens=120,
    )
    return (resp.choices[0].message.content or "").strip()


def _gather_context(
    results: list[SearchResult],
    store,
    neighbors: int,
) -> tuple[list[SearchResult], set[str]]:
    """Merge the hits with their message-id neighbours, in chronological order.

    Returns ``(chunks, hit_ids)`` where ``hit_ids`` marks the chunks that were
    actual retrieval matches (vs. context pulled in around them).
    """
    by_chunk: dict[str, SearchResult] = {r.chunk_id: r for r in results}
    hit_ids = set(by_chunk)

    if neighbors > 0 and store is not None:
        ids_by_chat: dict[str, set[int]] = defaultdict(set)
        for r in results:
            ids_by_chat[r.chat].add(r.message_id)
        for chat, ids in ids_by_chat.items():
            rows = store.fetch_around(
                chat,
                sorted(ids),
                before=neighbors,
                after=neighbors,
                modalities=_NEIGHBOR_MODALITIES,
            )
            for row in rows:
                sr = SearchResult.from_row(row)
                by_chunk.setdefault(sr.chunk_id, sr)

    merged = sorted(by_chunk.values(), key=lambda r: (r.message_id, r.chunk_id))
    if len(merged) > _MAX_CONTEXT_CHUNKS:
        # Keep all hits, then fill remaining budget with the nearest context.
        kept = [r for r in merged if r.chunk_id in hit_ids]
        extra = [r for r in merged if r.chunk_id not in hit_ids]
        merged = sorted(
            kept + extra[: max(0, _MAX_CONTEXT_CHUNKS - len(kept))],
            key=lambda r: (r.message_id, r.chunk_id),
        )
    return merged, hit_ids


def _format_context(chunks: list[SearchResult]) -> str:
    lines = []
    for r in chunks:
        tag = f"[msg {r.message_id} | {r.modality} | {r.sender} | {r.date_str}]"
        lines.append(f"{tag}\n{r.content}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    settings: Settings,
    *,
    k: int = 12,
    retriever: Retriever | None = None,
    use_hyde: bool | None = None,
    neighbors: int | None = None,
) -> tuple[str, list[SearchResult]]:
    """Return ``(answer, sources)`` for a question.

    ``sources`` are the direct retrieval matches (for citation/display); the
    model additionally sees the surrounding messages as grounding context.
    """
    retriever = retriever or Retriever(settings)
    use_hyde = settings.enable_hyde if use_hyde is None else use_hyde
    neighbors = settings.context_neighbors if neighbors is None else neighbors

    client = OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_request_timeout,
        max_retries=settings.llm_max_retries,
    )

    # HyDE: retrieve with question + a hypothetical answer, but rerank against
    # the original question so precision is judged on what was actually asked.
    retrieval_query = question
    if use_hyde:
        try:
            hypothetical = _hyde_query(question, client, settings)
            if hypothetical:
                retrieval_query = f"{question}\n{hypothetical}"
        except Exception:
            retrieval_query = question

    results = retriever.search(retrieval_query, k=k, rerank_query=question)
    if not results:
        return ("I couldn't find anything relevant in the conversation.", [])

    chunks, _hit_ids = _gather_context(
        results, getattr(retriever, "store", None), neighbors
    )
    context = _format_context(chunks)
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Conversation excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=600,
    )
    answer = (resp.choices[0].message.content or "").strip()
    return answer, results
