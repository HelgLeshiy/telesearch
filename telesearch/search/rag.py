"""Answer natural-language questions over the conversation (RAG).

Retrieves the most relevant chunks, then asks a local open-weight chat model to
synthesize an answer grounded in that context, citing message ids.
"""

from __future__ import annotations

from openai import OpenAI

from ..config import Settings
from .retriever import Retriever, SearchResult

_SYSTEM_PROMPT = (
    "You answer questions about a personal Telegram conversation using only the "
    "provided excerpts. Excerpts may be typed messages, photo descriptions, "
    "video summaries or voice transcripts. Cite the message ids you rely on like "
    "[msg 1234]. If the excerpts do not contain the answer, say so plainly."
)


def _format_context(results: list[SearchResult]) -> str:
    lines = []
    for r in results:
        tag = f"[msg {r.message_id} | {r.modality} | {r.sender} | {r.date_str}]"
        lines.append(f"{tag}\n{r.content}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    settings: Settings,
    *,
    k: int = 12,
    retriever: Retriever | None = None,
) -> tuple[str, list[SearchResult]]:
    """Return ``(answer, sources)`` for a question."""
    retriever = retriever or Retriever(settings)
    results = retriever.search(question, k=k)

    if not results:
        return ("I couldn't find anything relevant in the conversation.", [])

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    context = _format_context(results)
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
