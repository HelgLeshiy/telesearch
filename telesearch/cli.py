"""Command-line interface for telesearch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

app = typer.Typer(
    add_completion=False,
    help="Local multimodal AI search over a Telegram conversation export.",
)
console = Console()


@app.command()
def index(
    export: Path = typer.Argument(
        ...,
        help="Path to a source: a Telegram export (folder/result.json), a file, or a folder of files.",
    ),
    kind: Optional[str] = typer.Option(
        None,
        help="Force a parser (e.g. telegram, generic_text, generic_file). Default: auto-detect.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        help="Collection id grouping this source for scoped search (default: source name).",
    ),
    workspace: Optional[str] = typer.Option(
        None, help="Workspace whose index to write (default: config default_workspace)."
    ),
    images: bool = typer.Option(True, help="Caption photos with the VLM."),
    videos: bool = typer.Option(True, help="Summarize and transcribe videos."),
    audio: bool = typer.Option(True, help="Transcribe voice messages."),
    ocr: bool = typer.Option(True, help="Extract verbatim on-image text (OCR) as a separate chunk."),
    documents: bool = typer.Option(True, help="Extract & index text from file attachments (PDF, Office, text/code)."),
    conversation_windows: bool = typer.Option(
        True,
        "--conversation-windows/--no-conversation-windows",
        help="Also index overlapping windows of consecutive messages for conversational context.",
    ),
    rebuild: bool = typer.Option(False, help="Drop the existing index and start over."),
    resume: bool = typer.Option(True, help="Skip messages that are already indexed."),
    workers: Optional[int] = typer.Option(None, help="Concurrent media requests (default: config)."),
):
    """Parse a source and build the searchable index (resumable)."""
    from .service import RequestContext, index_source

    settings = get_settings()
    ctx = RequestContext(workspace_id=workspace or settings.default_workspace)

    console.print(f"[bold]Indexing[/bold] {export}")
    result = index_source(
        export,
        settings,
        ctx=ctx,
        kind=kind,
        collection_id=collection,
        do_images=images,
        do_videos=videos,
        do_audio=audio,
        do_ocr=ocr,
        do_documents=documents,
        do_conversation_windows=conversation_windows,
        rebuild=rebuild,
        resume=resume,
        workers=workers,
    )
    console.print(
        f"[bold green]Done.[/bold green] Parser [cyan]{result.parser}[/cyan] read "
        f"{result.messages} messages; indexed {result.chunks} new chunks into "
        f"{result.db_path} (collection [magenta]{result.collection_id}[/magenta])"
    )


@app.command(name="reindex-text")
def reindex_text(
    export: Path = typer.Argument(
        ...,
        help="Path to the source (Telegram export folder/result.json, file, or folder).",
    ),
    kind: Optional[str] = typer.Option(
        None, help="Force a parser (default: auto-detect)."
    ),
    collection: Optional[str] = typer.Option(
        None, help="Collection id (default: source name)."
    ),
    workspace: Optional[str] = typer.Option(
        None, help="Workspace whose index to write (default: config default_workspace)."
    ),
    conversation_windows: bool = typer.Option(
        True,
        "--conversation-windows/--no-conversation-windows",
        help="Index overlapping windows of consecutive messages for conversational context.",
    ),
):
    """Refresh only text + conversation chunks (no media re-processing, no vLLM).

    Use this to add conversation-window context and reply stitching to an
    existing index without the cost of a full ``--rebuild`` (which would
    re-caption every photo/video and re-transcribe all audio). Media chunks are
    left untouched. Safe to run with ``--no-deps``.
    """
    from .service import RequestContext, reindex_source_text

    settings = get_settings()
    ctx = RequestContext(workspace_id=workspace or settings.default_workspace)

    console.print(f"[bold]Refreshing text + conversation chunks for[/bold] {export}")
    result = reindex_source_text(
        export,
        settings,
        ctx=ctx,
        kind=kind,
        collection_id=collection,
        do_conversation_windows=conversation_windows,
    )
    console.print(
        f"[bold green]Done.[/bold green] Wrote {result.chunks} text/conversation "
        f"chunks into {result.db_path}"
    )


def _parse_date(value: Optional[str]) -> Optional[int]:
    """Parse a YYYY-MM-DD (or ISO) date string into a unix timestamp."""
    if not value:
        return None
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid date {value!r} (use YYYY-MM-DD)") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(10, help="Number of results."),
    modality: Optional[str] = typer.Option(
        None,
        help="Filter by type: text, conversation, image, video, audio, ocr, document.",
    ),
    collection: Optional[list[str]] = typer.Option(
        None, "--collection", help="Restrict to / combine these collection ids (repeatable)."
    ),
    sender: Optional[list[str]] = typer.Option(
        None, "--sender", help="Restrict to these senders (repeatable)."
    ),
    source_kind: Optional[list[str]] = typer.Option(
        None, "--source-kind", help="Restrict to these source kinds, e.g. telegram (repeatable)."
    ),
    since: Optional[str] = typer.Option(None, help="Only results on/after this date (YYYY-MM-DD)."),
    until: Optional[str] = typer.Option(None, help="Only results on/before this date (YYYY-MM-DD)."),
    workspace: Optional[str] = typer.Option(
        None, help="Workspace to search (default: config default_workspace)."
    ),
    rerank: Optional[bool] = typer.Option(
        None, "--rerank/--no-rerank", help="Cross-encoder rerank (default: config)."
    ),
):
    """Hybrid (semantic + keyword) search with filters, then cross-encoder rerank."""
    from .service import RequestContext, SearchQuery, SearchService

    settings = get_settings()
    ctx = RequestContext(workspace_id=workspace or settings.default_workspace)
    service = SearchService(settings, ctx)
    results = service.search(
        SearchQuery(
            text=query,
            k=k,
            modalities=[modality] if modality else None,
            collections=list(collection) if collection else None,
            senders=list(sender) if sender else None,
            source_kinds=list(source_kind) if source_kind else None,
            date_from=_parse_date(since),
            date_to=_parse_date(until),
            rerank=rerank,
        )
    )

    if not results:
        console.print("[yellow]No results.[/yellow]")
        raise typer.Exit()

    table = Table(show_lines=True)
    table.add_column("score", justify="right", style="cyan", no_wrap=True)
    table.add_column("type", style="magenta", no_wrap=True)
    table.add_column("when / who", style="green", no_wrap=True)
    table.add_column("content")
    for r in results:
        content = r.content if len(r.content) < 400 else r.content[:400] + "..."
        if r.media_path:
            content += f"\n[dim]{r.media_path}[/dim]"
        table.add_row(f"{r.score:.3f}", r.modality, f"{r.date_str}\n{r.sender}", content)
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question."),
    k: int = typer.Option(12, help="Number of context chunks to retrieve."),
    hyde: Optional[bool] = typer.Option(
        None,
        "--hyde/--no-hyde",
        help="Draft a hypothetical answer to improve retrieval recall (default: config).",
    ),
    neighbors: Optional[int] = typer.Option(
        None,
        help="Messages of surrounding context to include on each side of a hit (default: config).",
    ),
    workspace: Optional[str] = typer.Option(
        None, help="Workspace to query (default: config default_workspace)."
    ),
):
    """Ask a question and get an answer grounded in the conversation (RAG)."""
    from .service import RequestContext, SearchService

    settings = get_settings()
    ctx = RequestContext(workspace_id=workspace or settings.default_workspace)
    service = SearchService(settings, ctx)
    answer, sources = service.ask(question, k=k, use_hyde=hyde, neighbors=neighbors)

    console.print(Panel(answer, title="Answer", border_style="green"))
    if sources:
        console.print("[bold]Sources:[/bold]")
        for s in sources:
            console.print(
                f"  [cyan]msg {s.message_id}[/cyan] ({s.modality}, {s.sender}, {s.date_str})"
            )


@app.command()
def info():
    """Show current configuration and index status."""
    from .index.store import VectorStore, TABLE_NAME
    from .index.embeddings import TextEmbedder
    from .ingest import available as available_parsers

    settings = get_settings()
    table = Table(title="telesearch configuration")
    table.add_column("setting", style="cyan")
    table.add_column("value")
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("parsers", ", ".join(available_parsers()))
    table.add_row("text_embed_model (search)", settings.text_embed_model)
    table.add_row("reranker_model", settings.reranker_model)
    table.add_row("use_reranker", str(settings.use_reranker))
    table.add_row("enable_ocr", str(settings.enable_ocr))
    table.add_row("enable_documents", str(settings.enable_documents))
    table.add_row(
        "conversation_windows",
        f"{settings.enable_conversation_windows} "
        f"(size={settings.conversation_window_size}, stride={settings.conversation_window_stride})",
    )
    table.add_row("context_neighbors (ask)", str(settings.context_neighbors))
    table.add_row("enable_hyde (ask)", str(settings.enable_hyde))
    table.add_row("vlm_model (captioning)", settings.vlm_model)
    table.add_row("chat_model (ask/RAG)", settings.chat_model)
    table.add_row("whisper_model", settings.whisper_model)
    table.add_row("llm_base_url", settings.llm_base_url)
    table.add_row("device", settings.device)

    db_path = settings.db_path
    if (db_path / f"{TABLE_NAME}.lance").exists() or db_path.exists():
        try:
            embedder = TextEmbedder(settings)
            store = VectorStore(db_path, embedder.dim)
            table.add_row("indexed chunks", str(store.count()))
        except Exception:
            table.add_row("indexed chunks", "(index not built yet)")
    console.print(table)


if __name__ == "__main__":
    app()
