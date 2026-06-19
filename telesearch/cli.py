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
        help="Path to the Telegram export folder (containing result.json) or the result.json file.",
    ),
    images: bool = typer.Option(True, help="Caption photos with the VLM."),
    videos: bool = typer.Option(True, help="Summarize and transcribe videos."),
    audio: bool = typer.Option(True, help="Transcribe voice messages."),
    ocr: bool = typer.Option(True, help="Extract verbatim on-image text (OCR) as a separate chunk."),
):
    """Parse an export and build the searchable index."""
    from .ingest import parse_export
    from .index.build import build_index

    settings = get_settings()
    export_root = export if export.is_dir() else export.parent

    console.print(f"[bold]Parsing[/bold] {export}")
    messages = list(parse_export(export))
    console.print(f"Found [cyan]{len(messages)}[/cyan] messages. Building index...")

    count = build_index(
        messages,
        export_root,
        settings,
        do_images=images,
        do_videos=videos,
        do_audio=audio,
        do_ocr=ocr,
    )
    console.print(
        f"[bold green]Done.[/bold green] Indexed {count} chunks into {settings.db_path}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(10, help="Number of results."),
    modality: Optional[str] = typer.Option(
        None, help="Filter by type: text, image, video, audio, ocr."
    ),
    rerank: Optional[bool] = typer.Option(
        None, "--rerank/--no-rerank", help="Cross-encoder rerank (default: config)."
    ),
):
    """Hybrid (semantic + keyword) search, then cross-encoder rerank."""
    from .search import Retriever

    settings = get_settings()
    retriever = Retriever(settings)
    results = retriever.search(query, k=k, modality=modality, rerank=rerank)

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
):
    """Ask a question and get an answer grounded in the conversation (RAG)."""
    from .search import answer_question

    settings = get_settings()
    answer, sources = answer_question(question, settings, k=k)

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

    settings = get_settings()
    table = Table(title="telesearch configuration")
    table.add_column("setting", style="cyan")
    table.add_column("value")
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("text_embed_model (search)", settings.text_embed_model)
    table.add_row("reranker_model", settings.reranker_model)
    table.add_row("use_reranker", str(settings.use_reranker))
    table.add_row("enable_ocr", str(settings.enable_ocr))
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
