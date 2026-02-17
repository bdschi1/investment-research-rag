"""CLI entry point — Typer app for rag commands.

Usage:
    rag ingest filing.pdf --ticker AAPL --doc-type sec_filing
    rag query "What were the key risk factors?"
    rag eval --scenario-dir eval_data/scenarios
    rag status
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rag",
    help="Investment Research RAG — ingest, query, evaluate.",
    no_args_is_help=True,
)

console = Console()

_INGEST_PATH = typer.Argument(..., help="Path to the document to ingest")
_EVAL_PATH = typer.Argument(..., help="Path to scenario YAML files")


@app.command()
def ingest(
    path: Annotated[Path, _INGEST_PATH],
    ticker: str | None = typer.Option(
        None, "--ticker", "-t", help="Stock ticker",
    ),
    doc_type: str = typer.Option(
        "other", "--doc-type", "-d",
        help="Document type (sec_filing, earnings_transcript, "
             "research_report, financial_model)",
    ),
    embedding_provider: str = typer.Option(
        "ollama", "--embedding", "-e", help="Embedding provider",
    ),
    vector_store: str = typer.Option(
        "faiss", "--store", "-s", help="Vector store backend",
    ),
) -> None:
    """Ingest a document into the vector store."""
    from rag.documents.schemas import DocType
    from rag.embeddings.factory import get_embedding_provider
    from rag.pipeline.ingest import IngestPipeline
    from rag.vectorstore.factory import get_vector_store

    dt = DocType(doc_type)
    emb = get_embedding_provider(embedding_provider)
    store = get_vector_store(vector_store, dimension=emb.dimension)

    pipeline = IngestPipeline(embedding_provider=emb, vector_store=store)
    result = pipeline.ingest_file(path, doc_type=dt, ticker=ticker)

    console.print(f"\n[bold green]Ingested:[/] {path.name}")
    console.print(f"  Chunks: {result.chunks_created}")
    console.print(f"  Embedded: {result.chunks_embedded}")
    console.print(f"  Stored: {result.chunks_stored}")

    if result.warnings:
        for w in result.warnings:
            console.print(f"  [yellow]Warning:[/] {w}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    ticker: str | None = typer.Option(
        None, "--ticker", "-t", help="Filter by ticker",
    ),
    top_k: int = typer.Option(
        10, "--top-k", "-k", help="Number of context docs",
    ),
    embedding_provider: str = typer.Option(
        "ollama", "--embedding", "-e", help="Embedding provider",
    ),
    vector_store: str = typer.Option(
        "faiss", "--store", "-s", help="Vector store backend",
    ),
    llm_provider: str = typer.Option(
        "ollama", "--llm", "-l", help="LLM provider",
    ),
) -> None:
    """Query the RAG system with a question."""
    from rag.embeddings.factory import get_embedding_provider
    from rag.llm.factory import get_llm_provider
    from rag.pipeline.citations import format_citations
    from rag.pipeline.query import QueryPipeline
    from rag.pipeline.schemas import RAGQuery
    from rag.vectorstore.factory import get_vector_store

    emb = get_embedding_provider(embedding_provider)
    store = get_vector_store(vector_store, dimension=emb.dimension)
    llm = get_llm_provider(llm_provider)

    pipeline = QueryPipeline(
        embedding_provider=emb,
        vector_store=store,
        llm_provider=llm,
    )

    rag_query = RAGQuery(question=question, ticker=ticker, top_k=top_k)
    response = pipeline.query(rag_query)

    console.print(f"\n[bold]Q:[/] {response.question}")
    console.print(f"\n[bold green]A:[/] {response.answer}")

    if response.citations:
        console.print(format_citations(response.citations))

    console.print(
        f"\n[dim]Model: {response.model} "
        f"| Context docs: {response.retrieval_count}[/]",
    )


@app.command()
def eval(
    scenario_dir: Annotated[Path, _EVAL_PATH],
) -> None:
    """Run evaluation scenarios and report metrics."""
    from rag.evaluation.runner import EvalRunner

    runner = EvalRunner()
    scenarios = runner.load_scenarios(scenario_dir)

    console.print(f"\nLoaded [bold]{len(scenarios)}[/] evaluation scenarios")
    console.print(
        "[yellow]Note: Full eval requires a running pipeline. "
        "Showing scenario summary.[/]\n",
    )

    table = Table(title="Evaluation Scenarios")
    table.add_column("ID", style="cyan")
    table.add_column("Question")
    table.add_column("Ticker")
    table.add_column("Tags")

    for s in scenarios:
        table.add_row(
            s.id, s.question[:60], s.ticker or "", ", ".join(s.tags),
        )

    console.print(table)


@app.command()
def status() -> None:
    """Show system status (installed providers, config, vector store)."""
    from rag.chunking.factory import available_chunkers
    from rag.embeddings.factory import available_providers as emb_providers
    from rag.llm.factory import available_providers as llm_providers
    from rag.vectorstore.factory import available_stores

    console.print("\n[bold green]investment-research-rag[/] v0.1.0\n")

    table = Table(title="Available Components")
    table.add_column("Layer", style="cyan")
    table.add_column("Available")

    table.add_row("Chunkers", ", ".join(available_chunkers()))
    table.add_row("Embedding Providers", ", ".join(emb_providers()))
    table.add_row("Vector Stores", ", ".join(available_stores()))
    table.add_row("LLM Providers", ", ".join(llm_providers()))

    console.print(table)


if __name__ == "__main__":
    app()
