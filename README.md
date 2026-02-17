# Investment Research RAG

[![CI](https://github.com/bdschi1/investment-research-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/bdschi1/investment-research-rag/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-255%20passed-brightgreen)
![Ruff](https://img.shields.io/badge/lint-ruff%20%E2%9C%93-black)

![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Meta-0467DF?logo=meta&logoColor=white)
![tiktoken](https://img.shields.io/badge/tiktoken-OpenAI-412991?logo=openai&logoColor=white)
![Typer](https://img.shields.io/badge/Typer-CLI-009688)
![Rich](https://img.shields.io/badge/Rich-Console-4B8BBE)

**Retrieval-augmented generation for institutional investment research -- SEC filings, earnings transcripts, equity research reports, and financial models.**

---

## Why This Exists

Institutional research documents have heterogeneous structures: SEC filings split on ITEM section boundaries, earnings calls have speaker turns, equity research mixes narrative with valuation tables, and financial models live in spreadsheets. Generic RAG pipelines treat all text as uniform paragraphs, losing context at section boundaries and discarding structural metadata that matters for downstream reasoning.

This project provides document-type-specific chunking, smart page scoring for large PDFs, and end-to-end citation traceability from retrieved chunk back to source page and section -- the kind of provenance that compliance teams require when an LLM-generated answer informs an investment decision.

All document inputs pass through an 8-pattern prompt injection defense layer before entering the pipeline.

---

## Architecture

```
                         investment-research-rag
 ───────────────────────────────────────────────────────────────────────

  PDF / DOCX / TXT / XLSX
         |
         v
  ┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
  │   Loader     │────>│  Sanitizer   │────>│  Boilerplate      │
  │  (pdfplumber,│     │  (8 injection│     │  Filter           │
  │   docx, etc.)│     │   patterns)  │     │  (disclaimers,    │
  └─────────────┘     └──────────────┘     │   cert strips)    │
                                            └───────┬───────────┘
                                                    |
         ┌──────────────────────────────────────────┘
         v
  ┌─────────────────────────────────────────────────────────┐
  │              Document-Type Chunking                      │
  │                                                          │
  │  SecChunker ─── ITEM boundaries, smart page scoring      │
  │  TranscriptChunker ─── speaker-turn segmentation         │
  │  ResearchChunker ─── paragraph splitting, disclosure rm  │
  │  ExcelChunker ─── per-sheet markdown table extraction    │
  └─────────────────────────┬───────────────────────────────┘
                            |
                            v
  ┌─────────────────────────────────────────────────────────┐
  │              Embedding + Vector Store                     │
  │                                                          │
  │  Ollama / OpenAI / HuggingFace  ───>  FAISS / Qdrant    │
  └─────────────────────────┬───────────────────────────────┘
                            |
                            v
  ┌─────────────────────────────────────────────────────────┐
  │              Retrieval + Reranking                        │
  │                                                          │
  │  Dense search (top-k * 3) ───> Cross-encoder reranker    │
  │  ───> Final top-k with metadata + citation refs          │
  └─────────────────────────┬───────────────────────────────┘
                            |
                            v
  ┌─────────────────────────────────────────────────────────┐
  │              LLM Response Generation                      │
  │                                                          │
  │  Ollama (local) / Anthropic (Claude) / OpenAI (GPT-4)   │
  │  ───> Cited answer with source page + section refs       │
  └─────────────────────────────────────────────────────────┘
```

---

## Feature Highlights

- **4 document-type-specific chunkers** -- not one-size-fits-all. SEC filings split on ITEM boundaries, transcripts on speaker turns, research on paragraphs with disclosure removal, Excel on per-sheet tables.
- **Smart page scoring** for large PDFs -- importance-ranked page selection using heuristics for executive summaries, risk factors, valuation sections, and financial highlights. Budget-constrained so embedding costs stay bounded.
- **8-pattern prompt injection defense** on all document inputs -- detects and redacts injection attempts ("ignore previous instructions", role reassignment, system prompt overrides) before any text enters the pipeline.
- **Dense retrieval with cross-encoder reranking** -- initial retrieval over-fetches at 3x top-k, then a cross-encoder reranker prunes to the final result set for higher precision.
- **3 embedding providers** -- Ollama (local, zero API cost), OpenAI (`text-embedding-3-small/large`), HuggingFace (`sentence-transformers`).
- **2 vector stores** -- FAISS (local, zero infrastructure) and Qdrant (production-grade, metadata filtering).
- **3 LLM providers** -- Ollama (DeepSeek-R1, Llama, Mistral), Anthropic (Claude), OpenAI (GPT-4). Swap via config or environment variable.
- **End-to-end citation traceability** -- every answer chunk maps back to source document, page number, and section header.
- **Built-in evaluation suite** -- retrieval metrics (precision@k, recall@k, MRR, nDCG) and LLM answer grading with scenario-based test data.
- **CLI interface** -- `rag ingest`, `rag query`, `rag eval` via Typer with Rich console output.

---

## Document Types

| Type | Chunker | Strategy |
|---|---|---|
| SEC Filing (10-K, 10-Q) | `SecChunker` | ITEM section boundaries, smart page scoring |
| Earnings Transcript | `TranscriptChunker` | Speaker-turn segmentation |
| Equity Research | `ResearchChunker` | Token-aware paragraph splitting, disclosure removal |
| Financial Model (Excel) | `ExcelChunker` | Per-sheet extraction with markdown tables |

---

## Provider Matrix

### Embedding Providers

| Provider | Models | Notes |
|---|---|---|
| **Ollama** (default) | `nomic-embed-text` | Local, zero API cost |
| **OpenAI** | `text-embedding-3-small`, `text-embedding-3-large` | Requires API key |
| **HuggingFace** | Any `sentence-transformers` model | Local, GPU optional |

### Vector Stores

| Store | Notes |
|---|---|
| **FAISS** (default) | Local file-based, zero infrastructure |
| **Qdrant** | Production-grade, supports metadata filtering |

### LLM Providers

| Provider | Models | Notes |
|---|---|---|
| **Ollama** (default) | DeepSeek-R1, Llama, Mistral | Local, no API key |
| **Anthropic** | Claude | Requires API key |
| **OpenAI** | GPT-4 | Requires API key |

---

## Quick Start

```bash
# Clone
git clone https://github.com/bdschi1/investment-research-rag.git
cd investment-research-rag

# Install with local stack (no API keys needed)
pip install -e ".[local]"

# Ingest a document
rag ingest filing.pdf --ticker AAPL --doc-type sec_filing

# Query with citations
rag query "What were the key risk factors?"

# Run evaluation
rag eval --scenario-dir eval_data/scenarios
```

---

## Installation

```bash
# Core only (no embedding/vector/LLM providers)
pip install -e .

# Local stack -- Ollama + FAISS + sentence-transformers
pip install -e ".[local]"

# Development -- adds pytest, ruff, coverage
pip install -e ".[dev]"

# Full stack -- every optional dependency
pip install -e ".[all]"
```

### Optional Dependency Groups

| Group | Packages | Purpose |
|---|---|---|
| `faiss` | faiss-cpu | Local vector store |
| `qdrant` | qdrant-client | Production vector store |
| `openai` | openai | OpenAI embeddings + LLM |
| `anthropic` | anthropic | Claude LLM |
| `ollama` | ollama | Local LLM + embeddings |
| `huggingface` | sentence-transformers, torch | Local embeddings |
| `reranker` | sentence-transformers | Cross-encoder reranking |
| `sec` | pymupdf | Enhanced SEC filing parsing |
| `excel` | openpyxl | Excel/XLSX model ingestion |
| `local` | faiss-cpu, sentence-transformers, ollama | Zero-API-key stack |
| `dev` | pytest, pytest-cov, ruff, fpdf2 | Testing and linting |
| `all` | Everything above | Full install |

---

## Configuration

Default settings live in `settings.yaml`. Override per-project or via environment variables:

```bash
# Switch embedding provider
RAG_EMBEDDING__PROVIDER=openai
RAG_EMBEDDING__MODEL=text-embedding-3-small

# Switch vector store
RAG_VECTORSTORE__PROVIDER=qdrant

# Switch LLM
RAG_LLM__PROVIDER=anthropic
RAG_LLM__MODEL=claude-sonnet-4-20250514
```

Configuration is managed via `pydantic-settings` -- any nested key can be set as an environment variable using double-underscore (`__`) separators.

---

## Project Structure

```
investment-research-rag/
│
├── src/rag/
│   ├── config.py               # Pydantic-settings configuration
│   │
│   ├── documents/              # Load, clean, parse (PDF, DOCX, TXT, XLSX)
│   │   ├── loader.py           # Multi-format document loader
│   │   ├── sanitize.py         # 8-pattern prompt injection defense
│   │   ├── boilerplate.py      # Disclaimer/certification stripping
│   │   ├── sec_parser.py       # SEC ITEM section boundary detection
│   │   ├── transcript_parser.py# Speaker-turn parsing for earnings calls
│   │   └── schemas.py          # Document data models
│   │
│   ├── chunking/               # Split into retrievable pieces
│   │   ├── base.py             # Abstract chunker interface
│   │   ├── sec_chunker.py      # SEC filing chunker (ITEM-aware)
│   │   ├── transcript_chunker.py # Earnings call chunker (speaker turns)
│   │   ├── research_chunker.py # Equity research chunker
│   │   ├── excel_chunker.py    # Financial model chunker (per-sheet)
│   │   ├── scoring.py          # Smart page importance scoring
│   │   ├── factory.py          # Chunker factory by document type
│   │   └── schemas.py          # Chunk data models
│   │
│   ├── embeddings/             # Vector representations
│   │   ├── base.py             # Abstract embedding provider
│   │   ├── ollama_provider.py  # Ollama (nomic-embed-text)
│   │   ├── openai_provider.py  # OpenAI (text-embedding-3-*)
│   │   ├── huggingface_provider.py # HuggingFace sentence-transformers
│   │   └── factory.py          # Provider factory
│   │
│   ├── vectorstore/            # Storage + similarity search
│   │   ├── base.py             # Abstract vector store
│   │   ├── faiss_store.py      # FAISS (local, file-based)
│   │   ├── qdrant_store.py     # Qdrant (production-grade)
│   │   ├── factory.py          # Store factory
│   │   └── schemas.py          # Search result models
│   │
│   ├── retrieval/              # Search orchestration
│   │   ├── retriever.py        # Embed -> search -> rerank pipeline
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── schemas.py          # Retrieval config + result models
│   │
│   ├── llm/                    # Response generation
│   │   ├── base.py             # Abstract LLM provider
│   │   ├── ollama_provider.py  # Local models (DeepSeek, Llama, Mistral)
│   │   ├── anthropic_provider.py # Claude
│   │   ├── openai_provider.py  # GPT-4
│   │   └── factory.py          # Provider factory
│   │
│   ├── pipeline/               # End-to-end orchestration
│   │   ├── ingest.py           # Document -> chunks -> vectors
│   │   ├── query.py            # Question -> retrieval -> LLM -> answer
│   │   ├── citations.py        # Citation extraction and formatting
│   │   ├── prompts.py          # System/user prompt templates
│   │   └── schemas.py          # Pipeline result models
│   │
│   └── evaluation/             # Quality measurement
│       ├── retrieval_metrics.py # precision@k, recall@k, MRR, nDCG
│       ├── answer_grader.py    # LLM-based answer quality grading
│       ├── runner.py           # Scenario-based eval runner
│       └── schemas.py          # Eval data models
│
├── cli/
│   └── main.py                 # Typer CLI: rag ingest | query | eval
│
├── tests/                      # 255 tests across 11 test modules
│   ├── test_boilerplate.py     # Boilerplate filter tests (21)
│   ├── test_chunkers.py        # All 4 chunkers (42)
│   ├── test_embeddings.py      # Embedding providers (15)
│   ├── test_evaluation.py      # Eval metrics + grading (30)
│   ├── test_loader.py          # Document loading (21)
│   ├── test_pipeline.py        # Ingest + query pipeline (23)
│   ├── test_retrieval.py       # Retriever + reranker (13)
│   ├── test_sanitize.py        # Injection defense (18)
│   ├── test_scoring.py         # Page importance scoring (24)
│   ├── test_sec_parser.py      # SEC ITEM parsing (14)
│   └── test_vectorstore.py     # FAISS + Qdrant stores (34)
│
├── settings.yaml               # Default configuration
├── pyproject.toml              # Build config, deps, tool settings
└── LICENSE                     # MIT
```

---

## Testing

255 tests across 11 modules. Zero ruff lint errors.

```bash
# Run full suite
pip install -e ".[dev]"
pytest tests/ -v

# With coverage
pytest tests/ --cov=rag --cov-report=term-missing

# Lint
ruff check src/ tests/ cli/
```

---

## License

MIT
