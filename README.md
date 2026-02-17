# investment-research-rag

Retrieval-augmented generation for institutional investment research — SEC filings, earnings transcripts, equity research reports, and financial models.

## Architecture

```
document → load → sanitize → chunk → embed → store → retrieve → rerank → LLM → cited answer
```

### Document Types

| Type | Chunker | Strategy |
|---|---|---|
| SEC Filing (10-K, 10-Q) | `SecChunker` | ITEM section boundaries, smart page scoring |
| Earnings Transcript | `TranscriptChunker` | Speaker-turn segmentation |
| Equity Research | `ResearchChunker` | Token-aware paragraph splitting, disclosure removal |
| Financial Model (Excel) | `ExcelChunker` | Per-sheet extraction with markdown tables |

### Embedding Providers

- **Ollama** (default, local) — `nomic-embed-text`
- **OpenAI** — `text-embedding-3-small`, `text-embedding-3-large`
- **HuggingFace** — any `sentence-transformers` model

### Vector Stores

- **FAISS** (default, local) — zero infrastructure
- **Qdrant** — production-grade, metadata filtering

### LLM Providers

- **Ollama** (default, local) — DeepSeek-R1, Llama, Mistral
- **Anthropic** — Claude
- **OpenAI** — GPT-4

## Quick Start

```bash
# Install with local stack (no API keys needed)
pip install -e ".[local]"

# Ingest a document
rag ingest filing.pdf --ticker AAPL --doc-type sec_filing

# Query with citations
rag query "What were the key risk factors?"

# Run evaluation
rag eval --scenario-dir eval_data/scenarios
```

## Installation

```bash
# Core only
pip install -e .

# Development
pip install -e ".[dev]"

# Full stack
pip install -e ".[all]"
```

### Optional dependency groups

| Group | Packages |
|---|---|
| `faiss` | faiss-cpu |
| `qdrant` | qdrant-client |
| `openai` | openai |
| `anthropic` | anthropic |
| `huggingface` | sentence-transformers, torch |
| `local` | faiss-cpu, sentence-transformers, ollama |
| `dev` | pytest, pytest-cov, ruff, fpdf2 |
| `all` | everything above |

## Configuration

Default settings in `settings.yaml`. Override per-project or via environment variables:

```bash
RAG_EMBEDDING__PROVIDER=openai
RAG_EMBEDDING__MODEL=text-embedding-3-small
RAG_VECTORSTORE__PROVIDER=qdrant
```

## Project Structure

```
src/rag/
├── documents/      # Load, clean, parse (PDF, DOCX, TXT, XLSX)
├── chunking/       # Split into retrievable pieces (section-aware)
├── embeddings/     # Vector representations (Ollama, OpenAI, HF)
├── vectorstore/    # Storage + search (FAISS, Qdrant)
├── retrieval/      # Similarity search + reranking
├── llm/            # Response generation (Claude, GPT-4, local)
├── pipeline/       # End-to-end orchestration with citations
└── evaluation/     # Retrieval metrics + answer grading
```

## License

MIT
