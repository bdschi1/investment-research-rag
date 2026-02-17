"""Application settings loaded from YAML with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Settings sections
# ---------------------------------------------------------------------------


class EmbeddingSettings(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    dimension: int = 768


class VectorStoreSettings(BaseModel):
    backend: str = "faiss"
    path: str = "local_data/vectorstore"
    collection: str = "investment_docs"


class LLMSettings(BaseModel):
    provider: str = "ollama"
    model: str = "deepseek-r1:32b"
    temperature: float = 0.2
    max_tokens: int = 2048


class ChunkingSettings(BaseModel):
    max_tokens: int = 800
    overlap_tokens: int = 100
    small_file_threshold: int = 3000


class RetrievalSettings(BaseModel):
    top_k: int = 5
    rerank: bool = False
    similarity_threshold: float = 0.0


class IngestionSettings(BaseModel):
    supported_formats: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".xlsx"]
    )
    max_file_size_mb: int = 100


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)


def _find_settings_file() -> Path | None:
    """Walk up from cwd looking for settings.yaml."""
    profile = os.getenv("RAG_PROFILE", "")
    names = [f"settings-{profile}.yaml", "settings.yaml"] if profile else ["settings.yaml"]

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        for name in names:
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def load_settings() -> Settings:
    """Load settings from YAML file, falling back to defaults."""
    path = _find_settings_file()
    if path is None:
        return Settings()

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Settings(**raw)
