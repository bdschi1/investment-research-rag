"""Ingestion pipeline — file → load → sanitize → chunk → embed → store.

This is the main entry point for adding documents to the vector store.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from rag.chunking.factory import get_chunker
from rag.chunking.schemas import ChunkMetadata
from rag.documents.loader import DocumentLoader
from rag.documents.sanitize import sanitize_document_text
from rag.documents.schemas import DocType, DocumentMetadata
from rag.embeddings.base import EmbeddingProvider
from rag.pipeline.schemas import IngestResult
from rag.vectorstore.base import VectorStore
from rag.vectorstore.schemas import VectorRecord

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Orchestrates document ingestion: load → sanitize → chunk → embed → store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        loader: DocumentLoader | None = None,
        batch_size: int = 32,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.loader = loader or DocumentLoader()
        self.batch_size = batch_size

    def ingest_file(
        self,
        path: str | Path,
        doc_type: DocType = DocType.OTHER,
        ticker: str | None = None,
        filing_date: str | None = None,
    ) -> IngestResult:
        """Ingest a single file into the vector store.

        Args:
            path: Path to the document.
            doc_type: Document type for chunker selection.
            ticker: Stock ticker for metadata.
            filing_date: Filing/publication date.

        Returns:
            An ``IngestResult`` with counts and warnings.
        """
        path = Path(path)
        warnings: list[str] = []

        # Step 1: Load
        doc_meta = DocumentMetadata(
            doc_type=doc_type,
            ticker=ticker,
            filing_date=filing_date,
        )
        result = self.loader.load_file(path, metadata=doc_meta)
        warnings.extend(result.warnings)

        if not result.text.strip():
            warnings.append("Document loaded but contains no extractable text")
            return IngestResult(
                source=str(path),
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                warnings=warnings,
            )

        # Step 2: Sanitize
        clean_text = sanitize_document_text(result.text)

        # Step 3: Chunk
        chunk_meta = ChunkMetadata(
            doc_type=doc_type,
            ticker=ticker,
            filing_date=filing_date,
            source_filename=str(path.name),
        )
        chunker = get_chunker(doc_type)
        chunks = chunker.chunk(clean_text, metadata=chunk_meta)

        if not chunks:
            warnings.append("Chunker produced zero chunks")
            return IngestResult(
                source=str(path),
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                warnings=warnings,
            )

        # Step 4: Embed in batches
        texts = [c.text for c in chunks]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.embedding_provider.embed_texts(batch)
            all_embeddings.extend(embeddings)

        # Step 5: Store
        records = []
        for chunk, embedding in zip(chunks, all_embeddings, strict=True):
            records.append(VectorRecord(
                id=str(uuid.uuid4()),
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.metadata,
            ))

        stored = self.vector_store.add(records)

        logger.info(
            "Ingested %s: %d chunks → %d embedded → %d stored",
            path.name,
            len(chunks),
            len(all_embeddings),
            stored,
        )

        return IngestResult(
            source=str(path),
            chunks_created=len(chunks),
            chunks_embedded=len(all_embeddings),
            chunks_stored=stored,
            warnings=warnings,
        )

    def ingest_text(
        self,
        text: str,
        source_name: str = "inline",
        doc_type: DocType = DocType.OTHER,
        ticker: str | None = None,
    ) -> IngestResult:
        """Ingest raw text directly (no file loading step).

        Useful for testing or programmatic ingestion.
        """
        clean_text = sanitize_document_text(text)

        chunk_meta = ChunkMetadata(
            doc_type=doc_type,
            ticker=ticker,
            source_filename=source_name,
        )
        chunker = get_chunker(doc_type)
        chunks = chunker.chunk(clean_text, metadata=chunk_meta)

        if not chunks:
            return IngestResult(
                source=source_name,
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
            )

        texts = [c.text for c in chunks]
        embeddings = self.embedding_provider.embed_texts(texts)

        records = [
            VectorRecord(
                id=str(uuid.uuid4()),
                text=chunk.text,
                embedding=emb,
                metadata=chunk.metadata,
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        stored = self.vector_store.add(records)

        return IngestResult(
            source=source_name,
            chunks_created=len(chunks),
            chunks_embedded=len(embeddings),
            chunks_stored=stored,
        )
