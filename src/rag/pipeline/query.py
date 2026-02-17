"""Query pipeline — question → retrieve → LLM → cited answer.

Adapted from Projects/z_oldProjects/doc-chunker/rag.py with citations.
"""

from __future__ import annotations

import logging

from rag.embeddings.base import EmbeddingProvider
from rag.llm.base import LLMProvider
from rag.pipeline.citations import extract_citations
from rag.pipeline.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from rag.pipeline.schemas import RAGQuery, RAGResponse
from rag.retrieval.retriever import Retriever
from rag.retrieval.schemas import RetrievalConfig
from rag.vectorstore.base import VectorStore
from rag.vectorstore.schemas import MetadataFilter

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Orchestrates question → retrieve → generate → cite."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        system_prompt: str = RAG_SYSTEM_PROMPT,
    ):
        self.retriever = Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt

    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """Run a full RAG query: retrieve → prompt → generate → cite.

        Args:
            rag_query: The query with optional filters.

        Returns:
            A ``RAGResponse`` with answer, citations, and context.
        """
        # Build metadata filter from query
        mf = None
        if rag_query.ticker or rag_query.doc_type:
            mf = MetadataFilter(
                ticker=rag_query.ticker,
                doc_type=rag_query.doc_type,
            )

        # Retrieve relevant context
        retrieval_config = RetrievalConfig(
            top_k=rag_query.top_k,
            rerank=rag_query.rerank,
            metadata_filter=mf,
        )
        retrieval_result = self.retriever.retrieve(
            rag_query.question,
            config=retrieval_config,
        )

        if not retrieval_result.results:
            return RAGResponse(
                question=rag_query.question,
                answer="No relevant documents found for this query.",
                model=getattr(self.llm_provider, "model", "unknown"),
                retrieval_count=0,
            )

        # Build prompt with context
        context_texts = [r.text for r in retrieval_result.results]
        sources = [
            r.metadata.source_filename or r.metadata.ticker or "source"
            for r in retrieval_result.results
        ]

        prompt = build_rag_prompt(
            question=rag_query.question,
            context_texts=context_texts,
            sources=sources,
        )

        # Generate answer
        answer = self.llm_provider.generate(prompt, system=self.system_prompt)

        # Extract citations
        citations = extract_citations(answer, retrieval_result.results)

        logger.info(
            "Query answered: %d context docs, %d citations",
            len(retrieval_result.results),
            len(citations),
        )

        return RAGResponse(
            question=rag_query.question,
            answer=answer,
            citations=citations,
            context_texts=context_texts,
            model=getattr(self.llm_provider, "model", "unknown"),
            retrieval_count=len(retrieval_result.results),
        )

    def query_simple(self, question: str, **kwargs) -> RAGResponse:
        """Convenience method for simple queries.

        Args:
            question: The question string.
            **kwargs: Additional fields for ``RAGQuery``.

        Returns:
            A ``RAGResponse``.
        """
        return self.query(RAGQuery(question=question, **kwargs))
