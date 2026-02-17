"""Tests for Lambda handlers — ingest and query.

The ``lambda/`` directory uses a Python reserved word, so we import modules
via importlib and patch attributes directly on the loaded module objects.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from rag.pipeline.schemas import Citation, IngestResult, RAGResponse


# ---------------------------------------------------------------------------
# Module loading helpers (``lambda`` is a reserved keyword)
# ---------------------------------------------------------------------------

def _load_ingest_handler() -> ModuleType:
    """Import lambda/ingest_handler.py via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    lambda_dir = repo_root / "lambda"
    if str(lambda_dir) not in sys.path:
        sys.path.insert(0, str(lambda_dir))
    spec = importlib.util.spec_from_file_location(
        "ingest_handler", lambda_dir / "ingest_handler.py"
    )
    mod = importlib.util.module_from_spec(spec)
    return mod, spec


def _load_query_handler() -> ModuleType:
    """Import lambda/query_handler.py via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    lambda_dir = repo_root / "lambda"
    if str(lambda_dir) not in sys.path:
        sys.path.insert(0, str(lambda_dir))
    spec = importlib.util.spec_from_file_location(
        "query_handler", lambda_dir / "query_handler.py"
    )
    mod = importlib.util.module_from_spec(spec)
    return mod, spec


def _mock_settings() -> MagicMock:
    """Create a standard mock settings object."""
    settings = MagicMock()
    settings.embedding.provider = "ollama"
    settings.embedding.model = "nomic-embed-text"
    settings.embedding.dimension = 768
    settings.vectorstore.backend = "faiss"
    settings.vectorstore.opensearch_endpoint = None
    settings.vectorstore.collection = "investment_docs"
    settings.vectorstore.opensearch_region = "us-east-1"
    settings.llm.provider = "ollama"
    settings.llm.model = "deepseek-r1:32b"
    return settings


# ---------------------------------------------------------------------------
# Ingest Handler Tests
# ---------------------------------------------------------------------------


class TestIngestHandler:
    """Tests for lambda/ingest_handler.py."""

    def _make_s3_sqs_event(
        self,
        bucket: str = "research-docs",
        key: str = "AAPL/sec_filing/10k_2024.pdf",
    ) -> dict:
        """Build a minimal SQS event wrapping an S3 notification."""
        return {
            "Records": [
                {
                    "body": json.dumps({
                        "Records": [
                            {
                                "s3": {
                                    "bucket": {"name": bucket},
                                    "object": {"key": key},
                                }
                            }
                        ]
                    })
                }
            ]
        }

    @patch("rag.vectorstore.factory.get_vector_store")
    @patch("rag.embeddings.factory.get_embedding_provider")
    @patch("rag.config.load_settings")
    def test_processes_s3_event(
        self, mock_load_settings, mock_emb_factory, mock_store_factory
    ):
        mock_load_settings.return_value = _mock_settings()
        mock_emb_factory.return_value = MagicMock()
        mock_store_factory.return_value = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = IngestResult(
            source="10k_2024.pdf",
            chunks_created=15,
            chunks_embedded=15,
            chunks_stored=15,
        )

        mock_boto3 = MagicMock()
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            with patch("rag.pipeline.ingest.IngestPipeline", return_value=mock_pipeline):
                mod, spec = _load_ingest_handler()
                spec.loader.exec_module(mod)

                event = self._make_s3_sqs_event()
                result = mod.handler(event, None)

        assert result["statusCode"] == 200
        assert len(result["results"]) == 1
        assert result["results"][0]["chunks_created"] == 15
        assert result["results"][0]["source"] == "AAPL/sec_filing/10k_2024.pdf"

    @patch("rag.vectorstore.factory.get_vector_store")
    @patch("rag.embeddings.factory.get_embedding_provider")
    @patch("rag.config.load_settings")
    def test_extracts_ticker_from_key(
        self, mock_load_settings, mock_emb_factory, mock_store_factory
    ):
        mock_load_settings.return_value = _mock_settings()
        mock_emb_factory.return_value = MagicMock()
        mock_store_factory.return_value = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = IngestResult(
            source="10k.pdf",
            chunks_created=5,
            chunks_embedded=5,
            chunks_stored=5,
        )

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            with patch("rag.pipeline.ingest.IngestPipeline", return_value=mock_pipeline):
                mod, spec = _load_ingest_handler()
                spec.loader.exec_module(mod)

                event = self._make_s3_sqs_event(key="MSFT/sec_filing/10k.pdf")
                mod.handler(event, None)

                # Verify ingest_file was called with ticker="MSFT"
                call_kwargs = mock_pipeline.ingest_file.call_args
                assert call_kwargs[1]["ticker"] == "MSFT"

    @patch("rag.vectorstore.factory.get_vector_store")
    @patch("rag.embeddings.factory.get_embedding_provider")
    @patch("rag.config.load_settings")
    def test_empty_event(
        self, mock_load_settings, mock_emb_factory, mock_store_factory
    ):
        mock_load_settings.return_value = _mock_settings()
        mock_emb_factory.return_value = MagicMock()
        mock_store_factory.return_value = MagicMock()

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            with patch("rag.pipeline.ingest.IngestPipeline"):
                mod, spec = _load_ingest_handler()
                spec.loader.exec_module(mod)

                result = mod.handler({"Records": []}, None)

        assert result["statusCode"] == 200
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Query Handler Tests
# ---------------------------------------------------------------------------


class TestQueryHandler:
    """Tests for lambda/query_handler.py."""

    def _make_apigw_event(self, body: dict | None = None) -> dict:
        """Build a minimal API Gateway event."""
        return {
            "body": json.dumps(body) if body else None,
            "httpMethod": "POST",
            "path": "/query",
        }

    def _load_and_patch_query_module(self, mock_pipeline=None):
        """Load query_handler with all dependencies mocked."""
        mod, spec = _load_query_handler()

        with patch("rag.config.load_settings", return_value=_mock_settings()):
            with patch("rag.embeddings.factory.get_embedding_provider"):
                with patch("rag.vectorstore.factory.get_vector_store"):
                    with patch("rag.llm.factory.get_llm_provider"):
                        spec.loader.exec_module(mod)

        if mock_pipeline is not None:
            mod._pipeline = mock_pipeline

        return mod

    def test_valid_request(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = RAGResponse(
            question="What is AAPL revenue?",
            answer="Apple reported $394B in revenue.",
            citations=[
                Citation(
                    index=0,
                    text="Revenue was $394B",
                    source="10k_2024.pdf",
                    section="Income Statement",
                    ticker="AAPL",
                )
            ],
            model="deepseek-r1:32b",
            retrieval_count=5,
        )

        mod = self._load_and_patch_query_module(mock_pipeline)
        event = self._make_apigw_event({"question": "What is AAPL revenue?"})
        result = mod.handler(event, None)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["answer"] == "Apple reported $394B in revenue."
        assert len(body["citations"]) == 1
        assert body["citations"][0]["ticker"] == "AAPL"

    def test_missing_question(self):
        mod = self._load_and_patch_query_module()
        event = self._make_apigw_event({"ticker": "AAPL"})
        result = mod.handler(event, None)

        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert "error" in body

    def test_invalid_json(self):
        mod = self._load_and_patch_query_module()
        event = {"body": "not-valid-json{{{", "httpMethod": "POST"}
        result = mod.handler(event, None)

        assert result["statusCode"] == 400

    def test_warm_start_reuse(self):
        """_get_pipeline should return the cached pipeline on second call."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = RAGResponse(
            question="test",
            answer="answer",
            model="test-model",
            retrieval_count=0,
        )

        mod = self._load_and_patch_query_module(mock_pipeline)
        event = self._make_apigw_event({"question": "test"})

        mod.handler(event, None)
        mod.handler(event, None)

        # Pipeline.query should be called twice (warm-start reuse)
        assert mock_pipeline.query.call_count == 2

    def test_with_filters(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = RAGResponse(
            question="AAPL risk factors",
            answer="Key risks include...",
            model="deepseek-r1:32b",
            retrieval_count=3,
        )

        mod = self._load_and_patch_query_module(mock_pipeline)
        event = self._make_apigw_event({
            "question": "What are the risk factors?",
            "ticker": "AAPL",
            "doc_type": "sec_filing",
            "top_k": 5,
        })
        result = mod.handler(event, None)

        assert result["statusCode"] == 200
        # Verify the pipeline was called with a RAGQuery containing filters
        call_args = mock_pipeline.query.call_args[0][0]
        assert call_args.ticker == "AAPL"
        assert call_args.doc_type == "sec_filing"
        assert call_args.top_k == 5
