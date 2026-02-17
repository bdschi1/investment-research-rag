"""Tests for all chunker implementations."""

from __future__ import annotations

import pytest

from rag.chunking.base import BaseChunker
from rag.chunking.excel_chunker import ExcelChunker
from rag.chunking.factory import available_chunkers, clear_cache, get_chunker
from rag.chunking.research_chunker import ResearchChunker, _count_tokens
from rag.chunking.schemas import Chunk, ChunkMetadata
from rag.chunking.sec_chunker import SecChunker
from rag.chunking.transcript_chunker import TranscriptChunker
from rag.documents.schemas import DocType

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_count_tokens_basic(self):
        tokens = _count_tokens("Hello world this is a test")
        assert tokens > 0
        assert tokens < 20

    def test_count_tokens_empty(self):
        assert _count_tokens("") == 0

    def test_count_tokens_financial_text(self):
        text = "Revenue was $94.9 billion, up 6% year-over-year."
        tokens = _count_tokens(text)
        assert 8 < tokens < 25

    def test_count_tokens_long_text(self):
        text = "word " * 1000
        tokens = _count_tokens(text)
        assert tokens > 500  # Should be roughly 1000 tokens


# ---------------------------------------------------------------------------
# ResearchChunker
# ---------------------------------------------------------------------------


class TestResearchChunker:
    @pytest.fixture
    def chunker(self) -> ResearchChunker:
        return ResearchChunker(max_tokens=100)

    def test_basic_chunking(self, chunker: ResearchChunker, sample_txt_content: str):
        chunks = chunker.chunk(sample_txt_content)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunks_have_metadata(self, chunker: ResearchChunker, sample_txt_content: str):
        meta = ChunkMetadata(ticker="AAPL", doc_type=DocType.RESEARCH_REPORT)
        chunks = chunker.chunk(sample_txt_content, metadata=meta)
        for c in chunks:
            assert c.metadata.ticker == "AAPL"
            assert c.metadata.doc_type == DocType.RESEARCH_REPORT

    def test_chunk_indices(self, chunker: ResearchChunker, sample_txt_content: str):
        chunks = chunker.chunk(sample_txt_content)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)

    def test_token_counts(self, chunker: ResearchChunker, sample_txt_content: str):
        chunks = chunker.chunk(sample_txt_content)
        for c in chunks:
            assert c.token_count > 0

    def test_disclosure_truncation(self, chunker: ResearchChunker):
        text = (
            "Revenue grew 10%.\n\n"
            "EPS was $6.97.\n\n"
            "Important Disclosures\n\n"
            "This report was prepared by our research team.\n\n"
            "Analyst holds no position."
        )
        chunks = chunker.chunk(text)
        all_text = " ".join(c.text for c in chunks)
        assert "Revenue grew" in all_text
        assert "EPS was" in all_text
        assert "Analyst holds" not in all_text

    def test_section_break_forces_new_chunk(self, chunker: ResearchChunker):
        text = (
            "First paragraph of analysis.\n\n"
            "Exhibit 1 Financial Summary\n\n"
            "Data follows here."
        )
        chunks = chunker.chunk(text)
        # "Exhibit 1" should force a break, so we get at least 2 chunks
        assert len(chunks) >= 2

    def test_respects_max_tokens(self):
        chunker = ResearchChunker(max_tokens=50)
        # Long text should be split into multiple small chunks
        text = ("This is a paragraph of financial analysis text. " * 20 + "\n\n") * 5
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_strategy_name(self):
        assert ResearchChunker.strategy_name() == "ResearchChunker"

    def test_empty_text(self, chunker: ResearchChunker):
        chunks = chunker.chunk("")
        assert chunks == []

    def test_is_base_chunker(self):
        assert issubclass(ResearchChunker, BaseChunker)


# ---------------------------------------------------------------------------
# SecChunker
# ---------------------------------------------------------------------------


class TestSecChunker:
    @pytest.fixture
    def chunker(self) -> SecChunker:
        return SecChunker(max_tokens=200)

    def test_section_aware_chunking(self, chunker: SecChunker, sec_filing_text: str):
        chunks = chunker.chunk(sec_filing_text)
        assert len(chunks) > 0
        # Should detect at least some of the ITEM sections
        section_names = [c.metadata.section_name for c in chunks if c.metadata.section_name]
        assert len(section_names) > 0

    def test_item_metadata(self, chunker: SecChunker, sec_filing_text: str):
        meta = ChunkMetadata(doc_type=DocType.SEC_FILING, ticker="AAPL")
        chunks = chunker.chunk(sec_filing_text, metadata=meta)
        for c in chunks:
            assert c.metadata.doc_type == DocType.SEC_FILING
            assert c.metadata.ticker == "AAPL"

    def test_fallback_to_paragraph_chunking(self, chunker: SecChunker):
        """Text without ITEM markers should fall back to paragraph chunking."""
        text = "Revenue grew 10%.\n\nEPS was $6.97.\n\nMargins expanded."
        chunks = chunker.chunk(text)
        assert len(chunks) > 0
        # No section names since no ITEM sections detected
        for c in chunks:
            assert c.metadata.section_name is None

    def test_sub_chunks_large_sections(self):
        """Large sections should be sub-chunked."""
        chunker = SecChunker(max_tokens=50)
        text = (
            "Item 1. Business\n\n"
            + ("Apple designs and sells consumer electronics. " * 50)
        )
        chunks = chunker.chunk(text)
        # Should produce multiple chunks for the oversized section
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self, chunker: SecChunker, sec_filing_text: str):
        chunks = chunker.chunk(sec_filing_text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_is_base_chunker(self):
        assert issubclass(SecChunker, BaseChunker)


# ---------------------------------------------------------------------------
# TranscriptChunker
# ---------------------------------------------------------------------------


class TestTranscriptChunker:
    @pytest.fixture
    def chunker(self) -> TranscriptChunker:
        return TranscriptChunker(max_tokens=200)

    def test_speaker_turn_chunking(self, chunker: TranscriptChunker, earnings_transcript_text: str):
        chunks = chunker.chunk(earnings_transcript_text)
        assert len(chunks) > 0

    def test_speaker_metadata(self, chunker: TranscriptChunker, earnings_transcript_text: str):
        chunks = chunker.chunk(earnings_transcript_text)
        speakers = [c.metadata.speaker for c in chunks if c.metadata.speaker]
        assert len(speakers) > 0

    def test_qa_detection(self, chunker: TranscriptChunker, earnings_transcript_text: str):
        chunks = chunker.chunk(earnings_transcript_text)
        section_types = [c.metadata.section_name for c in chunks if c.metadata.section_name]
        # Should have both prepared_remarks and qa sections
        assert any("qa" in s for s in section_types) or any("prepared" in s for s in section_types)

    def test_fallback_for_unstructured(self, chunker: TranscriptChunker):
        text = "This is just plain text without any speaker attribution."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_turn_splitting(self):
        chunker = TranscriptChunker(max_tokens=30)
        text = (
            "Tim Cook -- Chief Executive Officer\n\n"
            "This is sentence one. This is sentence two. This is sentence three. "
            "This is sentence four. This is sentence five. This is sentence six. "
            "This is sentence seven. This is sentence eight. This is sentence nine. "
            "This is sentence ten. This is sentence eleven. This is sentence twelve."
        )
        chunks = chunker.chunk(text)
        # Long turn should be split into multiple chunks
        assert len(chunks) > 1

    def test_chunk_metadata_propagation(
        self, chunker: TranscriptChunker, earnings_transcript_text: str,
    ):
        meta = ChunkMetadata(
            doc_type=DocType.EARNINGS_TRANSCRIPT,
            ticker="AAPL",
            filing_date="2024-10-30",
        )
        chunks = chunker.chunk(earnings_transcript_text, metadata=meta)
        for c in chunks:
            assert c.metadata.doc_type == DocType.EARNINGS_TRANSCRIPT
            assert c.metadata.ticker == "AAPL"

    def test_is_base_chunker(self):
        assert issubclass(TranscriptChunker, BaseChunker)


# ---------------------------------------------------------------------------
# ExcelChunker
# ---------------------------------------------------------------------------


class TestExcelChunker:
    @pytest.fixture
    def chunker(self) -> ExcelChunker:
        return ExcelChunker()

    def test_text_based_chunking(self, chunker: ExcelChunker):
        text = (
            "## Sheet: Summary (10 rows, 4 cols)\n\n"
            "| Year | Revenue |\n|---|---|\n| 2024 | $395B |\n\n"
            "---\n\n"
            "## Sheet: Segments (5 rows, 3 cols)\n\n"
            "| Segment | Revenue |\n|---|---|\n| iPhone | $200B |"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_sheet_metadata(self, chunker: ExcelChunker):
        text = "Sheet 1 content\n\n---\n\nSheet 2 content"
        meta = ChunkMetadata(doc_type=DocType.FINANCIAL_MODEL, ticker="AAPL")
        chunks = chunker.chunk(text, metadata=meta)
        for c in chunks:
            assert c.metadata.doc_type == DocType.FINANCIAL_MODEL
            assert c.metadata.ticker == "AAPL"
            assert c.metadata.section_name is not None

    def test_file_based_chunking(self, chunker: ExcelChunker, sample_xlsx_file):
        chunks = chunker.chunk_file(sample_xlsx_file)
        assert len(chunks) == 2  # Two sheets
        # Verify sheet names
        sheet_names = [c.metadata.section_name for c in chunks]
        assert "Summary" in sheet_names
        assert "Segments" in sheet_names

    def test_file_based_metadata(self, chunker: ExcelChunker, sample_xlsx_file):
        meta = ChunkMetadata(doc_type=DocType.FINANCIAL_MODEL, ticker="MSFT")
        chunks = chunker.chunk_file(sample_xlsx_file, metadata=meta)
        for c in chunks:
            assert c.metadata.ticker == "MSFT"

    def test_empty_sheet_separator(self, chunker: ExcelChunker):
        text = "Content\n\n---\n\n\n\n---\n\nMore content"
        chunks = chunker.chunk(text)
        # Empty sheets should be skipped
        assert all(c.text.strip() for c in chunks)

    def test_is_base_chunker(self):
        assert issubclass(ExcelChunker, BaseChunker)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestChunkerFactory:
    def setup_method(self):
        clear_cache()

    def test_get_research_chunker(self):
        chunker = get_chunker(DocType.RESEARCH_REPORT)
        assert isinstance(chunker, ResearchChunker)

    def test_get_sec_chunker(self):
        chunker = get_chunker(DocType.SEC_FILING)
        assert isinstance(chunker, SecChunker)

    def test_get_transcript_chunker(self):
        chunker = get_chunker(DocType.EARNINGS_TRANSCRIPT)
        assert isinstance(chunker, TranscriptChunker)

    def test_get_excel_chunker(self):
        chunker = get_chunker(DocType.FINANCIAL_MODEL)
        assert isinstance(chunker, ExcelChunker)

    def test_default_falls_back_to_research(self):
        chunker = get_chunker(DocType.OTHER)
        assert isinstance(chunker, ResearchChunker)

    def test_caching(self):
        c1 = get_chunker(DocType.RESEARCH_REPORT)
        c2 = get_chunker(DocType.RESEARCH_REPORT)
        assert c1 is c2  # Same instance

    def test_kwargs_bypass_cache(self):
        c1 = get_chunker(DocType.RESEARCH_REPORT)
        c2 = get_chunker(DocType.RESEARCH_REPORT, max_tokens=500)
        assert c1 is not c2

    def test_clear_cache(self):
        c1 = get_chunker(DocType.RESEARCH_REPORT)
        clear_cache()
        c2 = get_chunker(DocType.RESEARCH_REPORT)
        assert c1 is not c2

    def test_available_chunkers(self):
        names = available_chunkers()
        assert "research_report" in names
        assert "sec_filing" in names
        assert "earnings_transcript" in names
        assert "financial_model" in names
