"""Tests for DocumentLoader — all formats, error cases, edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.documents.loader import SUPPORTED_EXTENSIONS, DocumentLoader
from rag.documents.schemas import DocType, DocumentMetadata


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


# ---------------------------------------------------------------------------
# TXT loading
# ---------------------------------------------------------------------------


class TestTxtLoader:
    def test_load_txt_file(self, loader: DocumentLoader, sample_txt_file: Path):
        result = loader.load_file(sample_txt_file)
        assert result.text
        assert "Apple" in result.text
        assert result.format == "txt"
        assert result.page_count == 1
        assert result.char_count > 0
        assert result.source_path == str(sample_txt_file)

    def test_load_txt_bytes(self, loader: DocumentLoader, sample_txt_content: str):
        data = sample_txt_content.encode("utf-8")
        result = loader.load_bytes(data, "report.txt")
        assert "Apple" in result.text
        assert result.format == "txt"
        assert result.source_path == "report.txt"

    def test_load_txt_with_metadata(self, loader: DocumentLoader, sample_txt_file: Path):
        meta = DocumentMetadata(doc_type=DocType.RESEARCH_REPORT, ticker="AAPL")
        result = loader.load_file(sample_txt_file, metadata=meta)
        assert result.metadata.ticker == "AAPL"
        assert result.metadata.doc_type == DocType.RESEARCH_REPORT

    def test_load_txt_latin1(self, loader: DocumentLoader, tmp_path: Path):
        text = "Revenue was \xa31.5 billion"  # pound sign in latin-1
        p = tmp_path / "latin1.txt"
        p.write_bytes(text.encode("latin-1"))
        result = loader.load_file(p)
        assert "1.5 billion" in result.text

    def test_load_txt_page_texts(self, loader: DocumentLoader, sample_txt_file: Path):
        result = loader.load_file(sample_txt_file)
        assert len(result.page_texts) == 1
        assert result.page_texts[0] == result.text


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------


class TestPdfLoader:
    def test_load_pdf_file(self, loader: DocumentLoader, sample_pdf_file: Path):
        result = loader.load_file(sample_pdf_file)
        assert result.text
        assert result.format == "pdf"
        assert result.page_count == 3
        assert len(result.page_texts) == 3
        assert result.char_count > 0

    def test_load_pdf_bytes(self, loader: DocumentLoader, sample_pdf_file: Path):
        data = sample_pdf_file.read_bytes()
        result = loader.load_bytes(data, "filing.pdf")
        assert result.text
        assert result.format == "pdf"
        assert result.source_path == "filing.pdf"

    def test_load_pdf_per_page_text(self, loader: DocumentLoader, sample_pdf_file: Path):
        result = loader.load_file(sample_pdf_file)
        # Each page should have content
        for page_text in result.page_texts:
            assert page_text.strip()

    def test_load_pdf_with_metadata(self, loader: DocumentLoader, sample_pdf_file: Path):
        meta = DocumentMetadata(
            doc_type=DocType.SEC_FILING,
            ticker="AAPL",
            filing_type="10-K",
        )
        result = loader.load_file(sample_pdf_file, metadata=meta)
        assert result.metadata.doc_type == DocType.SEC_FILING


# ---------------------------------------------------------------------------
# DOCX loading
# ---------------------------------------------------------------------------


class TestDocxLoader:
    def test_load_docx_file(self, loader: DocumentLoader, sample_docx_file: Path):
        result = loader.load_file(sample_docx_file)
        assert result.text
        assert "Apple" in result.text or "AAPL" in result.text
        assert result.format == "docx"
        assert result.page_count == 1
        assert result.char_count > 0

    def test_load_docx_bytes(self, loader: DocumentLoader, sample_docx_file: Path):
        data = sample_docx_file.read_bytes()
        result = loader.load_bytes(data, "research.docx")
        assert result.text
        assert result.format == "docx"


# ---------------------------------------------------------------------------
# XLSX loading
# ---------------------------------------------------------------------------


class TestXlsxLoader:
    def test_load_xlsx_file(self, loader: DocumentLoader, sample_xlsx_file: Path):
        result = loader.load_file(sample_xlsx_file)
        assert result.text
        assert result.format == "xlsx"
        assert result.page_count == 2  # two sheets
        assert "Summary" in result.text
        assert "Segments" in result.text

    def test_load_xlsx_bytes(self, loader: DocumentLoader, sample_xlsx_file: Path):
        data = sample_xlsx_file.read_bytes()
        result = loader.load_bytes(data, "model.xlsx")
        assert result.text
        assert result.format == "xlsx"

    def test_xlsx_sheet_separator(self, loader: DocumentLoader, sample_xlsx_file: Path):
        result = loader.load_file(sample_xlsx_file)
        # Sheets are separated by ---
        assert "---" in result.text

    def test_xlsx_markdown_tables(self, loader: DocumentLoader, sample_xlsx_file: Path):
        result = loader.load_file(sample_xlsx_file)
        # Should contain markdown table formatting
        assert "|" in result.text


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestLoaderErrors:
    def test_file_not_found(self, loader: DocumentLoader, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            loader.load_file(tmp_path / "nonexistent.pdf")

    def test_unsupported_extension_file(self, loader: DocumentLoader, tmp_path: Path):
        p = tmp_path / "report.csv"
        p.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_file(p)

    def test_unsupported_extension_bytes(self, loader: DocumentLoader):
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_bytes(b"data", "file.json")

    def test_supported_extensions_constant(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


class TestMetadataPropagation:
    def test_default_metadata(self, loader: DocumentLoader, sample_txt_file: Path):
        result = loader.load_file(sample_txt_file)
        assert result.metadata.doc_type == DocType.OTHER
        assert result.metadata.ticker is None

    def test_custom_metadata_preserved(self, loader: DocumentLoader, sample_txt_file: Path):
        meta = DocumentMetadata(
            doc_type=DocType.EARNINGS_TRANSCRIPT,
            ticker="MSFT",
            filing_date="2024-01-25",
        )
        result = loader.load_file(sample_txt_file, metadata=meta)
        assert result.metadata.ticker == "MSFT"
        assert result.metadata.doc_type == DocType.EARNINGS_TRANSCRIPT
        assert result.metadata.filing_date == "2024-01-25"
