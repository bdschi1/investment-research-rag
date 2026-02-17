"""Unified document loader — PDF, DOCX, TXT, XLSX.

Adapted from redflag_ex1_analyst (document_loader.py). Supports both
filesystem paths and in-memory bytes for web/Streamlit uploads.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.documents.schemas import DocumentMetadata, LoadResult

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".xlsx"}


class DocumentLoader:
    """Load documents into a structured ``LoadResult``."""

    def load_file(self, path: str | Path, metadata: DocumentMetadata | None = None) -> LoadResult:
        """Load a document from a filesystem path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        data = path.read_bytes()
        result = self._dispatch(data, ext, metadata)
        result.source_path = str(path)
        return result

    def load_bytes(
        self,
        data: bytes,
        filename: str,
        metadata: DocumentMetadata | None = None,
    ) -> LoadResult:
        """Load a document from in-memory bytes."""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        result = self._dispatch(data, ext, metadata)
        result.source_path = filename
        return result

    # ------------------------------------------------------------------
    # Private dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        data: bytes,
        ext: str,
        metadata: DocumentMetadata | None,
    ) -> LoadResult:
        meta = metadata or DocumentMetadata()
        handlers = {
            ".txt": self._load_txt,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".xlsx": self._load_xlsx,
        }
        handler = handlers[ext]
        result = handler(data)
        result.format = ext.lstrip(".")
        result.metadata = meta
        result.char_count = len(result.text)
        return result

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_txt(data: bytes) -> LoadResult:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                text = data.decode(encoding)
                return LoadResult(text=text, page_texts=[text], page_count=1)
            except UnicodeDecodeError:
                continue
        text = data.decode("utf-8", errors="replace")
        return LoadResult(
            text=text,
            page_texts=[text],
            page_count=1,
            warnings=["Encoding detection fell back to utf-8 with replacements"],
        )

    @staticmethod
    def _load_pdf(data: bytes) -> LoadResult:
        import io

        import pdfplumber

        warnings: list[str] = []
        page_texts: list[str] = []

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    page_texts.append(text)
        except Exception as exc:
            warnings.append(f"PDF extraction error: {exc}")
            return LoadResult(text="", warnings=warnings)

        full_text = "\n\n".join(page_texts)
        if not full_text.strip():
            warnings.append("PDF contains no extractable text (may be scanned/image-only)")

        return LoadResult(
            text=full_text,
            page_texts=page_texts,
            page_count=len(page_texts),
            warnings=warnings,
        )

    @staticmethod
    def _load_docx(data: bytes) -> LoadResult:
        import io

        from docx import Document

        warnings: list[str] = []
        try:
            doc = Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
        except Exception as exc:
            warnings.append(f"DOCX extraction error: {exc}")
            return LoadResult(text="", warnings=warnings)

        return LoadResult(
            text=text,
            page_texts=[text],
            page_count=1,
            warnings=warnings,
        )

    @staticmethod
    def _load_xlsx(data: bytes) -> LoadResult:
        import io

        import pandas as pd

        warnings: list[str] = []
        page_texts: list[str] = []

        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                md_table = df.head(25).to_markdown(index=False)
                header = f"## Sheet: {sheet_name} ({len(df)} rows, {len(df.columns)} cols)\n\n"
                page_texts.append(header + (md_table or "[empty sheet]"))
        except Exception as exc:
            warnings.append(f"Excel extraction error: {exc}")
            return LoadResult(text="", warnings=warnings)

        full_text = "\n\n---\n\n".join(page_texts)
        return LoadResult(
            text=full_text,
            page_texts=page_texts,
            page_count=len(page_texts),
            warnings=warnings,
        )
