"""Tests for boilerplate filtering."""

from __future__ import annotations

import pytest

from rag.documents.boilerplate import (
    BoilerplateFilter,
    BoilerplateFilterConfig,
)


@pytest.fixture
def bp_filter() -> BoilerplateFilter:
    return BoilerplateFilter()


# ---------------------------------------------------------------------------
# Section-level filtering
# ---------------------------------------------------------------------------


class TestSectionLevel:
    def test_removes_disclosure_appendix(self, bp_filter: BoilerplateFilter):
        text = (
            "Key Findings\n\n"
            "Revenue grew 10% year-over-year.\n\n"
            "Important Disclosures\n\n"
            "This report was prepared by analysts.\n"
            "Not an offer to sell or solicitation."
        )
        result = bp_filter.filter(text)
        assert "Key Findings" in result.text
        assert "Revenue grew" in result.text
        assert "Important Disclosures" not in result.text
        assert result.sections_removed >= 1

    def test_removes_regulatory_disclosures(self, bp_filter: BoilerplateFilter):
        text = (
            "Investment Thesis\n\n"
            "AAPL is a buy.\n\n"
            "Regulatory Disclosures\n\n"
            "This is required regulatory text."
        )
        result = bp_filter.filter(text)
        assert "Investment Thesis" in result.text
        assert "Regulatory Disclosures" not in result.text

    def test_removes_analyst_certification(self, bp_filter: BoilerplateFilter):
        text = (
            "Valuation\n\nPrice target $220.\n\n"
            "Analyst Certification\n\n"
            "I certify that the views expressed..."
        )
        result = bp_filter.filter(text)
        assert "Valuation" in result.text
        assert "Analyst Certification" not in result.text

    def test_removes_general_disclosures(self, bp_filter: BoilerplateFilter):
        text = (
            "Summary\n\nStrong quarter.\n\n"
            "General Disclosures\n\n"
            "Legal boilerplate here."
        )
        result = bp_filter.filter(text)
        assert "Summary" in result.text
        assert "General Disclosures" not in result.text

    def test_stops_at_new_substantive_header(self, bp_filter: BoilerplateFilter):
        text = (
            "Investment Thesis\n\n"
            "AAPL is a buy.\n\n"
            "Important Disclosures\n\n"
            "Legal stuff here.\n"
            "More legal stuff.\n"
            "Financial Summary\n\n"
            "Revenue was $100B."
        )
        result = bp_filter.filter(text)
        assert "Investment Thesis" in result.text
        # The "Financial Summary" section starts with uppercase, short line
        # and should resume inclusion
        assert "Financial Summary" in result.text


# ---------------------------------------------------------------------------
# Paragraph-level filtering
# ---------------------------------------------------------------------------


class TestParagraphLevel:
    def test_removes_not_an_offer(self, bp_filter: BoilerplateFilter):
        text = "Revenue grew.\n\nThis is not an offer to sell securities.\n\nEPS was $6."
        result = bp_filter.filter(text)
        assert "Revenue grew" in result.text
        assert "not an offer to sell" not in result.text
        assert "EPS was $6" in result.text

    def test_removes_past_performance(self, bp_filter: BoilerplateFilter):
        text = "Strong Q4.\n\nPast performance is not indicative of future results.\n\nEnd."
        result = bp_filter.filter(text)
        assert "Strong Q4" in result.text
        assert "Past performance" not in result.text

    def test_removes_forward_looking(self, bp_filter: BoilerplateFilter):
        text = "Guidance raised.\n\nForward-looking statements involve risks.\n\nDone."
        result = bp_filter.filter(text)
        assert "Guidance raised" in result.text
        assert "Forward-looking statements" not in result.text

    def test_removes_safe_harbor(self, bp_filter: BoilerplateFilter):
        text = "Results.\n\nSafe harbor statement: these projections are uncertain.\n\nEnd."
        result = bp_filter.filter(text)
        assert "safe harbor statement" not in result.text.lower()

    def test_removes_sox_certification(self, bp_filter: BoilerplateFilter):
        text = "Filing.\n\nSOX Section 302 certification requirements.\n\nEnd."
        result = bp_filter.filter(text)
        assert "SOX Section" not in result.text

    def test_removes_xbrl(self, bp_filter: BoilerplateFilter):
        text = "Data.\n\nXBRL Instance Document follows.\n\nEnd."
        result = bp_filter.filter(text)
        assert "XBRL Instance" not in result.text

    def test_removes_edgar(self, bp_filter: BoilerplateFilter):
        text = "Filing.\n\nEDGAR Filing Header information.\n\nEnd."
        result = bp_filter.filter(text)
        assert "EDGAR Filing" not in result.text

    def test_removes_sec_commission(self, bp_filter: BoilerplateFilter):
        text = (
            "Overview.\n\n"
            "The Securities and Exchange Commission has not approved these securities.\n\n"
            "End."
        )
        result = bp_filter.filter(text)
        assert "Securities and Exchange Commission has not" not in result.text


# ---------------------------------------------------------------------------
# Protected keywords
# ---------------------------------------------------------------------------


class TestProtectedKeywords:
    def test_protected_paragraph_not_removed(self, bp_filter: BoilerplateFilter):
        """Paragraphs containing protected keywords should be preserved."""
        text = (
            "Overview.\n\n"
            "Material nonpublic information: this is not an offer to sell.\n\n"
            "End."
        )
        result = bp_filter.filter(text)
        # "material nonpublic" is protected, so even though "not an offer to sell"
        # matches a discard pattern, the paragraph should be kept
        assert "Material nonpublic" in result.text

    def test_insider_keyword_protected(self, bp_filter: BoilerplateFilter):
        text = "Data.\n\nInsider trading policy: past performance not indicative.\n\nEnd."
        result = bp_filter.filter(text)
        assert "Insider" in result.text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_disabled_filter(self):
        config = BoilerplateFilterConfig(enabled=False)
        bf = BoilerplateFilter(config)
        text = "Important Disclosures\nAll the legal stuff."
        result = bf.filter(text)
        assert result.text == text
        assert result.chars_removed == 0

    def test_custom_patterns(self):
        config = BoilerplateFilterConfig(
            custom_patterns=[r"proprietary\s+model"]
        )
        bf = BoilerplateFilter(config)
        text = "Analysis.\n\nBased on our proprietary model.\n\nConclusion."
        result = bf.filter(text)
        assert "proprietary model" not in result.text
        assert result.paragraphs_removed >= 1

    def test_filter_result_stats(self, bp_filter: BoilerplateFilter):
        text = (
            "Revenue grew 10%.\n\n"
            "This is not an offer to sell securities.\n\n"
            "Forward-looking statements involve risks.\n\n"
            "EPS was $6.97."
        )
        result = bp_filter.filter(text)
        assert result.chars_removed > 0
        assert result.paragraphs_removed >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text(self, bp_filter: BoilerplateFilter):
        result = bp_filter.filter("")
        assert result.text == ""
        assert result.chars_removed == 0

    def test_no_boilerplate(self, bp_filter: BoilerplateFilter):
        text = "Apple Inc. reported strong quarterly results.\n\nRevenue exceeded expectations."
        result = bp_filter.filter(text)
        assert result.text == text
        assert result.chars_removed == 0

    def test_all_boilerplate(self, bp_filter: BoilerplateFilter):
        text = (
            "Not an offer to sell.\n\n"
            "Past performance is not indicative.\n\n"
            "Forward-looking statements are uncertain."
        )
        result = bp_filter.filter(text)
        assert result.paragraphs_removed >= 3
