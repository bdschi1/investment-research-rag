"""Tests for smart page scoring."""

from __future__ import annotations

from rag.chunking.scoring import ScoredPage, score_page, select_pages


# Helper to create text that exceeds the 200-char minimum to avoid short-page penalty
def _pad(text: str, min_len: int = 250) -> str:
    """Pad text to avoid the short-page penalty."""
    if len(text) >= min_len:
        return text
    padding = " This is additional content to reach the minimum page length." * 5
    return text + padding


class TestScorePage:
    def test_high_value_header_bonus(self):
        text = _pad("Executive Summary\n\nKey findings from our analysis of Apple Inc.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0  # +3.0 for high-value header

    def test_valuation_header(self):
        text = _pad("Valuation\n\nOur DCF model implies fair value of $195 per share.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0

    def test_risk_factors_header(self):
        text = _pad("Risk Factors\n\nForeign exchange headwinds remain a concern.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0

    def test_table_bonus(self):
        text = _pad("| Year | Revenue | EPS |\n|---|---|---|\n| 2024 | $395B | $6.97 |")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 2.0  # +2.0 for table

    def test_digit_density_bonus(self):
        # >5% digits — use mostly numeric content
        text = _pad("Revenue: $94,932,000,000. EPS: $6.97. P/E: 28.4x. Shares: 15,408,095,000.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 1.5  # +1.5 for high digit density

    def test_first_page_bonus(self):
        text = _pad("Generic content without any special markers or headers here.")
        score = score_page(text, page_index=0, total_pages=20)
        assert score >= 1.0  # +1.0 for first page

    def test_second_page_bonus(self):
        text = _pad("Generic content without any special markers or headers here.")
        score = score_page(text, page_index=1, total_pages=20)
        assert score >= 1.0

    def test_last_page_bonus(self):
        text = _pad("Generic content without any special markers or headers here.")
        score = score_page(text, page_index=19, total_pages=20)
        assert score >= 1.0

    def test_middle_page_no_position_bonus(self):
        # Use only alphabetic chars (no digits) and no headers
        text = "a" * 250
        score = score_page(text, page_index=10, total_pages=20)
        # No position bonus, no content bonus, no penalty
        assert score == 0.0

    def test_short_page_penalty(self):
        text = "Cover page."  # <200 chars
        score = score_page(text, page_index=5, total_pages=20)
        assert score <= -2.0  # -2.0 penalty

    def test_cumulative_scoring(self):
        # High-value header + table + digits + first page
        text = _pad(
            "Executive Summary\n\n"
            "| Metric | Value |\n|---|---|\n"
            "| Revenue | $94,932,000,000 |\n"
            "| EPS | $6.97 |\n"
            "| Net Income | $25,010,000,000 |"
        )
        score = score_page(text, page_index=0, total_pages=20)
        # Should get +3.0 (header) + 2.0 (table) + 1.5 (digits) + 1.0 (first page)
        assert score >= 5.0

    def test_md_and_a_header(self):
        text = _pad("MD&A\n\nRevenue discussion and analysis of quarterly results.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0

    def test_investment_thesis_header(self):
        text = _pad("Investment Thesis\n\nWe are bullish on AAPL given strong services growth.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0

    def test_earnings_header(self):
        text = _pad("Earnings\n\nQuarterly earnings exceeded consensus estimates.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0

    def test_guidance_header(self):
        text = _pad("Guidance\n\nManagement raised full-year guidance citing strong demand.")
        score = score_page(text, page_index=5, total_pages=20)
        assert score >= 3.0


class TestSelectPages:
    def test_select_all_when_no_limit(self):
        pages = ["Page one." * 30, "Page two." * 30, "Page three." * 30]
        selected = select_pages(pages)
        assert len(selected) == 3

    def test_sorted_by_page_order(self):
        pages = ["A" * 300, "B" * 300, "C" * 300, "D" * 300, "E" * 300]
        selected = select_pages(pages)
        indices = [s.page_index for s in selected]
        assert indices == sorted(indices)

    def test_budget_selection(self):
        pages = [f"Page {i} content." * 50 for i in range(20)]
        # Add high-value page in the middle
        pages[10] = _pad(
            "Executive Summary\n\n| Data | Value |\n|---|---|\n| Rev | $100B |",
            min_len=300,
        )
        selected = select_pages(pages, max_pages=5)
        assert len(selected) == 5
        # First two and last page should always be included
        indices = {s.page_index for s in selected}
        assert 0 in indices
        assert 1 in indices
        assert 19 in indices

    def test_must_include_pages(self):
        """First 2 + last page always included regardless of score."""
        pages = ["x" * 50] * 10  # All short pages (get penalty)
        selected = select_pages(pages, max_pages=3)
        indices = {s.page_index for s in selected}
        assert 0 in indices
        assert 1 in indices
        assert 9 in indices  # last page

    def test_high_value_pages_preferred(self):
        pages = ["Generic text." * 30 for _ in range(10)]
        pages[5] = _pad(
            "Investment Thesis\n\nKey Findings\n\nRevenue: $94,932,000,000",
            min_len=300,
        )
        selected = select_pages(pages, max_pages=5)
        indices = {s.page_index for s in selected}
        assert 5 in indices  # High-value page should be included

    def test_empty_pages(self):
        selected = select_pages([])
        assert selected == []

    def test_single_page(self):
        selected = select_pages(["Only page."])
        assert len(selected) == 1

    def test_scored_page_dataclass(self):
        sp = ScoredPage(page_index=0, text="Hello", score=3.5)
        assert sp.page_index == 0
        assert sp.score == 3.5

    def test_max_pages_greater_than_total(self):
        pages = ["Page one." * 30, "Page two." * 30]
        selected = select_pages(pages, max_pages=10)
        assert len(selected) == 2  # Only 2 pages available
