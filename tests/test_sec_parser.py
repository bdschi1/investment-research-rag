"""Tests for SEC filing section detection."""

from __future__ import annotations

import pytest

from rag.documents.sec_parser import (
    _ITEM_TITLES,
    Section,
    parse_sec_filing,
)


class TestParseSecFiling:
    def test_empty_input(self):
        result = parse_sec_filing([])
        assert result.total_pages == 0
        assert result.sections == []
        assert result.has_sections is False

    def test_no_items_found(self):
        pages = [
            "This is a generic document with no SEC items.",
            "Page two of generic content.",
        ]
        result = parse_sec_filing(pages)
        assert result.total_pages == 2
        assert result.has_sections is False
        assert len(result.sections) == 0

    def test_single_item(self):
        pages = [
            "Item 1. Business\n\nApple designs smartphones."
        ]
        result = parse_sec_filing(pages)
        assert result.has_sections is True
        assert len(result.sections) == 1
        assert result.sections[0].item_number == "1"
        assert result.sections[0].title == "Item 1. Business"

    def test_multiple_items(self):
        pages = [
            "Item 1. Business\n\nApple designs smartphones.",
            "Item 1A. Risk Factors\n\nRisks include...",
            "Item 7. MD&A\n\nRevenue grew 6%.",
        ]
        result = parse_sec_filing(pages)
        assert result.has_sections is True
        assert len(result.sections) == 3

        items = [s.item_number for s in result.sections]
        assert items == ["1", "1A", "7"]

    def test_item_titles_mapping(self):
        """Verify known item numbers get proper titles."""
        pages = [
            "Item 1A. Risk Factors\n\nContent here.",
            "Item 7. Management Discussion\n\nMore content.",
        ]
        result = parse_sec_filing(pages)
        titles = {s.item_number: s.title for s in result.sections}
        assert "Risk Factors" in titles["1A"]
        assert "MD&A" in titles["7"]

    def test_page_boundaries(self):
        pages = [
            "Item 1. Business\n\nPage one content.",
            "More business content on page two.",
            "Item 1A. Risk Factors\n\nRisk content here.",
        ]
        result = parse_sec_filing(pages)
        assert len(result.sections) == 2

        # Item 1 should span pages 0-1
        assert result.sections[0].start_page == 0
        # Item 1A should start at page 2
        assert result.sections[1].start_page == 2

    def test_total_pages_and_chars(self):
        pages = ["Page one.", "Page two.", "Page three."]
        result = parse_sec_filing(pages)
        assert result.total_pages == 3
        assert result.total_chars > 0

    def test_case_insensitive(self):
        pages = ["ITEM 1. BUSINESS\n\nContent.", "item 7. md&a\n\nRevenue."]
        result = parse_sec_filing(pages)
        assert result.has_sections is True
        assert len(result.sections) == 2

    def test_item_with_dash(self):
        pages = ["Item 1A - Risk Factors\n\nContent."]
        result = parse_sec_filing(pages)
        assert result.has_sections is True
        assert result.sections[0].item_number == "1A"

    def test_section_end_char(self):
        """Last section should extend to end of text."""
        pages = [
            "Item 1. Business\n\nContent one.",
            "Item 7. MD&A\n\nContent two. End of filing.",
        ]
        result = parse_sec_filing(pages)
        last = result.sections[-1]
        assert last.end_char == result.total_chars


class TestItemTitles:
    def test_all_standard_items_present(self):
        expected = {"1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A",
                    "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"}
        assert set(_ITEM_TITLES.keys()) == expected

    def test_risk_factors_mapping(self):
        assert _ITEM_TITLES["1A"] == "Risk Factors"

    def test_mda_mapping(self):
        assert _ITEM_TITLES["7"] == "MD&A"


class TestSectionDataclass:
    def test_section_frozen(self):
        s = Section(
            title="Item 1. Business",
            item_number="1",
            start_page=0,
            end_page=2,
            start_char=0,
            end_char=500,
        )
        with pytest.raises(AttributeError):
            s.title = "Changed"  # type: ignore[misc]
