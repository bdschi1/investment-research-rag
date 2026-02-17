"""Earnings transcript parser — detect speakers, sections, Q&A boundaries.

Parses common earnings call transcript formats into structured sections
with speaker attribution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class TranscriptSectionType(StrEnum):
    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"
    OPERATOR = "operator"


@dataclass(frozen=True)
class TranscriptSection:
    """A single speaker turn in an earnings transcript."""

    speaker: str
    role: str
    section_type: TranscriptSectionType
    text: str
    start_char: int
    end_char: int


@dataclass
class ParsedTranscript:
    """Result of parsing an earnings transcript."""

    sections: list[TranscriptSection]
    has_qa: bool
    speaker_count: int


# Speaker line patterns (common formats):
# "Tim Cook -- Chief Executive Officer"
# "Tim Cook - CEO"
# "Tim Cook, Chief Executive Officer"
_SPEAKER_RE = re.compile(
    r"^([A-Z][a-zA-Z\.\-' ]+ [A-Z][a-zA-Z\.\-' ]+)"
    r"\s*[-\u2013\u2014,]+\s*"
    r"(.+)$",
    re.MULTILINE,
)

# Q&A section boundary
_QA_BOUNDARY_RE = re.compile(
    r"(?i)^(?:question[- ]?and[- ]?answer|q\s*&\s*a|q&a\s+session)",
    re.MULTILINE,
)

# Operator lines
_OPERATOR_RE = re.compile(r"(?i)^operator\s*$", re.MULTILINE)


def parse_transcript(text: str) -> ParsedTranscript:
    """Parse an earnings transcript into speaker sections.

    Args:
        text: Full transcript text.

    Returns:
        A ``ParsedTranscript`` with speaker-attributed sections.
    """
    if not text.strip():
        return ParsedTranscript(sections=[], has_qa=False, speaker_count=0)

    # Detect Q&A boundary
    qa_match = _QA_BOUNDARY_RE.search(text)
    qa_start = qa_match.start() if qa_match else len(text)

    # Find all speaker turns
    speaker_matches = list(_SPEAKER_RE.finditer(text))
    if not speaker_matches:
        return ParsedTranscript(sections=[], has_qa=False, speaker_count=0)

    sections: list[TranscriptSection] = []
    speakers_seen: set[str] = set()

    for i, m in enumerate(speaker_matches):
        speaker = m.group(1).strip()
        role = m.group(2).strip()
        start = m.end()
        end = speaker_matches[i + 1].start() if i + 1 < len(speaker_matches) else len(text)

        body = text[start:end].strip()
        if not body:
            continue

        is_qa = m.start() >= qa_start
        section_type = TranscriptSectionType.QA if is_qa else TranscriptSectionType.PREPARED_REMARKS

        speakers_seen.add(speaker)
        sections.append(TranscriptSection(
            speaker=speaker,
            role=role,
            section_type=section_type,
            text=body,
            start_char=m.start(),
            end_char=end,
        ))

    has_qa = qa_match is not None
    logger.info(
        "Parsed transcript: %d speaker turns, %d unique speakers, Q&A=%s",
        len(sections), len(speakers_seen), has_qa,
    )

    return ParsedTranscript(
        sections=sections,
        has_qa=has_qa,
        speaker_count=len(speakers_seen),
    )
