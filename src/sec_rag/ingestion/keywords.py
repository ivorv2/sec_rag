"""Canonical keyword-to-section mapping for employment agreement analysis.

Single source of truth imported by both the chunker (for section classification)
and the filter (for agreement detection). Extracted to avoid hidden coupling.
"""

from sec_rag.models.documents import SectionType

# Mapping from lowercase keywords to SectionType.
# Order matters: longer/more-specific keywords should appear first so they match
# before shorter substrings (e.g., "base salary" before "salary").
KEYWORD_TO_SECTION: list[tuple[str, SectionType]] = [
    ("base salary", SectionType.COMPENSATION),
    ("salary", SectionType.COMPENSATION),
    ("compensation", SectionType.COMPENSATION),
    ("duties", SectionType.DUTIES),
    ("responsibilities", SectionType.DUTIES),
    ("termination", SectionType.TERMINATION),
    ("non-compete", SectionType.NON_COMPETE),
    ("non-competition", SectionType.NON_COMPETE),
    ("restrictive covenant", SectionType.NON_COMPETE),
    ("non-solicitation", SectionType.NON_COMPETE),
    ("equity", SectionType.EQUITY),
    ("stock option", SectionType.EQUITY),
    ("rsu", SectionType.EQUITY),
    ("benefits", SectionType.BENEFITS),
    ("insurance", SectionType.BENEFITS),
    ("vacation", SectionType.VACATION),
    ("paid time off", SectionType.VACATION),
    ("pto", SectionType.VACATION),
    ("bonus", SectionType.BONUS),
    ("incentive", SectionType.BONUS),
    ("severance", SectionType.SEVERANCE),
    ("separation", SectionType.SEVERANCE),
    ("confidential", SectionType.CONFIDENTIALITY),
    ("non-disclosure", SectionType.CONFIDENTIALITY),
    ("nda", SectionType.CONFIDENTIALITY),
    ("intellectual property", SectionType.INTELLECTUAL_PROPERTY),
    ("invention", SectionType.INTELLECTUAL_PROPERTY),
    ("work product", SectionType.INTELLECTUAL_PROPERTY),
    ("governing law", SectionType.GOVERNING_LAW),
    ("jurisdiction", SectionType.GOVERNING_LAW),
    ("term of employment", SectionType.TERM_AND_RENEWAL),
    ("term and renewal", SectionType.TERM_AND_RENEWAL),
    ("commencement", SectionType.TERM_AND_RENEWAL),
    ("change in control", SectionType.CHANGE_OF_CONTROL),
    ("change of control", SectionType.CHANGE_OF_CONTROL),
    ("arbitration", SectionType.ARBITRATION),
    ("dispute resolution", SectionType.ARBITRATION),
    ("indemnification", SectionType.INDEMNIFICATION),
    ("indemnity", SectionType.INDEMNIFICATION),
]

# Unique keyword strings for fast membership checks (used by filter.py).
SECTION_KEYWORDS: list[str] = sorted({kw for kw, _ in KEYWORD_TO_SECTION})
