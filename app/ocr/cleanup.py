"""Lightweight post-processing for noisy marriage-license OCR text."""

from __future__ import annotations

import re

# Known pre-printed labels that often glue to handwritten values.
# Keep labels long/specific — short tokens like "Age" false-match inside "Marriage".
_FIELD_LABELS = (
    "Residence of Groom",
    "Residence of Bride",
    "Occupation",
    "Nativity",
    "Color",
    "Name of Person consenting if Groom is a Minor",
    "Name of Person consenting if Bride is a Minor",
    "Signature of Officiating Clergyman",
    "Official Character",
    "Address",
)

# Whole-word OCR confusions common on these forms (apply conservatively).
_WORD_FIXES = {
    "NCNE": "NONE",
    "N0NE": "NONE",
    "NONF": "NONE",
    "ALLIME": "ALLINE",
    "PENNSLYVANIA": "PENNSYLVANIA",
    "Clegyman": "Clergyman",
    "Certity": "Certify",
    "Adres": "Address",
    "Nume": "Name",
    "troom": "Groom",
    "wbether": "whether",
    "sinzle": "single",
    "contractinz": "contracting",
}

# Regex replacements run before label splitting.
_REGEX_FIXES: list[tuple[str, str]] = [
    (r"\bI\s*Hereby\b", "I Hereby"),
    (r"\bF\s*Hereby\b", "I Hereby"),
    (r"\bAPPLI\b", "APPLICATION"),
    (r"Color\s+COL\.?\s*Nativity", "Color COL\nNativity"),
    (r"COLNativity", "Color COL\nNativity"),
    (r"Residence of Bride(?=[A-Z0-9])", "Residence of Bride\n"),
    (r"Residence of Groom(?=[A-Z0-9])", "Residence of Groom\n"),
    (r"OccupationONE\b", "Occupation\nNONE"),
    (r"Occupation\.ONE\b", "Occupation\nNONE"),
    (r"\bBa t\b", "Balt"),
    (r"\bBaltimore ND\.\b", "BALTIMORE MD."),
    (r"\bTHT\b", "WHT"),
    (r"\bOILER\b", "BOILER"),
    (r"\bJo day ot\b", "20 day of"),
    (r"\bot Aug\b", "of Aug"),
    (r"No\.\s*\n\s*APPLICATION", "No. \nAPPLICATION"),
]


def clean_ocr_text(text: str) -> str:
    """Normalize common OCR artifacts on Baltimore marriage license forms."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    for pattern, replacement in _REGEX_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    for label in _FIELD_LABELS:
        cleaned = re.sub(
            rf"(?<=\S)({re.escape(label)})\b",
            r"\n\1",
            cleaned,
        )

    for wrong, right in _WORD_FIXES.items():
        cleaned = re.sub(rf"\b{re.escape(wrong)}\b", right, cleaned)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)

    return cleaned.strip()
