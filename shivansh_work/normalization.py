"""Normalization utilities for the VERILENS entity-extraction module.

This file only contains small, reusable, deterministic helpers:

* text normalization (Unicode, Devanagari digits, invisible characters)
* number / amount normalization (lakh, crore, hazar ...)
* date helpers (month names, ISO 8601 building, Indian financial years)

No NLP preprocessing (language detection, claim extraction) is done here -
that is Member 2's responsibility.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, Union

Number = Union[int, float]

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_INVISIBLE_CHARACTERS = dict.fromkeys(ord(c) for c in ("\u200b", "\u200c", "\u200d", "\ufeff"))
_HORIZONTAL_SPACES = re.compile(r"[ \t\u00a0]+")


def normalize_text(text: str) -> str:
    """Return ``text`` in the form used for all matching.

    Steps: Unicode NFC, Devanagari digits -> ASCII digits, removal of
    zero-width characters, collapsing of repeated spaces/tabs.  Newlines are
    kept because they separate clauses.  All character offsets reported by
    the extractor refer to this normalized text.
    """
    if not isinstance(text, str):
        raise TypeError("normalize_text expects a string")
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_DEVANAGARI_DIGITS).translate(_INVISIBLE_CHARACTERS)
    return _HORIZONTAL_SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Numbers and amounts
# --------------------------------------------------------------------------

#: Small number words (English, Hindi, Hinglish) used for things like
#: "do lakh" or "three instalments".
NUMBER_WORDS: Dict[str, int] = {
    normalize_text(word): value
    for word, value in {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
        "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
        "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5, "panch": 5,
        "chhe": 6, "chah": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    }.items()
}

#: Multipliers for Indian / international magnitude words.
UNIT_MULTIPLIERS: Dict[str, int] = {
    normalize_text(word): value
    for word, value in {
        "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "लाख": 100_000,
        "crore": 10_000_000, "crores": 10_000_000, "करोड़": 10_000_000, "करोड": 10_000_000,
        "thousand": 1_000, "hazar": 1_000, "hazaar": 1_000, "हज़ार": 1_000, "हजार": 1_000,
    }.items()
}

#: Frequencies understood by the module.
FREQUENCIES = ("per_year", "per_month", "per_instalment", "one_time")


def parse_number(token: str) -> Optional[Decimal]:
    """Convert a numeric token ("6,000", "2.5", "1,00,000", "do") to Decimal.

    Returns ``None`` when the token is not a valid number.
    """
    if token is None:
        return None
    cleaned = normalize_text(str(token)).lower()
    if cleaned in NUMBER_WORDS:
        return Decimal(NUMBER_WORDS[cleaned])
    try:
        return Decimal(cleaned.replace(",", ""))
    except InvalidOperation:
        return None


def to_amount_value(number_token: str, unit_token: Optional[str] = None) -> Optional[Number]:
    """Combine a number token and an optional magnitude word into a value.

    ``("2.5", "lakh")`` -> ``250000``; ``("6,000", None)`` -> ``6000``.
    Returns ``None`` for invalid numbers or unknown units.
    """
    base = parse_number(number_token)
    if base is None:
        return None
    multiplier = 1
    if unit_token:
        key = normalize_text(unit_token).lower()
        if key not in UNIT_MULTIPLIERS:
            return None
        multiplier = UNIT_MULTIPLIERS[key]
    value = base * multiplier
    return int(value) if value == value.to_integral_value() else float(value)


def annual_equivalent(amount: Number, frequency: Optional[str]) -> Optional[Number]:
    """Return the yearly value when it can be derived without guessing.

    Only ``per_year`` and ``per_month`` frequencies are converted.  Per-
    instalment amounts are never turned into annual amounts here.
    """
    if frequency == "per_year":
        return amount
    if frequency == "per_month":
        return amount * 12
    return None


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

MONTH_NAMES: Dict[str, int] = {
    normalize_text(name): number
    for name, number in {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
        "जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5,
        "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9,
        "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11,
        "दिसंबर": 12, "दिसम्बर": 12,
    }.items()
}


def month_number(name: str) -> Optional[int]:
    """Return 1-12 for an English/Hindi month name, else ``None``."""
    return MONTH_NAMES.get(normalize_text(name).lower())


def build_iso_date(year: int, month: int, day: int) -> Optional[str]:
    """Return ``YYYY-MM-DD`` for a valid calendar date, else ``None``."""
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def build_iso_month(year: int, month: int) -> Optional[str]:
    """Return ``YYYY-MM`` for a valid year/month pair, else ``None``."""
    if 1 <= month <= 12 and 1 <= year <= 9999:
        return f"{year:04d}-{month:02d}"
    return None


def financial_year_bounds(start_year: int, end_token: str) -> Optional[Dict[str, str]]:
    """Return the label and bounds of an Indian financial year (1 Apr - 31 Mar).

    ``end_token`` may be a 2-digit ("26") or 4-digit ("2026") year.  The end
    year must be ``start_year + 1``; otherwise ``None`` is returned.
    """
    if not end_token.isdigit():
        return None
    end_year = int(end_token)
    if len(end_token) == 2:
        if end_year != (start_year + 1) % 100:
            return None
        end_year = start_year + 1
    elif end_year != start_year + 1:
        return None
    return {
        "label": f"{start_year}-{str(end_year)[-2:]}",
        "start_iso": f"{start_year:04d}-04-01",
        "end_iso": f"{end_year:04d}-03-31",
    }
