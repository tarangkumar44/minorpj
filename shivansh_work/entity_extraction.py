"""Deterministic entity extraction for VERILENS (Member 3).

What this file does
-------------------
* loads the scheme registry (``data/scheme_registry.json``) and the keyword
  rules (``data/extraction_rules.json``) - both external, both editable;
* extracts scheme names, amounts, dates, age limits and eligibility
  conditions from the text received from the NLP module;
* returns plain, JSON-serializable dictionaries.

What it does NOT do
-------------------
No language detection, no preprocessing, no semantic (NLI) comparison.  Those
belong to other members.  No sentence in this file is a "demo" sentence: all
logic works on whatever text the NLP module sends.

All character offsets (``span``) refer to ``processed_text``, i.e. the input
after :func:`normalization.normalize_text`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

from .normalization import (
    MONTH_NAMES,
    NUMBER_WORDS,
    annual_equivalent,
    build_iso_date,
    build_iso_month,
    financial_year_bounds,
    month_number,
    normalize_text,
    parse_number,
    to_amount_value,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_REGISTRY_PATH = DEFAULT_DATA_DIR / "scheme_registry.json"
DEFAULT_RULES_PATH = DEFAULT_DATA_DIR / "extraction_rules.json"


class ConfigError(Exception):
    """Raised when the registry or rules file is missing or malformed."""


# --------------------------------------------------------------------------
# Small regex helpers
# --------------------------------------------------------------------------

_LETTER = r"A-Za-z\u0900-\u097F"
_NO_LETTER_BEFORE = rf"(?<![{_LETTER}])"
_NO_LETTER_AFTER = rf"(?![{_LETTER}])"


def _compile(pattern: str, flags: int = re.IGNORECASE) -> Pattern[str]:
    """Compile a pattern after normalizing its Unicode form (same as the text)."""
    return re.compile(normalize_text(pattern), flags)


def _edge_before(first_char: str) -> str:
    return r"(?<![A-Za-z0-9])" if first_char.isascii() and first_char.isalnum() else r"(?<![\u0900-\u097F])"


def _edge_after(last_char: str) -> str:
    return r"(?![A-Za-z0-9])" if last_char.isascii() and last_char.isalnum() else r"(?![\u0900-\u097F])"


def build_keyword_regex(words: Iterable[str]) -> Pattern[str]:
    """Build one case-insensitive regex that matches any of ``words``.

    Latin words need word boundaries; Devanagari words may not be glued to
    other Devanagari letters.  Longer words win over their prefixes.
    """
    unique = sorted({normalize_text(w) for w in words if isinstance(w, str) and w.strip()}, key=len, reverse=True)
    parts = [
        _edge_before(word[0]) + r"\s+".join(re.escape(tok) for tok in word.split()) + _edge_after(word[-1])
        for word in unique
    ]
    return re.compile("|".join(parts) if parts else r"(?!x)x", re.IGNORECASE)


def load_json_file(path: Path) -> Any:
    """Read a JSON file; raise :class:`ConfigError` with a clear message."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc}") from exc


def mask_spans(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    """Blank out ``spans`` so later extractors do not re-read them."""
    chars = list(text)
    for start, end in spans:
        for index in range(start, min(end, len(chars))):
            chars[index] = " "
    return "".join(chars)


def _context(text: str, start: int, end: int, width: int = 40) -> str:
    return text[max(0, start - width): end + width].strip()


def _overlaps(span: Tuple[int, int], taken: List[Tuple[int, int]]) -> bool:
    return any(span[0] < t_end and t_start < span[1] for t_start, t_end in taken)


# --------------------------------------------------------------------------
# Scheme registry
# --------------------------------------------------------------------------

class SchemeRegistry:
    """Maps aliases (English / Hindi / Hinglish) to canonical scheme names."""

    def __init__(self, schemes: List[Dict[str, Any]], version: str = "unknown", min_alias_length: int = 3):
        self.version = version
        self.warnings: List[str] = []
        self._records: Dict[str, Dict[str, str]] = {}
        self._patterns: List[Tuple[Pattern[str], str, str, str]] = []
        alias_owner: Dict[str, str] = {}

        for index, scheme in enumerate(schemes):
            if not isinstance(scheme, dict) or not scheme.get("scheme_id") or not scheme.get("canonical_name"):
                raise ConfigError(f"Registry entry {index} needs 'scheme_id' and 'canonical_name'")
            scheme_id, canonical = str(scheme["scheme_id"]), str(scheme["canonical_name"])
            if scheme_id in self._records:
                raise ConfigError(f"Duplicate scheme_id in registry: {scheme_id}")
            self._records[scheme_id] = {"scheme_id": scheme_id, "canonical_name": canonical}

            aliases = scheme.get("aliases", {})
            if not isinstance(aliases, dict):
                raise ConfigError(f"'aliases' of {scheme_id} must map language -> list")
            entries = [("canonical", canonical)] + [
                (lang, alias) for lang, items in aliases.items() for alias in items
            ]
            for language, alias in entries:
                clean = normalize_text(alias) if isinstance(alias, str) else ""
                if len(clean) < min_alias_length:
                    self.warnings.append(f"Alias '{alias}' of {scheme_id} ignored (shorter than {min_alias_length})")
                    continue
                key = clean.casefold()
                if alias_owner.setdefault(key, scheme_id) != scheme_id:
                    self.warnings.append(f"Alias '{alias}' is used by {alias_owner[key]} and {scheme_id}; kept for the first")
                    continue
                self._patterns.append((self._alias_regex(clean), scheme_id, language, clean))

    @staticmethod
    def _alias_regex(alias: str) -> Pattern[str]:
        tokens = [t for t in re.split(r"[\s\-_.]+", alias) if t]
        body = r"[\s\-_.]*".join(re.escape(t) for t in tokens)
        return re.compile(_edge_before(alias[0]) + body + _edge_after(alias[-1]), re.IGNORECASE)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "SchemeRegistry":
        data = load_json_file(Path(path))
        if not isinstance(data, dict) or not isinstance(data.get("schemes"), list):
            raise ConfigError(f"{path}: expected an object with a 'schemes' list")
        return cls(data["schemes"], str(data.get("registry_version", "unknown")), int(data.get("min_alias_length", 3)))

    def get(self, scheme_id: str) -> Optional[Dict[str, str]]:
        return self._records.get(scheme_id)

    def lookup_alias(self, alias_text: str) -> Optional[Dict[str, str]]:
        """Return the scheme whose alias equals ``alias_text`` (whole string)."""
        candidate = normalize_text(alias_text)
        for pattern, scheme_id, _lang, _alias in self._patterns:
            if pattern.fullmatch(candidate):
                return self._records[scheme_id]
        return None

    def find(self, text: str) -> List[Dict[str, Any]]:
        """Find schemes in ``text`` (longest alias wins, one entry per scheme)."""
        hits = []
        for pattern, scheme_id, language, alias in self._patterns:
            for match in pattern.finditer(text):
                hits.append((match.start(), match.end(), scheme_id, language, alias))
        hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
        taken: List[Tuple[int, int]] = []
        accepted = []
        for start, end, scheme_id, language, alias in hits:
            if not _overlaps((start, end), taken):
                taken.append((start, end))
                accepted.append((start, end, scheme_id, language, alias))
        accepted.sort()

        found: Dict[str, Dict[str, Any]] = {}
        for start, end, scheme_id, language, alias in accepted:
            entry = found.get(scheme_id)
            if entry is None:
                record = self._records[scheme_id]
                found[scheme_id] = {
                    "scheme_id": scheme_id,
                    "canonical_name": record["canonical_name"],
                    "matched_text": text[start:end],
                    "matched_alias": alias,
                    "alias_language": language,
                    "span": [start, end],
                    "all_spans": [[start, end]],
                    "mentions": 1,
                    "source": "text",
                }
            else:
                entry["mentions"] += 1
                entry["all_spans"].append([start, end])
        return list(found.values())


# --------------------------------------------------------------------------
# Keyword rules
# --------------------------------------------------------------------------

class ExtractionRules:
    """Compiled keyword lists loaded from ``extraction_rules.json``."""

    def __init__(self, data: Dict[str, Any]):
        try:
            self.version = str(data.get("rules_version", "unknown"))
            self.eligibility: Dict[str, Dict[str, Pattern[str]]] = {
                category: {name: build_keyword_regex(words) for name, words in values.items()}
                for category, values in data["eligibility_categories"].items()
            }
            bounds = data["bound_keywords"]
            self.upper_before = build_keyword_regex(bounds["upper_before"])
            self.lower_before = build_keyword_regex(bounds["lower_before"])
            self.upper_after = build_keyword_regex(bounds["upper_after"])
            self.lower_after = build_keyword_regex(bounds["lower_after"])
            roles = data["amount_role_keywords"]
            self.income_words = build_keyword_regex(roles["income"])
            self.benefit_words = build_keyword_regex(roles["benefit"])
            self.non_amount_nouns = build_keyword_regex(data["non_amount_nouns"])
            self.land_units = {name: build_keyword_regex(words) for name, words in data["land_units"].items()}
            self.duration_words = build_keyword_regex(data["duration_words"])
        except (KeyError, AttributeError, TypeError) as exc:
            raise ConfigError(f"Extraction rules file is malformed: {exc!r}") from exc

    @classmethod
    def from_file(cls, path: Path = DEFAULT_RULES_PATH) -> "ExtractionRules":
        data = load_json_file(Path(path))
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a JSON object")
        return cls(data)


# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------

_WORD_NUMBERS = "|".join(re.escape(w) for w in sorted(NUMBER_WORDS, key=len, reverse=True))
_NUM_DIGITS = r"(?<![\d,])(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)"
_NUM = rf"(?:{_NUM_DIGITS}|{_NO_LETTER_BEFORE}(?:{_WORD_NUMBERS}){_NO_LETTER_AFTER})"
_UNIT = rf"(?:lakhs?|lacs?|crores?|thousand|hazaa?r|लाख|करोड़|करोड|हज़ार|हजार){_NO_LETTER_AFTER}"
_GROUPED_OR_LONG = r"(?<![\d,])(?:\d{1,3}(?:,\d{2,3})+|\d{4,})(?:\.\d+)?"
_CUR_PREFIX = r"(?:₹|(?<![A-Za-z])(?:Rs\.?|INR|rupees?)|रु\.?|रू\.?|रुपये|रुपए|रूपये|रुपया)"
_CUR_SUFFIX = rf"(?:rupees?|rupaye|rupay[ea]?|rupiya|Rs\.?|INR|रुपये|रुपए|रूपये|रुपया|रूपए|रुपयों){_NO_LETTER_AFTER}"

_AMOUNT_PREFIXED = _compile(rf"{_CUR_PREFIX}\s*(?P<num>{_NUM})(?:\s*(?P<unit>{_UNIT}))?")
_AMOUNT_SUFFIXED = _compile(rf"(?P<num>{_NUM})(?:\s*(?P<unit>{_UNIT}))?\s*{_CUR_SUFFIX}")
_AMOUNT_BARE_UNIT = _compile(rf"(?P<num>{_NUM})\s*(?P<unit>{_UNIT})")
_AMOUNT_BARE_FREQ = _compile(
    rf"(?P<num>{_GROUPED_OR_LONG})(?=\s*(?:(?:per|a|each|every)\s+(?:year|annum|month)|annually|monthly|yearly"
    r"|/\s*(?:year|month)|प्रति\s*(?:वर्ष|माह)|सालाना|मासिक))"
)

_INSTALMENT_COUNT = _compile(
    rf"(?P<n>{_NUM})\s*(?:equal\s+|equated\s+)?(?:instal+ments?|किस्त[\u0900-\u097F]*|kist(?:on|o|h|e)?){_NO_LETTER_AFTER}"
)
_INSTALMENT_OF_BEFORE = _compile(r"instal+ments?\s+of\s*$")
_EACH_AT_START = _compile(r"\s*each\b")
_ANNUAL_ADJECTIVE = _compile(
    rf"{_NO_LETTER_BEFORE}(?:annual|yearly){_NO_LETTER_AFTER}|वार्षिक|सालाना"
    r"|प्रति\s*वर्ष|प्रतिवर्ष|प्रति\s*साल|हर\s+(?:साल|वर्ष)")
_MONTHLY_ADJECTIVE = _compile(
    rf"{_NO_LETTER_BEFORE}monthly{_NO_LETTER_AFTER}|मासिक"
    r"|प्रति\s*माह|प्रतिमाह|प्रति\s*महीने?|हर\s+(?:महीने|माह)")

_FREQUENCY_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("per_instalment", _compile(
        r"(?:per|each|every)\s+(?:instal+ment|kist\w*)|(?:प्रति|हर|प्रत्येक)\s*किस्त|har\s+kist")),
    ("per_year", _compile(
        rf"(?<![A-Za-z])(?:per|a|each|every)\s+(?:year|annum|yr){_NO_LETTER_AFTER}|/\s*(?:year|yr|annum)|p\.a\."
        r"|annual(?:ly)?|yearly|सालाना|वार्षिक|प्रति\s*वर्ष|प्रतिवर्ष|प्रति\s*साल|हर\s+(?:साल|वर्ष)"
        r"|(?:per|har)\s+saal|saalana|salana|साल\s+में|वर्ष\s+में")),
    ("per_month", _compile(
        rf"(?<![A-Za-z])(?:per|a|each|every)\s+month{_NO_LETTER_AFTER}|/\s*month|monthly|प्रति\s*माह|प्रतिमाह"
        r"|प्रति\s*महीने?|मासिक|हर\s+(?:महीने|माह)|(?:per|har)\s+(?:mahine|mahina|maah)")),
    ("one_time", _compile(r"one[-\s]time|lump\s*sum|एकमुश्त|एक\s+बार|ek\s+baar|\bonce\b")),
]

_CLAUSE_END = re.compile(r"[;।\n]|\.\s+(?!\d)")


def _cut_at_clause_end(fragment: str) -> str:
    match = _CLAUSE_END.search(fragment)
    return fragment[: match.start()] if match else fragment


def _find_instalment_count(text: str) -> Optional[int]:
    for match in _INSTALMENT_COUNT.finditer(text):
        number = parse_number(match.group("n"))
        if number is not None and number == number.to_integral_value() and 1 < number <= 24:
            return int(number)
    return None


def _detect_frequency(before: str, after: str, instalment_count: Optional[int]) -> Optional[str]:
    if _INSTALMENT_OF_BEFORE.search(before):
        return "per_instalment"
    found = []
    for name, pattern in _FREQUENCY_PATTERNS:
        match = pattern.search(after)
        if match:
            found.append((match.start(), name))
    if instalment_count and _EACH_AT_START.match(after):
        found.append((0, "per_instalment"))
    if found:
        return min(found)[1]
    tail = before[-30:]
    if _ANNUAL_ADJECTIVE.search(tail):
        return "per_year"
    if _MONTHLY_ADJECTIVE.search(tail):
        return "per_month"
    return None


def _detect_amount_role(before: str, after: str, rules: ExtractionRules) -> str:
    if rules.income_words.search(before[-35:]) or rules.income_words.search(after[:12]):
        return "income_limit"
    if rules.benefit_words.search(before[-45:]) or rules.benefit_words.search(after[:30]):
        return "benefit"
    return "unspecified"


def _find_amount_matches(text: str, rules: ExtractionRules) -> List[Dict[str, Any]]:
    taken: List[Tuple[int, int]] = []
    found: List[Dict[str, Any]] = []
    for pattern, explicit in (
        (_AMOUNT_PREFIXED, True), (_AMOUNT_SUFFIXED, True),
        (_AMOUNT_BARE_UNIT, False), (_AMOUNT_BARE_FREQ, False),
    ):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, taken):
                continue
            if not explicit:
                position = span[1]
                while position < len(text) and text[position] == " ":
                    position += 1
                if rules.non_amount_nouns.match(text, position):
                    continue  # e.g. "5 lakh farmers" is a head-count, not money
            taken.append(span)
            found.append({
                "span": span,
                "num": match.group("num"),
                "unit": match.groupdict().get("unit"),
                "explicit": explicit,
            })
    return sorted(found, key=lambda item: item["span"])


def extract_amounts(text: str, rules: ExtractionRules, warnings: List[str]) -> List[Dict[str, Any]]:
    """Extract and normalize monetary amounts from ``text``."""
    matches = _find_amount_matches(text, rules)
    instalment_count = _find_instalment_count(text)
    results: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    implicit_currency_seen = False

    for index, item in enumerate(matches):
        start, end = item["span"]
        value = to_amount_value(item["num"], item["unit"])
        if value is None:
            warnings.append(f"Invalid amount ignored: '{text[start:end]}'")
            continue
        previous_end = matches[index - 1]["span"][1] if index else 0
        next_start = matches[index + 1]["span"][0] if index + 1 < len(matches) else len(text)
        before = text[previous_end:start][-60:]
        after = _cut_at_clause_end(text[end:next_start][:60])

        frequency = _detect_frequency(before, after, instalment_count)
        role = _detect_amount_role(before, after, rules)
        key = (value, frequency, role)
        if key in results:
            results[key]["occurrences"] += 1
            continue
        if not item["explicit"]:
            implicit_currency_seen = True
        results[key] = {
            "type": "amount",
            "text": text[start:end],
            "span": [start, end],
            "amount": value,
            "currency": "INR",
            "currency_explicit": item["explicit"],
            "frequency": frequency,
            "annual_equivalent": annual_equivalent(value, frequency),
            "instalment_count": instalment_count,
            "role": role,
            "context": _context(text, start, end),
            "occurrences": 1,
        }
    if implicit_currency_seen:
        warnings.append("Some amounts have no explicit currency symbol/word; INR was assumed")
    return list(results.values())


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_MONTH_ALTERNATIVES = "|".join(re.escape(k) for k in sorted(MONTH_NAMES, key=len, reverse=True))
_MONTH = rf"{_NO_LETTER_BEFORE}(?P<month>{_MONTH_ALTERNATIVES}){_NO_LETTER_AFTER}"
_YEAR = r"(?:19|20)\d{2}"

_DATE_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("iso", _compile(r"(?<![\d/.-])(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)")),
    ("numeric", _compile(r"(?<![\d/.-])(?P<d>\d{1,2})[/.\-](?P<m>\d{1,2})[/.\-](?P<y>\d{4})(?!\d)")),
    ("day_month_text", _compile(
        rf"(?<!\d)(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?{_MONTH},?\s*(?P<y>{_YEAR})?(?!\d)")),
    ("month_day_text", _compile(
        rf"{_MONTH}\s+(?P<d>\d{{1,2}})(?!\d)(?:st|nd|rd|th)?,?\s*(?P<y>{_YEAR})?(?!\d)")),
    ("month_year", _compile(rf"{_MONTH},?\s*(?P<y>{_YEAR})(?!\d)")),
    ("financial_year", _compile(
        r"(?:(?:FY|F\.Y\.|financial\s+year|वित्त\s+वर्ष|वित्तीय\s+वर्ष)\s*[:\-]?\s*)?"
        r"(?<![\d/.-])(?P<y>(?:19|20)\d{2})\s*[-–/]\s*(?P<end>\d{4}|\d{2})(?!\d)")),
]

_DEADLINE_WORDS = _compile(
    r"last\s+date|deadline|due\s+date|closing\s+date|apply\s+(?:by|before)|submit(?:ted)?\s+(?:by|before)"
    r"|applications?\s+(?:close|end)|अंतिम\s+(?:तिथि|तारीख)|आखिरी\s+(?:तिथि|तारीख)|antim\s+(?:tithi|tareekh)"
    r"|akhri\s+tareekh|last\s+tareekh")
_APPLY_WORDS = _compile(r"apply|application|आवेदन|aavedan")
_START_TAIL = _compile(
    r"(?:from|since|starting(?:\s+from)?|effective(?:\s+from)?|w\.?e\.?f\.?|begin(?:s|ning)?(?:\s+from)?|शुरू|प्रभावी)\s*[:\-]?\s*$")
_END_TAIL = _compile(
    r"(?:till|until|upto|up\s+to|to|through|ending|ends?(?:\s+on)?|expires?(?:\s+on)?|valid\s+(?:till|until|upto)|समाप्त)\s*[:\-]?\s*$")
_AFTER_START = _compile(r"\s*(?:से|se)(?![\u0900-\u097F])")
_AFTER_END = _compile(r"\s*(?:तक|tak)(?![\u0900-\u097F])")


def _date_role(before: str, after: str) -> str:
    if _DEADLINE_WORDS.search(before[-40:]):
        return "deadline"
    if _AFTER_END.match(after) and _APPLY_WORDS.search(before[-40:]):
        return "deadline"
    if _END_TAIL.search(before) or _AFTER_END.match(after):
        return "end"
    if _START_TAIL.search(before) or _AFTER_START.match(after):
        return "start"
    return "mentioned"


def _parse_date_match(kind: str, match: "re.Match[str]") -> Optional[Dict[str, Any]]:
    """Convert one regex match into a date record, or ``None`` if invalid."""
    groups = match.groupdict()
    year = int(groups["y"]) if groups.get("y") else None

    if kind == "financial_year":
        bounds = financial_year_bounds(year, groups["end"])
        if bounds is None:
            return None
        return {"type": "financial_year", "iso": None, "precision": "financial_year", **bounds}

    if kind in ("iso", "numeric"):
        iso = build_iso_date(year, int(groups["m"]), int(groups["d"]))
        return {"type": "date", "iso": iso, "precision": "day"} if iso else None

    month = month_number(groups["month"])
    if month is None:
        return None
    if kind == "month_year":
        iso = build_iso_month(year, month)
        return {"type": "date", "iso": iso, "precision": "month"} if iso else None

    day = int(groups["d"])
    if year is None:  # never invent a missing year
        if not 1 <= day <= 31:
            return None
        return {"type": "date", "iso": None, "precision": "day_month", "day": day, "month": month}
    iso = build_iso_date(year, month, day)
    return {"type": "date", "iso": iso, "precision": "day"} if iso else None


def _build_validity_periods(dates: List[Dict[str, Any]], warnings: List[str]) -> List[Dict[str, Any]]:
    starts = [d for d in dates if d.get("role") == "start" and d.get("precision") == "day"]
    ends = [d for d in dates if d.get("role") == "end" and d.get("precision") == "day"]
    periods = []
    used_ends: List[int] = []
    for start in starts:
        for end_index, end in enumerate(ends):
            if end_index in used_ends or end["span"][0] < start["span"][0]:
                continue
            used_ends.append(end_index)
            if end["iso"] < start["iso"]:
                warnings.append(f"Validity period ignored: end {end['iso']} is before start {start['iso']}")
            else:
                periods.append({"start": start["iso"], "end": end["iso"],
                                "start_text": start["text"], "end_text": end["text"]})
            break
    return periods


def extract_dates(text: str, warnings: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract dates and financial years; return ``(dates, validity_periods)``."""
    taken: List[Tuple[int, int]] = []
    raw: List[Tuple[int, int, str, "re.Match[str]"]] = []
    for kind, pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, taken):
                continue
            record = _parse_date_match(kind, match)
            if record is None:
                if kind != "financial_year":
                    warnings.append(f"Invalid date ignored: '{match.group(0)}'")
                    taken.append(span)  # do not re-read it as a different, partial date
                continue
            taken.append(span)
            raw.append((span[0], span[1], kind, match))
    raw.sort(key=lambda item: item[0])

    dates: List[Dict[str, Any]] = []
    previous_end = 0
    for start, end, kind, match in raw:
        record = _parse_date_match(kind, match)
        before = text[previous_end:start]
        after = text[end:end + 12]
        record.update({
            "text": text[start:end],
            "span": [start, end],
            "role": _date_role(before, after) if record["type"] == "date" else "mentioned",
            "context": _context(text, start, end),
        })
        dates.append(record)
        previous_end = end
    return dates, _build_validity_periods(dates, warnings)


# --------------------------------------------------------------------------
# Age limits
# --------------------------------------------------------------------------

_N = r"(?<![\d,.])(\d{1,3})(?!\d)"
_AGE_UNIT = r"(?:वर्षीय|वर्ष|साल|years?|yrs?|saal|baras)(?![A-Za-z])"
_AGE_KEYWORD = r"(?:(?<![A-Za-z])aged?(?![A-Za-z])|आयु|उम्र|(?<![A-Za-z])(?:umra?|umar)(?![A-Za-z]))"
_JOINER = r"(?:to|and|se|से|-|–|—)"
_MORE_WORDS = r"(?:अधिक|ज्यादा|ज़्यादा|ऊपर|upar|zyada|adhik)"
_LESS_WORDS = r"(?:कम|नीचे|kam|neeche)"

# (name, regex, kind, min_inclusive, max_inclusive, needs_duration_check)
_AGE_PATTERNS: List[Tuple[str, Pattern[str], str, Optional[bool], Optional[bool], bool]] = [
    ("range_with_unit", _compile(rf"{_N}\s*{_JOINER}\s*{_N}\s*{_AGE_UNIT}"), "range", True, True, True),
    ("range_with_keyword", _compile(
        rf"{_AGE_KEYWORD}(?:\s+(?:of|is|between|from|group))?\s*[:\-]?\s*(?:between\s+)?{_N}\s*{_JOINER}\s*{_N}"),
     "range", True, True, False),
    ("min_age_keyword", _compile(
        rf"(?:minimum\s+age(?:\s+(?:of|is|limit(?:\s+is)?|should\s+be))?|min\.?\s+age"
        rf"|न्यूनतम\s+(?:आयु|उम्र)(?:\s+सीमा)?(?:\s+है)?)\s*[:\-]?\s*{_N}"), "min", True, None, False),
    ("max_age_keyword", _compile(
        rf"(?:maximum\s+age(?:\s+(?:of|is|limit(?:\s+is)?|should\s+be))?|max\.?\s+age"
        rf"|अधिकतम\s+(?:आयु|उम्र)(?:\s+सीमा)?(?:\s+है)?)\s*[:\-]?\s*{_N}"), "max", None, True, False),
    ("min_inclusive_unit", _compile(
        rf"(?:at\s+least|not\s+less\s+than|minimum(?:\s+of)?|कम\s+से\s+कम)\s*{_N}\s*{_AGE_UNIT}"), "min", True, None, True),
    ("max_inclusive_unit", _compile(
        rf"(?:up\s*to|not\s+more\s+than|at\s+most|not\s+exceeding|maximum(?:\s+of)?|अधिकतम)\s*{_N}\s*{_AGE_UNIT}"),
     "max", None, True, True),
    ("min_exclusive_age_of", _compile(
        rf"(?:above|over|older\s+than|more\s+than|beyond)\s+the\s+age\s+of\s*{_N}"), "min", False, None, False),
    ("max_exclusive_age_of", _compile(
        rf"(?:below|under|younger\s+than|less\s+than)\s+the\s+age\s+of\s*{_N}"), "max", None, False, False),
    ("min_exclusive_keyword", _compile(
        rf"{_AGE_KEYWORD}\s*(?:of\s+)?(?:above|over|more\s+than)\s*{_N}"), "min", False, None, False),
    ("max_exclusive_keyword", _compile(
        rf"{_AGE_KEYWORD}\s*(?:of\s+)?(?:below|under|less\s+than)\s*{_N}"), "max", None, False, False),
    ("min_exclusive_unit", _compile(
        rf"(?:above|over|older\s+than|more\s+than|greater\s+than)\s*{_N}\s*{_AGE_UNIT}"), "min", False, None, True),
    ("max_exclusive_unit", _compile(
        rf"(?:below|under|younger\s+than|less\s+than|lower\s+than)\s*{_N}\s*{_AGE_UNIT}"), "max", None, False, True),
    ("min_inclusive_after", _compile(
        rf"{_N}\s*(?:{_AGE_UNIT})?\s*(?:\+|or\s+(?:above|older|over|more)|and\s+(?:above|older|over|more|up)"
        rf"|(?:या|और|ya)\s+(?:उससे|इससे|usse|isse)\s+{_MORE_WORDS})"), "min", True, None, True),
    ("max_inclusive_after", _compile(
        rf"{_N}\s*(?:{_AGE_UNIT})?\s*(?:or\s+(?:below|younger|under|less)|and\s+(?:below|younger|under)"
        rf"|(?:या|और|ya)\s+(?:उससे|इससे|usse|isse)\s+{_LESS_WORDS})"), "max", None, True, True),
    ("min_exclusive_after", _compile(
        rf"{_N}\s*{_AGE_UNIT}\s*(?:से|se)\s*(?:{_MORE_WORDS}|more|above|older)"), "min", False, None, True),
    ("max_exclusive_after", _compile(
        rf"{_N}\s*{_AGE_UNIT}\s*(?:से|se)\s*(?:{_LESS_WORDS}|less|below|younger)"), "max", None, False, True),
]
_AGE_UNIT_REGEX = _compile(_AGE_UNIT)
_AGE_KEYWORD_REGEX = _compile(_AGE_KEYWORD)
_MAX_AGE = 120


def extract_age_limits(text: str, rules: ExtractionRules, warnings: List[str]) -> Dict[str, Any]:
    """Extract minimum / maximum age information as structured numbers."""
    taken: List[Tuple[int, int]] = []
    matches: List[Dict[str, Any]] = []

    for name, pattern, kind, min_inc, max_inc, check_duration in _AGE_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, taken):
                continue
            before = text[max(0, span[0] - 30): span[0]]
            has_keyword = bool(_AGE_KEYWORD_REGEX.search(before)) or bool(_AGE_KEYWORD_REGEX.search(match.group(0)))
            if check_duration and not has_keyword and rules.duration_words.search(before):
                continue  # "loan tenure up to 5 years" is a duration, not an age
            if name in ("min_inclusive_after", "max_inclusive_after"):
                if not (_AGE_UNIT_REGEX.search(match.group(0)) or has_keyword):
                    continue  # bare "5000+" must not become an age
            numbers = [int(g) for g in match.groups() if g is not None]
            if any(n > _MAX_AGE for n in numbers):
                warnings.append(f"Age value above {_MAX_AGE} ignored: '{match.group(0)}'")
                continue
            record: Dict[str, Any] = {
                "text": match.group(0), "span": [span[0], span[1]], "kind": kind, "pattern": name,
                "min_age": None, "min_inclusive": None, "max_age": None, "max_inclusive": None,
                "context": _context(text, span[0], span[1]),
            }
            if kind == "range":
                if numbers[0] > numbers[1]:
                    warnings.append(f"Age range ignored (minimum above maximum): '{match.group(0)}'")
                    continue
                record.update(min_age=numbers[0], min_inclusive=min_inc, max_age=numbers[1], max_inclusive=max_inc)
            elif kind == "min":
                record.update(min_age=numbers[0], min_inclusive=min_inc)
            else:
                record.update(max_age=numbers[0], max_inclusive=max_inc)
            taken.append(span)
            matches.append(record)

    matches.sort(key=lambda r: r["span"][0])
    summary: Dict[str, Any] = {
        "min_age": None, "min_inclusive": None, "max_age": None, "max_inclusive": None, "matches": matches,
    }
    for side in ("min", "max"):
        candidates = [m for m in matches if m[f"{side}_age"] is not None]
        if candidates:
            first = candidates[0]
            summary[f"{side}_age"] = first[f"{side}_age"]
            summary[f"{side}_inclusive"] = first[f"{side}_inclusive"]
            if len({c[f"{side}_age"] for c in candidates}) > 1:
                warnings.append(f"Conflicting {side}imum ages found; the first one was used")
    return summary


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------

_CLAUSE_SPLIT = re.compile(r"(?<=[.;।!?])\s+(?!\d)|\n+")
_LAND_AMOUNT = _compile(rf"(?P<num>{_NUM_DIGITS})\s*")


def split_clauses(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` offsets of the clauses of ``text``."""
    spans: List[Tuple[int, int]] = []
    position = 0
    for match in _CLAUSE_SPLIT.finditer(text):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _detect_bound(text: str, start: int, end: int, rules: ExtractionRules) -> Optional[str]:
    """Return ``"upper"`` / ``"lower"`` for a limit value, or ``None``."""
    before = text[max(0, start - 25): start]
    after = text[end: end + 15]
    if rules.upper_before.search(before):
        return "upper"
    if rules.lower_before.search(before):
        return "lower"
    if rules.upper_after.search(after):
        return "upper"
    if rules.lower_after.search(after):
        return "lower"
    return None


def _land_limit(clause: str, offset: int, rules: ExtractionRules, full_text: str) -> Optional[Dict[str, Any]]:
    for unit_name, unit_regex in rules.land_units.items():
        for unit_match in unit_regex.finditer(clause):
            number_matches = list(_LAND_AMOUNT.finditer(clause[: unit_match.start()]))
            if not number_matches or number_matches[-1].end() < unit_match.start() - 3:
                continue
            number = number_matches[-1]
            value = parse_number(number.group("num"))
            if value is None:
                continue
            start, end = offset + number.start(), offset + unit_match.end()
            return {
                "limit_value": float(value) if value != value.to_integral_value() else int(value),
                "limit_unit": unit_name,
                "bound": _detect_bound(full_text, start, end, rules),
            }
    return None


def extract_eligibility(
    masked_text: str,
    display_text: str,
    amounts: List[Dict[str, Any]],
    rules: ExtractionRules,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Extract structured eligibility conditions, clause by clause.

    The original clause is always preserved in ``condition_text``.
    Categories come from ``extraction_rules.json``; the categories named
    ``income`` and ``landholding`` additionally try to read a numeric limit.
    """
    results: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for clause_start, clause_end in split_clauses(masked_text):
        clause = masked_text[clause_start:clause_end]
        condition_text = display_text[clause_start:clause_end].strip()

        for category, values in rules.eligibility.items():
            matched = {name: m.group(0) for name, rx in values.items() if (m := rx.search(clause))}
            record: Dict[str, Any] = {
                "type": "eligibility", "category": category, "values": sorted(matched),
                "matched_terms": [matched[name] for name in sorted(matched)],
                "condition_text": condition_text, "span": [clause_start, clause_end],
                "limit_value": None, "limit_unit": None, "bound": None,
            }
            if category == "landholding":
                land = _land_limit(clause, clause_start, rules, masked_text)
                if land:
                    record.update(land)
                    record["values"] = sorted(set(record["values"]) | {"landholding"})
            if not record["values"]:
                continue
            if category == "income":
                income_amounts = [
                    a for a in amounts
                    if a["role"] == "income_limit" and clause_start <= a["span"][0] < clause_end
                ]
                if income_amounts:
                    first = income_amounts[0]
                    record.update(
                        limit_value=first["amount"], limit_unit=first["currency"],
                        bound=_detect_bound(masked_text, first["span"][0], first["span"][1], rules),
                    )
            key = (category, tuple(record["values"]), record["limit_value"], record["bound"], condition_text)
            results.setdefault(key, record)
    return list(results.values())


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

class EntityExtractor:
    """Runs every extractor on one claim text."""

    def __init__(self, registry: SchemeRegistry, rules: ExtractionRules):
        self.registry = registry
        self.rules = rules

    def extract(self, text: str, entity_hints: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Extract all entities from ``text``.

        ``entity_hints`` are the (validated) NLP entities; they are only used
        to recover scheme aliases and to flag aliases missing from the registry.
        """
        warnings: List[str] = []
        processed = normalize_text(text)

        schemes = self.registry.find(processed)
        unknown: List[Dict[str, str]] = []
        for hint in entity_hints or []:
            if hint["kind"] != "scheme":
                continue
            record = self.registry.lookup_alias(hint["text"])
            if record is None:
                unknown.append({"text": hint["text"], "reason": "alias not found in scheme registry"})
                warnings.append(f"Unknown scheme alias from NLP module: '{hint['text']}'")
            elif all(s["scheme_id"] != record["scheme_id"] for s in schemes):
                schemes.append({
                    "scheme_id": record["scheme_id"], "canonical_name": record["canonical_name"],
                    "matched_text": hint["text"], "matched_alias": normalize_text(hint["text"]),
                    "alias_language": None, "span": None, "all_spans": [], "mentions": 0, "source": "nlp_hint",
                })

        scheme_spans = [tuple(s) for entry in schemes for s in entry["all_spans"]]
        masked = mask_spans(processed, scheme_spans)

        amounts = extract_amounts(masked, self.rules, warnings)
        dates, validity_periods = extract_dates(masked, warnings)
        age_limits = extract_age_limits(masked, self.rules, warnings)
        eligibility = extract_eligibility(masked, processed, amounts, self.rules, warnings)

        return {
            "processed_text": processed,
            "schemes": schemes,
            "unknown_scheme_mentions": unknown,
            "amounts": amounts,
            "dates": dates,
            "validity_periods": validity_periods,
            "age_limits": age_limits,
            "eligibility": eligibility,
            "warnings": warnings,
        }
