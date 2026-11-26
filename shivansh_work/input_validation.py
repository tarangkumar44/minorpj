"""Validation of records coming from the NLP module (Member 2).

The real NLP output contract may differ from the *proposed* one, so every
field name is configurable through ``field_map`` (dot notation is supported
for nested fields, e.g. ``"claim.text"``).

Proposed input contract (all fields except the text are optional)::

    {
      "claim_id": "string",
      "text": "original claim text",
      "normalized_text": "text after NLP preprocessing",
      "language": "en" | "hi" | "hinglish",
      "entities": [{"type": "SCHEME", "text": "..."}],
      "metadata": {"any": "extra information to preserve"}
    }

Nothing here raises for bad input; problems are returned as errors
(record cannot be processed) or warnings (record processed with caveats).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SUPPORTED_LANGUAGES = ("en", "hi", "hinglish")

DEFAULT_FIELD_MAP: Dict[str, str] = {
    "claim_id": "claim_id",
    "text": "text",
    "normalized_text": "normalized_text",
    "language": "language",
    "entities": "entities",
    "metadata": "metadata",
    "entity_type": "type",
    "entity_text": "text",
}

#: NLP entity type labels -> internal entity kinds.  Labels that are not
#: listed here are reported as "unsupported entity type" warnings.
DEFAULT_ENTITY_TYPE_ALIASES: Dict[str, str] = {
    "SCHEME": "scheme", "SCHEME_NAME": "scheme", "PROGRAMME": "scheme", "PROGRAM": "scheme",
    "AMOUNT": "amount", "MONEY": "amount",
    "DATE": "date", "DEADLINE": "date",
    "AGE": "age",
    "ELIGIBILITY": "eligibility",
}

#: Which text field to run the extractors on when both are present.
TEXT_PREFERENCES = ("text", "normalized_text")


@dataclass
class ValidatedClaim:
    """A claim record that passed validation and is ready for extraction."""

    claim_id: str
    text: str
    text_field_used: str
    language: Optional[str]
    entity_hints: List[Dict[str, str]]
    metadata: Dict[str, Any]
    source_record: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Outcome of validating one record."""

    claim: Optional[ValidatedClaim]
    claim_id: Optional[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_json_payload(raw: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Parse a JSON string/bytes.  Returns ``(payload, error_message)``."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, f"Input is not valid UTF-8: {exc}"
    if not isinstance(raw, str) or not raw.strip():
        return None, "Input JSON is empty or not a string"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"


def extract_records(payload: Any) -> Tuple[List[Any], Optional[str]]:
    """Return the list of claim records contained in a parsed payload.

    Accepted shapes: a list of records, ``{"claims": [...]}`` or one record.
    """
    if payload is None:
        return [], "Input payload is null"
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        if "claims" in payload:
            claims = payload["claims"]
            if not isinstance(claims, list):
                return [], "'claims' must be a list"
            return claims, None
        return [payload], None
    return [], f"Unexpected payload type: {type(payload).__name__}"


def _get_path(record: Dict[str, Any], path: str) -> Any:
    """Read ``record[a][b]`` for ``path == 'a.b'``; ``None`` when missing."""
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_entity_hints(
    raw_entities: Any,
    field_map: Dict[str, str],
    type_aliases: Dict[str, str],
    warnings: List[str],
) -> List[Dict[str, str]]:
    """Convert NLP-provided entities into ``{"kind", "label", "text"}`` hints."""
    if raw_entities is None:
        return []
    if not isinstance(raw_entities, list):
        warnings.append("Field 'entities' is not a list; NLP entity hints ignored")
        return []
    hints: List[Dict[str, str]] = []
    seen = set()
    for index, item in enumerate(raw_entities):
        if not isinstance(item, dict):
            warnings.append(f"entities[{index}] is not an object; ignored")
            continue
        label = _clean_string(_get_path(item, field_map["entity_type"]))
        text = _clean_string(_get_path(item, field_map["entity_text"]))
        if label is None or text is None:
            warnings.append(f"entities[{index}] is missing its type or text; ignored")
            continue
        kind = type_aliases.get(label.upper())
        if kind is None:
            warnings.append(f"Unsupported entity type '{label}' ignored")
            continue
        key = (kind, text.casefold())
        if key in seen:
            warnings.append(f"Duplicate NLP entity '{text}' ({label}) ignored")
            continue
        seen.add(key)
        hints.append({"kind": kind, "label": label, "text": text})
    return hints


def validate_claim_record(
    record: Any,
    index: int = 0,
    field_map: Optional[Dict[str, str]] = None,
    type_aliases: Optional[Dict[str, str]] = None,
    text_preference: str = "text",
) -> ValidationResult:
    """Validate one NLP record and convert it to a :class:`ValidatedClaim`."""
    mapping = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    aliases = type_aliases if type_aliases is not None else DEFAULT_ENTITY_TYPE_ALIASES
    if text_preference not in TEXT_PREFERENCES:
        text_preference = "text"

    if not isinstance(record, dict):
        return ValidationResult(None, None, errors=[f"Record {index} is not a JSON object"])

    warnings: List[str] = []
    raw_id = _get_path(record, mapping["claim_id"])
    claim_id = str(raw_id).strip() if raw_id not in (None, "") else None

    original = _clean_string(_get_path(record, mapping["text"]))
    normalized = _clean_string(_get_path(record, mapping["normalized_text"]))
    if original is None and normalized is None:
        return ValidationResult(
            None, claim_id, errors=["Claim text is missing or empty"], warnings=warnings
        )

    if text_preference == "text":
        text, used = (original, "text") if original else (normalized, "normalized_text")
    else:
        text, used = (normalized, "normalized_text") if normalized else (original, "text")
    if used != text_preference:
        warnings.append(f"Preferred field '{text_preference}' missing; used '{used}'")

    if claim_id is None:
        claim_id = "auto-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        warnings.append(f"claim_id missing; generated '{claim_id}'")

    language = _clean_string(_get_path(record, mapping["language"]))
    if language is not None:
        language = language.lower()
        if language not in SUPPORTED_LANGUAGES:
            warnings.append(f"Language '{language}' is not one of {SUPPORTED_LANGUAGES}")
    else:
        warnings.append("Language not provided by NLP module")

    metadata = _get_path(record, mapping["metadata"])
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        warnings.append("Field 'metadata' is not an object; kept under key 'value'")
        metadata = {"value": metadata}

    hints = _parse_entity_hints(
        _get_path(record, mapping["entities"]), mapping, aliases, warnings
    )

    claim = ValidatedClaim(
        claim_id=claim_id,
        text=text,
        text_field_used=used,
        language=language,
        entity_hints=hints,
        metadata=metadata,
        source_record=record,
        warnings=list(warnings),
    )
    return ValidationResult(claim, claim_id, errors=[], warnings=warnings)
