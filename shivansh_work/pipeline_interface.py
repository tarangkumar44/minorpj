"""Integration layer: NLP module (Member 2) -> this module -> next module.

Typical use from Python::

    from shivansh_work.pipeline_interface import EntityMutationPipeline
    pipeline = EntityMutationPipeline()            # loads data/*.json
    output = pipeline.process_claim(nlp_record)    # one dict per claim

Command line (from the project root)::

    python -m shivansh_work.pipeline_interface --input nlp_output.json --output result.json

Output schema (``schema_version`` "1.0"), every key is always present::

    claim_id, status ("ok" | "ok_with_warnings" | "invalid_input" | "error"),
    language, text_used, processed_text,
    schemes[]            - canonical scheme matches (scheme_id, canonical_name, matched_text, span ...)
    scheme               - the single identified scheme or null
    unknown_scheme_mentions[]
    entities             - {"amounts": [], "dates": [], "validity_periods": [],
                            "age_limits": {}, "eligibility": []}
    reference_comparison - {"performed": bool, "reason": str|null, "scheme_id", "reference_source",
                            "reference_verified"}
    mutations[]          - one record per compared attribute (see mutation_detection.py)
    mutation_summary     - {"any_change_detected", "mutation_types", "attributes_unchanged",
                            "attributes_incomparable"}
    warnings[], errors[]
    nlp_input            - the untouched NLP record (nothing is discarded)
    provenance           - module name, versions, field used, processing time
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .entity_extraction import (
    DEFAULT_REGISTRY_PATH,
    DEFAULT_RULES_PATH,
    ConfigError,
    EntityExtractor,
    ExtractionRules,
    SchemeRegistry,
)
from .input_validation import (
    DEFAULT_ENTITY_TYPE_ALIASES,
    DEFAULT_FIELD_MAP,
    extract_records,
    parse_json_payload,
    validate_claim_record,
)
from .mutation_detection import (
    DEFAULT_REFERENCE_PATH,
    ReferenceStore,
    detect_mutations,
    summarize_mutations,
)

SCHEMA_VERSION = "1.0"
MODULE_NAME = "verilens.entity_extraction_rule_mutation"


@dataclass
class PipelineConfig:
    """All configurable parts of the integration (paths and field names)."""

    registry_path: Path = DEFAULT_REGISTRY_PATH
    rules_path: Path = DEFAULT_RULES_PATH
    reference_path: Path = DEFAULT_REFERENCE_PATH
    field_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FIELD_MAP))
    entity_type_aliases: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ENTITY_TYPE_ALIASES))
    text_preference: str = "text"  # "text" or "normalized_text"


def _empty_entities() -> Dict[str, Any]:
    return {"amounts": [], "dates": [], "validity_periods": [], "age_limits": {}, "eligibility": []}


class EntityMutationPipeline:
    """Validates NLP records, extracts entities and detects rule-based mutations."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.registry = SchemeRegistry.from_file(self.config.registry_path)  # required: raises ConfigError
        self.rules = ExtractionRules.from_file(self.config.rules_path)       # required: raises ConfigError
        self.extractor = EntityExtractor(self.registry, self.rules)
        self.startup_warnings: List[str] = list(self.registry.warnings)
        try:
            self.references = ReferenceStore.from_file(self.config.reference_path)
        except ConfigError as exc:  # comparison is optional; extraction still works
            self.references = ReferenceStore.empty()
            self.startup_warnings.append(f"Reference records unavailable: {exc}")

    # ------------------------------------------------------------------
    def _base_output(self, claim_id: Optional[str], record: Any) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "claim_id": claim_id,
            "status": "ok",
            "language": None,
            "text_used": None,
            "processed_text": None,
            "schemes": [],
            "scheme": None,
            "unknown_scheme_mentions": [],
            "entities": _empty_entities(),
            "reference_comparison": {"performed": False, "reason": None, "scheme_id": None,
                                     "reference_source": None, "reference_verified": None},
            "mutations": [],
            "mutation_summary": summarize_mutations([]),
            "warnings": [],
            "errors": [],
            "nlp_input": copy.deepcopy(record),
            "provenance": {
                "module": MODULE_NAME, "member": 3, "schema_version": SCHEMA_VERSION,
                "registry_version": self.registry.version, "rules_version": self.rules.version,
                "reference_version": self.references.version, "text_field_used": None,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _compare_with_reference(self, output: Dict[str, Any]) -> None:
        """Fill ``reference_comparison``, ``mutations`` and ``mutation_summary``."""
        comparison = output["reference_comparison"]
        schemes = output["schemes"]
        if not schemes:
            comparison["reason"] = "no scheme identified in the claim"
            return
        if len(schemes) > 1:
            comparison["reason"] = "several schemes identified; values cannot be attributed to one scheme"
            output["warnings"].append("Reference comparison skipped: " + comparison["reason"])
            return
        scheme_id = schemes[0]["scheme_id"]
        comparison["scheme_id"] = scheme_id
        reference = self.references.get(scheme_id)
        if reference is None:
            comparison["reason"] = f"no reference record for scheme '{scheme_id}'"
            output["warnings"].append("Reference comparison skipped: " + comparison["reason"])
            return
        source = reference.get("source") or {}
        comparison.update(performed=True, reference_source=source.get("name"),
                          reference_verified=bool(reference.get("verified", False)))
        if not comparison["reference_verified"]:
            output["warnings"].append("Reference record is not marked as verified against an official source")
        output["mutations"] = detect_mutations(output["entities"], reference)
        output["mutation_summary"] = summarize_mutations(output["mutations"])

    # ------------------------------------------------------------------
    def process_claim(self, record: Any, index: int = 0) -> Dict[str, Any]:
        """Process one NLP record; never raises for bad input."""
        validation = validate_claim_record(
            record, index, self.config.field_map, self.config.entity_type_aliases, self.config.text_preference)
        output = self._base_output(validation.claim_id, record)
        output["warnings"].extend(self.startup_warnings)

        if validation.claim is None:
            output["status"] = "invalid_input"
            output["errors"].extend(validation.errors)
            output["warnings"].extend(validation.warnings)
            return output

        claim = validation.claim
        try:
            extracted = self.extractor.extract(claim.text, claim.entity_hints)
        except Exception as exc:  # defensive: one bad claim must not stop a batch
            output["status"] = "error"
            output["errors"].append(f"Extraction failed: {type(exc).__name__}: {exc}")
            return output

        output.update(
            claim_id=claim.claim_id, language=claim.language, text_used=claim.text,
            processed_text=extracted["processed_text"], schemes=extracted["schemes"],
            scheme=extracted["schemes"][0] if len(extracted["schemes"]) == 1 else None,
            unknown_scheme_mentions=extracted["unknown_scheme_mentions"],
        )
        output["entities"] = {key: extracted[key] for key in _empty_entities()}
        output["provenance"]["text_field_used"] = claim.text_field_used
        output["provenance"]["input_metadata"] = claim.metadata
        output["warnings"].extend(claim.warnings)
        output["warnings"].extend(extracted["warnings"])

        if not output["schemes"]:
            output["warnings"].append("No scheme identified in the claim")
        self._compare_with_reference(output)
        output["status"] = "ok_with_warnings" if output["warnings"] else "ok"
        return output

    def process_batch(self, records: List[Any]) -> List[Dict[str, Any]]:
        """Process a list of NLP records, one output per record, same order."""
        return [self.process_claim(record, index) for index, record in enumerate(records)]

    def process_json(self, raw: Any) -> List[Dict[str, Any]]:
        """Process a JSON string/bytes (one record, a list, or ``{"claims": [...]}``)."""
        payload, error = parse_json_payload(raw)
        if error is None:
            records, error = extract_records(payload)
        if error is not None:
            output = self._base_output(None, None)
            output["status"] = "invalid_input"
            output["errors"].append(error)
            return [output]
        return self.process_batch(records)

    def process_file(self, path: Path) -> List[Dict[str, Any]]:
        """Process a JSON file containing NLP output."""
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            output = self._base_output(None, None)
            output["status"] = "invalid_input"
            output["errors"].append(f"Cannot read input file: {exc}")
            return [output]
        return self.process_json(raw)


def to_json(outputs: Any, indent: Optional[int] = 2) -> str:
    """Serialize pipeline output (Hindi text is kept readable)."""
    return json.dumps(outputs, ensure_ascii=False, indent=indent)


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point.  Returns a process exit code."""
    parser = argparse.ArgumentParser(description="VERILENS entity extraction + rule-based mutation detection")
    parser.add_argument("--input", help="JSON file with NLP output (default: read stdin)")
    parser.add_argument("--output", help="write result JSON here (default: print)")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--field-map", help="JSON file overriding NLP field names")
    parser.add_argument("--text-preference", choices=["text", "normalized_text"], default="text")
    args = parser.parse_args(argv)

    config = PipelineConfig(
        registry_path=Path(args.registry), rules_path=Path(args.rules),
        reference_path=Path(args.reference), text_preference=args.text_preference,
    )
    try:
        if args.field_map:
            config.field_map.update(json.loads(Path(args.field_map).read_text(encoding="utf-8")))
        pipeline = EntityMutationPipeline(config)
    except (ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    outputs = pipeline.process_file(Path(args.input)) if args.input else pipeline.process_json(sys.stdin.buffer.read())
    text = to_json(outputs)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if all(o["status"] in ("ok", "ok_with_warnings") for o in outputs) else 1


if __name__ == "__main__":
    sys.exit(main())
