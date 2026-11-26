"""Rule-based mutation detection for VERILENS (Member 3).

Compares the structured values extracted from a claim with a trusted
reference record and reports what changed.  Everything is deterministic:
numbers and ISO dates are compared, text meaning is NOT interpreted (that is
the NLI module's job, Member 6).

Safety rule: when two values cannot be compared reliably the result is
``"incomparable"`` with a reason.  A change is never claimed in that case.

Reference record format (see ``data/reference_records.json``)::

    {
      "scheme_id": "...",
      "verified": false,
      "source": {"name": "...", "url": null, "published_on": null},
      "amounts": [{"role": "benefit", "amount": 0, "currency": "INR", "frequency": "per_year"}],
      "dates": {"deadline": "YYYY-MM-DD", "validity_start": "YYYY-MM-DD", "validity_end": "YYYY-MM-DD"},
      "age": {"min_age": 0, "min_inclusive": true, "max_age": 0, "max_inclusive": true},
      "eligibility": [{"category": "occupation", "values": ["farmer"], "limit_value": null, "bound": null}]
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .entity_extraction import ConfigError, load_json_file

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "data" / "reference_records.json"

RESULT_INCREASED = "increased"
RESULT_DECREASED = "decreased"
RESULT_UNCHANGED = "unchanged"
RESULT_CHANGED = "changed"
RESULT_INCOMPARABLE = "incomparable"


class ReferenceStore:
    """Holds reference records indexed by ``scheme_id``."""

    def __init__(self, records: List[Dict[str, Any]], version: str = "unknown"):
        self.version = version
        self._records: Dict[str, Dict[str, Any]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not record.get("scheme_id"):
                raise ConfigError(f"Reference record {index} needs a 'scheme_id'")
            self._records[str(record["scheme_id"])] = record

    @classmethod
    def from_file(cls, path: Path = DEFAULT_REFERENCE_PATH) -> "ReferenceStore":
        data = load_json_file(Path(path))
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ConfigError(f"{path}: expected an object with a 'records' list")
        return cls(data["records"], str(data.get("reference_version", "unknown")))

    @classmethod
    def empty(cls) -> "ReferenceStore":
        return cls([], "none")

    def get(self, scheme_id: str) -> Optional[Dict[str, Any]]:
        return self._records.get(scheme_id)


def _result(attribute: str, result: str, mutation_type: Optional[str], reason: str, **details: Any) -> Dict[str, Any]:
    """Build one comparison record (always the same keys)."""
    record = {"attribute": attribute, "result": result, "mutation_type": mutation_type, "reason": reason}
    record.update(details)
    return record


def _direction(claim_value: float, reference_value: float) -> str:
    return RESULT_INCREASED if claim_value > reference_value else RESULT_DECREASED


# --------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------

def compare_amounts(claim_amounts: List[Dict[str, Any]], reference_amounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare benefit amounts.  Income limits are handled by the eligibility check.

    A claim amount is compared with a reference amount only when both have
    the same frequency (so a yearly total is never compared with a
    per-instalment value).
    """
    results: List[Dict[str, Any]] = []
    candidates = [r for r in reference_amounts if r.get("role", "benefit") == "benefit"]

    for amount in claim_amounts:
        if amount["role"] == "income_limit":
            continue
        base = {
            "claim_value": amount["amount"], "claim_text": amount["text"],
            "frequency": amount["frequency"], "role": amount["role"], "currency": amount["currency"],
        }
        if not candidates:
            results.append(_result("amount", RESULT_INCOMPARABLE, None,
                                   "reference record has no benefit amount", reference_value=None, **base))
            continue

        frequency = amount["frequency"]
        if frequency is None:
            matching_value = [r for r in candidates if r["amount"] == amount["amount"]]
            if matching_value:
                results.append(_result(
                    "amount", RESULT_UNCHANGED, "amount_unchanged",
                    "value equals a reference amount; claim does not state its frequency",
                    reference_value=amount["amount"], confidence="medium", **base))
            else:
                results.append(_result(
                    "amount", RESULT_INCOMPARABLE, None,
                    "claim amount has no frequency and matches no reference amount",
                    reference_value=None, **base))
            continue

        same_frequency = [r for r in candidates if r.get("frequency") == frequency]
        if len(same_frequency) != 1:
            reason = ("reference has no amount with the same frequency" if not same_frequency
                      else "reference has several amounts with the same frequency")
            results.append(_result("amount", RESULT_INCOMPARABLE, None, reason, reference_value=None, **base))
            continue

        reference_value = same_frequency[0]["amount"]
        confidence = "high" if amount["role"] == "benefit" else "medium"
        if amount["amount"] == reference_value:
            results.append(_result("amount", RESULT_UNCHANGED, "amount_unchanged", "amounts are equal",
                                   reference_value=reference_value, confidence=confidence, **base))
            continue
        direction = _direction(amount["amount"], reference_value)
        percent = round((amount["amount"] - reference_value) / reference_value * 100, 2) if reference_value else None
        results.append(_result(
            "amount", direction, f"amount_{direction}", f"claim amount {direction} compared with reference",
            reference_value=reference_value, difference=amount["amount"] - reference_value,
            percent_change=percent, confidence=confidence, **base))
    return results


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

def compare_deadlines(claim_dates: List[Dict[str, Any]], reference_deadline: Optional[str]) -> List[Dict[str, Any]]:
    """Compare claim deadlines (ISO dates) with the reference deadline."""
    results: List[Dict[str, Any]] = []
    for item in (d for d in claim_dates if d.get("role") == "deadline"):
        base = {"claim_value": item.get("iso"), "claim_text": item["text"], "reference_value": reference_deadline}
        if reference_deadline is None:
            results.append(_result("deadline", RESULT_INCOMPARABLE, None, "reference record has no deadline", **base))
        elif item.get("precision") != "day" or not item.get("iso"):
            results.append(_result("deadline", RESULT_INCOMPARABLE, None,
                                   "claim date is not a complete day-month-year date", **base))
        elif item["iso"] == reference_deadline:
            results.append(_result("deadline", RESULT_UNCHANGED, None, "deadlines are equal", **base))
        else:
            later = item["iso"] > reference_deadline
            results.append(_result("deadline", RESULT_CHANGED, "deadline_changed",
                                   "claim deadline is later than reference" if later else "claim deadline is earlier than reference",
                                   direction="later" if later else "earlier", **base))
    return results


def compare_validity(claim_periods: List[Dict[str, Any]], reference_dates: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare claim validity periods with ``validity_start`` / ``validity_end``."""
    results: List[Dict[str, Any]] = []
    ref_start, ref_end = reference_dates.get("validity_start"), reference_dates.get("validity_end")
    for period in claim_periods:
        base = {"claim_value": {"start": period["start"], "end": period["end"]},
                "reference_value": {"start": ref_start, "end": ref_end}}
        if ref_start is None or ref_end is None:
            results.append(_result("validity_period", RESULT_INCOMPARABLE, None,
                                   "reference record has no complete validity period", **base))
            continue
        changed = [name for name, claim, ref in (("start", period["start"], ref_start), ("end", period["end"], ref_end))
                   if claim != ref]
        if changed:
            results.append(_result("validity_period", RESULT_CHANGED, "validity_period_changed",
                                   "validity " + " and ".join(changed) + " differ from reference",
                                   changed_fields=changed, **base))
        else:
            results.append(_result("validity_period", RESULT_UNCHANGED, None, "validity periods are equal", **base))
    return results


# --------------------------------------------------------------------------
# Age limits
# --------------------------------------------------------------------------

def compare_age_limits(claim_age: Dict[str, Any], reference_age: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare minimum and maximum ages (value, then inclusive/exclusive flag)."""
    results: List[Dict[str, Any]] = []
    for side in ("min", "max"):
        claim_value = claim_age.get(f"{side}_age")
        if claim_value is None:
            continue
        claim_inclusive = claim_age.get(f"{side}_inclusive")
        reference_value = reference_age.get(f"{side}_age")
        reference_inclusive = reference_age.get(f"{side}_inclusive")
        attribute = f"{side}_age"
        base = {"claim_value": claim_value, "reference_value": reference_value,
                "claim_inclusive": claim_inclusive, "reference_inclusive": reference_inclusive}
        if reference_value is None:
            results.append(_result(attribute, RESULT_INCOMPARABLE, None, f"reference record has no {side}imum age", **base))
        elif claim_value != reference_value:
            direction = _direction(claim_value, reference_value)
            results.append(_result(attribute, direction, f"{attribute}_changed",
                                   f"claim {side}imum age {direction} compared with reference",
                                   difference=claim_value - reference_value, **base))
        elif None not in (claim_inclusive, reference_inclusive) and claim_inclusive != reference_inclusive:
            results.append(_result(attribute, RESULT_CHANGED, f"{attribute}_changed",
                                   "same age value but the boundary (inclusive/exclusive) differs", **base))
        else:
            results.append(_result(attribute, RESULT_UNCHANGED, None, f"{side}imum ages are equal", **base))
    return results


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------

def compare_eligibility(claim_conditions: List[Dict[str, Any]], reference_conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare eligibility categories and numeric limits.

    A condition is flagged as changed when the claim names a value the
    reference does not contain, or when a numeric limit differs.  A claim that
    merely mentions fewer values than the reference is not treated as a change.
    """
    results: List[Dict[str, Any]] = []
    for condition in claim_conditions:
        category = condition["category"]
        references = [r for r in reference_conditions if r.get("category") == category]
        base = {"category": category, "claim_values": condition["values"], "condition_text": condition["condition_text"]}
        if not references:
            results.append(_result("eligibility", RESULT_INCOMPARABLE, None,
                                   f"reference record has no '{category}' condition", **base))
            continue
        reference = references[0]
        reference_values = set(reference.get("values", []))
        added = sorted(set(condition["values"]) - reference_values)
        reasons: List[str] = []
        details: Dict[str, Any] = {"reference_values": sorted(reference_values), "added_values": added}

        if added:
            reasons.append("claim names values not present in the reference: " + ", ".join(added))

        claim_limit, reference_limit = condition.get("limit_value"), reference.get("limit_value")
        if claim_limit is not None and reference_limit is not None:
            same_bound = condition.get("bound") == reference.get("bound") or None in (condition.get("bound"), reference.get("bound"))
            if not same_bound:
                results.append(_result("eligibility", RESULT_INCOMPARABLE, None,
                                       "limits use different bounds (upper vs lower)", **base, **details))
                continue
            if claim_limit != reference_limit:
                direction = _direction(claim_limit, reference_limit)
                reasons.append(f"limit {direction}: {reference_limit} -> {claim_limit}")
                details.update(claim_limit=claim_limit, reference_limit=reference_limit, limit_direction=direction)

        if reasons:
            results.append(_result("eligibility", RESULT_CHANGED, "eligibility_condition_changed", "; ".join(reasons),
                                   **base, **details))
        else:
            results.append(_result("eligibility", RESULT_UNCHANGED, None, "no difference found", **base, **details))
    return results


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def detect_mutations(entities: Dict[str, Any], reference: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run every comparison of the claim ``entities`` against ``reference``."""
    comparisons: List[Dict[str, Any]] = []
    comparisons += compare_amounts(entities.get("amounts", []), reference.get("amounts", []))
    reference_dates = reference.get("dates") or {}
    comparisons += compare_deadlines(entities.get("dates", []), reference_dates.get("deadline"))
    comparisons += compare_validity(entities.get("validity_periods", []), reference_dates)
    comparisons += compare_age_limits(entities.get("age_limits", {}), reference.get("age") or {})
    comparisons += compare_eligibility(entities.get("eligibility", []), reference.get("eligibility") or [])
    return comparisons


def summarize_mutations(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Condense comparison records into the summary passed downstream."""
    mutation_types = sorted({c["mutation_type"] for c in comparisons
                             if c["mutation_type"] and c["result"] not in (RESULT_UNCHANGED, RESULT_INCOMPARABLE)})
    return {
        "any_change_detected": bool(mutation_types),
        "mutation_types": mutation_types,
        "attributes_unchanged": sorted({c["attribute"] for c in comparisons if c["result"] == RESULT_UNCHANGED}),
        "attributes_incomparable": sorted({c["attribute"] for c in comparisons if c["result"] == RESULT_INCOMPARABLE}),
    }
