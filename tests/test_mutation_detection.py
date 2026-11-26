"""Tests for mutation_detection.py (rule-based comparison)."""

import unittest
from pathlib import Path

from shivansh_work.entity_extraction import ConfigError, EntityExtractor, ExtractionRules, SchemeRegistry
from shivansh_work.mutation_detection import (
    ReferenceStore,
    compare_age_limits,
    compare_amounts,
    compare_deadlines,
    compare_eligibility,
    compare_validity,
    detect_mutations,
    summarize_mutations,
)

EXTRACTOR = EntityExtractor(SchemeRegistry.from_file(), ExtractionRules.from_file())


def amount(value, frequency="per_year", role="benefit"):
    return {"amount": value, "text": str(value), "frequency": frequency, "role": role, "currency": "INR"}


REF_AMOUNTS = [
    {"role": "benefit", "amount": 6000, "frequency": "per_year"},
    {"role": "benefit", "amount": 2000, "frequency": "per_instalment"},
]


class AmountComparisonTests(unittest.TestCase):
    def test_increased(self):
        r = compare_amounts([amount(8000)], REF_AMOUNTS)[0]
        self.assertEqual((r["result"], r["mutation_type"], r["difference"], r["percent_change"]),
                         ("increased", "amount_increased", 2000, 33.33))

    def test_decreased(self):
        r = compare_amounts([amount(5000)], REF_AMOUNTS)[0]
        self.assertEqual((r["result"], r["mutation_type"]), ("decreased", "amount_decreased"))

    def test_unchanged(self):
        r = compare_amounts([amount(6000)], REF_AMOUNTS)[0]
        self.assertEqual((r["result"], r["mutation_type"]), ("unchanged", "amount_unchanged"))

    def test_instalment_is_not_compared_with_yearly_total(self):
        r = compare_amounts([amount(2000, "per_instalment")], REF_AMOUNTS)[0]
        self.assertEqual(r["result"], "unchanged")
        r = compare_amounts([amount(2500, "per_instalment")], REF_AMOUNTS)[0]
        self.assertEqual((r["result"], r["reference_value"]), ("increased", 2000))

    def test_unknown_frequency_without_match_is_incomparable(self):
        r = compare_amounts([amount(9999, None)], REF_AMOUNTS)[0]
        self.assertEqual((r["result"], r["mutation_type"]), ("incomparable", None))

    def test_unknown_frequency_with_matching_value_is_unchanged(self):
        self.assertEqual(compare_amounts([amount(6000, None)], REF_AMOUNTS)[0]["result"], "unchanged")

    def test_frequency_missing_in_reference_is_incomparable(self):
        r = compare_amounts([amount(500, "per_month")], REF_AMOUNTS)[0]
        self.assertEqual(r["result"], "incomparable")

    def test_reference_without_amounts_is_incomparable(self):
        self.assertEqual(compare_amounts([amount(500)], [])[0]["result"], "incomparable")

    def test_income_limits_are_skipped(self):
        self.assertEqual(compare_amounts([amount(200000, None, "income_limit")], REF_AMOUNTS), [])


class DateComparisonTests(unittest.TestCase):
    def deadline(self, iso, precision="day"):
        return {"role": "deadline", "iso": iso, "precision": precision, "text": "x"}

    def test_deadline_changed_later_and_earlier(self):
        later = compare_deadlines([self.deadline("2026-04-30")], "2026-03-31")[0]
        self.assertEqual((later["result"], later["mutation_type"], later["direction"]),
                         ("changed", "deadline_changed", "later"))
        earlier = compare_deadlines([self.deadline("2026-03-01")], "2026-03-31")[0]
        self.assertEqual(earlier["direction"], "earlier")

    def test_deadline_unchanged(self):
        self.assertEqual(compare_deadlines([self.deadline("2026-03-31")], "2026-03-31")[0]["result"], "unchanged")

    def test_partial_date_is_incomparable(self):
        self.assertEqual(compare_deadlines([self.deadline(None, "day_month")], "2026-03-31")[0]["result"],
                         "incomparable")

    def test_missing_reference_deadline_is_incomparable(self):
        self.assertEqual(compare_deadlines([self.deadline("2026-03-31")], None)[0]["result"], "incomparable")

    def test_non_deadline_dates_ignored(self):
        self.assertEqual(compare_deadlines([{"role": "start", "iso": "2026-01-01", "precision": "day", "text": "x"}],
                                           "2026-03-31"), [])

    def test_validity_changed_fields(self):
        r = compare_validity([{"start": "2026-04-01", "end": "2027-06-30"}],
                             {"validity_start": "2026-04-01", "validity_end": "2027-03-31"})[0]
        self.assertEqual((r["result"], r["mutation_type"], r["changed_fields"]),
                         ("changed", "validity_period_changed", ["end"]))

    def test_validity_unchanged_and_incomparable(self):
        same = compare_validity([{"start": "2026-04-01", "end": "2027-03-31"}],
                                {"validity_start": "2026-04-01", "validity_end": "2027-03-31"})[0]
        self.assertEqual(same["result"], "unchanged")
        missing = compare_validity([{"start": "2026-04-01", "end": "2027-03-31"}], {})[0]
        self.assertEqual(missing["result"], "incomparable")


class AgeComparisonTests(unittest.TestCase):
    REF = {"min_age": 18, "min_inclusive": True, "max_age": 40, "max_inclusive": True}

    def test_min_and_max_changed(self):
        results = {r["attribute"]: r for r in compare_age_limits({"min_age": 21, "max_age": 35}, self.REF)}
        self.assertEqual(results["min_age"]["mutation_type"], "min_age_changed")
        self.assertEqual(results["min_age"]["result"], "increased")
        self.assertEqual(results["max_age"]["result"], "decreased")

    def test_unchanged(self):
        results = compare_age_limits({"min_age": 18, "min_inclusive": True, "max_age": 40, "max_inclusive": True}, self.REF)
        self.assertEqual({r["result"] for r in results}, {"unchanged"})

    def test_boundary_difference_is_a_change(self):
        r = compare_age_limits({"max_age": 40, "max_inclusive": False}, self.REF)[0]
        self.assertEqual((r["result"], r["mutation_type"]), ("changed", "max_age_changed"))

    def test_unknown_boundary_is_not_a_change(self):
        self.assertEqual(compare_age_limits({"min_age": 18, "min_inclusive": None}, self.REF)[0]["result"], "unchanged")

    def test_missing_reference_age_is_incomparable(self):
        self.assertEqual(compare_age_limits({"min_age": 18}, {})[0]["result"], "incomparable")

    def test_claim_without_age_gives_nothing(self):
        self.assertEqual(compare_age_limits({"min_age": None, "max_age": None}, self.REF), [])


class EligibilityComparisonTests(unittest.TestCase):
    def cond(self, category, values, limit=None, bound=None):
        return {"category": category, "values": values, "condition_text": "t", "limit_value": limit, "bound": bound}

    def test_income_limit_changed(self):
        ref = [{"category": "income", "values": ["income"], "limit_value": 200000, "bound": "upper"}]
        r = compare_eligibility([self.cond("income", ["income"], 500000, "upper")], ref)[0]
        self.assertEqual((r["result"], r["mutation_type"], r["limit_direction"]),
                         ("changed", "eligibility_condition_changed", "increased"))

    def test_income_limit_unchanged(self):
        ref = [{"category": "income", "values": ["income"], "limit_value": 200000, "bound": "upper"}]
        self.assertEqual(compare_eligibility([self.cond("income", ["income"], 200000, "upper")], ref)[0]["result"],
                         "unchanged")

    def test_different_bounds_are_incomparable(self):
        ref = [{"category": "income", "values": ["income"], "limit_value": 200000, "bound": "upper"}]
        self.assertEqual(compare_eligibility([self.cond("income", ["income"], 500000, "lower")], ref)[0]["result"],
                         "incomparable")

    def test_new_beneficiary_value_is_a_change(self):
        ref = [{"category": "occupation", "values": ["farmer"]}]
        r = compare_eligibility([self.cond("occupation", ["student"])], ref)[0]
        self.assertEqual((r["result"], r["added_values"]), ("changed", ["student"]))

    def test_subset_of_reference_values_is_unchanged(self):
        ref = [{"category": "occupation", "values": ["farmer", "student"]}]
        self.assertEqual(compare_eligibility([self.cond("occupation", ["farmer"])], ref)[0]["result"], "unchanged")

    def test_category_missing_in_reference_is_incomparable(self):
        self.assertEqual(compare_eligibility([self.cond("residence", ["rural"])], [])[0]["result"], "incomparable")


class SummaryAndStoreTests(unittest.TestCase):
    def test_summary(self):
        comparisons = [
            {"attribute": "amount", "result": "increased", "mutation_type": "amount_increased"},
            {"attribute": "deadline", "result": "unchanged", "mutation_type": None},
            {"attribute": "min_age", "result": "incomparable", "mutation_type": None},
        ]
        s = summarize_mutations(comparisons)
        self.assertTrue(s["any_change_detected"])
        self.assertEqual(s["mutation_types"], ["amount_increased"])
        self.assertEqual(s["attributes_unchanged"], ["deadline"])
        self.assertEqual(s["attributes_incomparable"], ["min_age"])

    def test_empty_summary(self):
        self.assertFalse(summarize_mutations([])["any_change_detected"])

    def test_store_lookup_and_errors(self):
        store = ReferenceStore.from_file()
        self.assertIsNotNone(store.get("pm_kisan"))
        self.assertIsNone(store.get("unknown"))
        with self.assertRaises(ConfigError):
            ReferenceStore.from_file(Path("missing.json"))
        with self.assertRaises(ConfigError):
            ReferenceStore([{"no_id": True}])


class DetectOnExtractedTextTests(unittest.TestCase):
    """Extraction + comparison together, using the illustrative reference file."""

    def run_claim(self, text):
        entities = EXTRACTOR.extract(text)
        reference = ReferenceStore.from_file().get(entities["schemes"][0]["scheme_id"])
        return detect_mutations(entities, reference)

    def test_changed_yearly_amount(self):
        results = self.run_claim("PM Kisan gives ₹8,000 per year to farmers")
        self.assertIn("amount_increased", [r["mutation_type"] for r in results])

    def test_age_limit_changed(self):
        results = self.run_claim("Atal Pension Yojana is for people aged 18 to 45 years")
        changed = [r for r in results if r["mutation_type"] == "max_age_changed"]
        self.assertEqual(changed[0]["claim_value"], 45)


if __name__ == "__main__":
    unittest.main()
