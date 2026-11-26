"""Tests for entity_extraction.py.  Sample sentences live only in test files."""

import json
import tempfile
import unittest
from pathlib import Path

from shivansh_work.entity_extraction import (
    ConfigError,
    EntityExtractor,
    ExtractionRules,
    SchemeRegistry,
    split_clauses,
)

REGISTRY = SchemeRegistry.from_file()
RULES = ExtractionRules.from_file()
EXTRACTOR = EntityExtractor(REGISTRY, RULES)


def extract(text, hints=None):
    return EXTRACTOR.extract(text, hints)


def scheme_ids(result):
    return [s["scheme_id"] for s in result["schemes"]]


class SchemeExtractionTests(unittest.TestCase):
    def test_english_alias(self):
        r = extract("PM Kisan provides financial help to farmers.")
        self.assertEqual(scheme_ids(r), ["pm_kisan"])
        self.assertEqual(r["schemes"][0]["canonical_name"], "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)")

    def test_hindi_alias(self):
        r = extract("प्रधानमंत्री आवास योजना के तहत मकान मिलेगा")
        self.assertEqual(scheme_ids(r), ["pm_awas_yojana"])

    def test_hinglish_alias(self):
        r = extract("Ayushman Bharat Yojana ke under ilaaj hota hai")
        self.assertEqual(scheme_ids(r), ["ayushman_bharat_pmjay"])

    def test_case_insensitive_and_separator_tolerant(self):
        self.assertEqual(scheme_ids(extract("pm-kisan")), ["pm_kisan"])
        self.assertEqual(scheme_ids(extract("PMKISAN update")), ["pm_kisan"])

    def test_longest_alias_wins_and_duplicates_removed(self):
        r = extract("PM Kisan Samman Nidhi is also called PM Kisan or PM-KISAN.")
        self.assertEqual(scheme_ids(r), ["pm_kisan"])
        self.assertEqual(r["schemes"][0]["mentions"], 3)

    def test_two_different_schemes(self):
        r = extract("Atal Pension Yojana and Ujjwala Yojana")
        self.assertEqual(sorted(scheme_ids(r)), ["atal_pension_yojana", "pm_ujjwala_yojana"])

    def test_false_positives_prevented(self):
        for text in ("PMKisanxyz is not a scheme", "the pm said hello", "Kisan Credit", "a random sentence"):
            self.assertEqual(scheme_ids(extract(text)), [], text)

    def test_unknown_scheme_hint_reported(self):
        r = extract("Some claim", [{"kind": "scheme", "label": "SCHEME", "text": "Mystery Yojana"}])
        self.assertEqual(r["unknown_scheme_mentions"][0]["text"], "Mystery Yojana")
        self.assertTrue(any("Unknown scheme alias" in w for w in r["warnings"]))

    def test_known_scheme_hint_recovered(self):
        r = extract("claim without the name", [{"kind": "scheme", "label": "SCHEME", "text": "PM Kisan"}])
        self.assertEqual(scheme_ids(r), ["pm_kisan"])
        self.assertEqual(r["schemes"][0]["source"], "nlp_hint")

    def test_registry_errors(self):
        with self.assertRaises(ConfigError):
            SchemeRegistry([{"scheme_id": "x"}])
        with self.assertRaises(ConfigError):
            SchemeRegistry.from_file(Path("does_not_exist.json"))

    def test_new_scheme_needs_only_registry_data(self):
        registry = SchemeRegistry([{"scheme_id": "demo", "canonical_name": "Demo Scheme",
                                    "aliases": {"en": ["Demo Yojana"], "hi": ["डेमो योजना"]}}])
        found = registry.find("the डेमो योजना is new")
        self.assertEqual([s["scheme_id"] for s in found], ["demo"])

    def test_registry_loaded_from_temp_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reg.json"
            path.write_text(json.dumps({"schemes": [{"scheme_id": "a", "canonical_name": "Alpha Scheme",
                                                       "aliases": {"en": ["Alpha"]}}]}), encoding="utf-8")
            self.assertEqual([s["scheme_id"] for s in SchemeRegistry.from_file(path).find("alpha!")], ["a"])


class AmountExtractionTests(unittest.TestCase):
    def amounts(self, text):
        return extract(text)["amounts"]

    def test_symbol_and_word_formats(self):
        for text, expected in [
            ("₹6,000", 6000), ("Rs. 6000", 6000), ("Rs.6000", 6000), ("INR 6,000", 6000),
            ("6000 rupees", 6000), ("रुपये 6000", 6000), ("६००० रुपये", 6000),
        ]:
            found = self.amounts(text)
            self.assertEqual([a["amount"] for a in found], [expected], text)
            self.assertTrue(found[0]["currency_explicit"], text)

    def test_lakh_and_crore(self):
        self.assertEqual(self.amounts("₹1 lakh")[0]["amount"], 100000)
        self.assertEqual(self.amounts("Rs 2.5 lakh")[0]["amount"], 250000)
        self.assertEqual(self.amounts("₹1 crore")[0]["amount"], 10_000_000)
        self.assertEqual(self.amounts("5 लाख रुपये")[0]["amount"], 500000)
        self.assertEqual(self.amounts("do lakh rupaye")[0]["amount"], 200000)

    def test_original_text_is_preserved(self):
        self.assertEqual(self.amounts("benefit of Rs. 2.5 lakh now")[0]["text"], "Rs. 2.5 lakh")

    def test_bare_amount_with_frequency_assumes_inr_with_warning(self):
        r = extract("assistance of 50,000 per year")
        self.assertEqual(r["amounts"][0]["amount"], 50000)
        self.assertEqual(r["amounts"][0]["frequency"], "per_year")
        self.assertFalse(r["amounts"][0]["currency_explicit"])
        self.assertTrue(any("no explicit currency" in w for w in r["warnings"]))

    def test_annual_amount_not_confused_with_instalment(self):
        found = self.amounts("₹6,000 per year in three instalments of ₹2,000 each")
        by_value = {a["amount"]: a for a in found}
        self.assertEqual(by_value[6000]["frequency"], "per_year")
        self.assertEqual(by_value[6000]["annual_equivalent"], 6000)
        self.assertEqual(by_value[2000]["frequency"], "per_instalment")
        self.assertIsNone(by_value[2000]["annual_equivalent"])
        self.assertEqual(by_value[2000]["instalment_count"], 3)

    def test_hindi_frequency_words(self):
        self.assertEqual(self.amounts("सालाना 6000 रुपये")[0]["frequency"], "per_year")
        self.assertEqual(self.amounts("प्रति माह 500 रुपये")[0]["frequency"], "per_month")
        self.assertEqual(self.amounts("₹500 monthly")[0]["annual_equivalent"], 6000)

    def test_roles_income_vs_benefit(self):
        found = self.amounts("Families with income below ₹2 lakh receive ₹6,000 per year")
        by_value = {a["amount"]: a["role"] for a in found}
        self.assertEqual(by_value[200000], "income_limit")
        self.assertEqual(by_value[6000], "benefit")

    def test_head_counts_are_not_money(self):
        self.assertEqual(self.amounts("1 lakh farmers enrolled"), [])
        self.assertEqual(self.amounts("5 लाख किसान"), [])

    def test_duplicates_merged(self):
        found = self.amounts("₹6,000 per year. Again ₹6,000 per year.")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["occurrences"], 2)

    def test_no_amount(self):
        self.assertEqual(self.amounts("nothing numeric here"), [])


class DateExtractionTests(unittest.TestCase):
    def dates(self, text):
        return extract(text)["dates"]

    def test_formats_normalize_to_iso(self):
        for text, iso in [
            ("31 March 2026", "2026-03-31"), ("March 31, 2026", "2026-03-31"), ("31/03/2026", "2026-03-31"),
            ("31-03-2026", "2026-03-31"), ("2026-03-31", "2026-03-31"), ("31 मार्च 2026", "2026-03-31"),
            ("1st April 2026", "2026-04-01"),
        ]:
            found = self.dates(text)
            self.assertEqual([d["iso"] for d in found], [iso], text)
            self.assertEqual(found[0]["precision"], "day", text)

    def test_missing_year_is_not_invented(self):
        d = self.dates("apply by 15 June")[0]
        self.assertIsNone(d["iso"])
        self.assertEqual(d["precision"], "day_month")

    def test_month_only_precision(self):
        d = self.dates("starts March 2026")[0]
        self.assertEqual((d["iso"], d["precision"]), ("2026-03", "month"))

    def test_invalid_date_warns_and_is_dropped(self):
        r = extract("on 30 February 2026")
        self.assertEqual(r["dates"], [])
        self.assertTrue(any("Invalid date" in w for w in r["warnings"]))

    def test_financial_year(self):
        d = self.dates("for FY 2025-26")[0]
        self.assertEqual((d["type"], d["label"], d["start_iso"], d["end_iso"]),
                         ("financial_year", "2025-26", "2025-04-01", "2026-03-31"))

    def test_inconsistent_financial_year_rejected(self):
        self.assertEqual(self.dates("2025-28"), [])

    def test_deadline_role(self):
        self.assertEqual(self.dates("The last date to apply is 31 March 2026")[0]["role"], "deadline")
        self.assertEqual(self.dates("आवेदन की अंतिम तिथि 31 मार्च 2026 है")[0]["role"], "deadline")
        self.assertEqual(self.dates("आवेदन 31 मार्च 2026 तक करें")[0]["role"], "deadline")

    def test_validity_period(self):
        r = extract("Valid from 1 April 2026 to 31 March 2027")
        self.assertEqual([d["role"] for d in r["dates"]], ["start", "end"])
        self.assertEqual(r["validity_periods"][0]["start"], "2026-04-01")
        self.assertEqual(r["validity_periods"][0]["end"], "2027-03-31")

    def test_validity_period_hindi(self):
        r = extract("1 अप्रैल 2026 से 31 मार्च 2027 तक")
        self.assertEqual((r["validity_periods"][0]["start"], r["validity_periods"][0]["end"]),
                         ("2026-04-01", "2027-03-31"))

    def test_reversed_validity_period_warns(self):
        r = extract("valid from 1 April 2027 until 31 March 2026")
        self.assertEqual(r["validity_periods"], [])
        self.assertTrue(any("Validity period ignored" in w for w in r["warnings"]))


class AgeLimitTests(unittest.TestCase):
    def age(self, text):
        a = extract(text)["age_limits"]
        return (a["min_age"], a["min_inclusive"], a["max_age"], a["max_inclusive"])

    def test_ranges(self):
        self.assertEqual(self.age("aged between 18 and 40 years"), (18, True, 40, True))
        self.assertEqual(self.age("age 18-40"), (18, True, 40, True))
        self.assertEqual(self.age("18 से 40 वर्ष की आयु"), (18, True, 40, True))

    def test_minimum_age(self):
        self.assertEqual(self.age("minimum age of 18"), (18, True, None, None))
        self.assertEqual(self.age("at least 21 years"), (21, True, None, None))
        self.assertEqual(self.age("60 years or above"), (60, True, None, None))
        self.assertEqual(self.age("above 60 years"), (60, False, None, None))
        self.assertEqual(self.age("60 saal se upar"), (60, False, None, None))

    def test_maximum_age(self):
        self.assertEqual(self.age("maximum age 45"), (None, None, 45, True))
        self.assertEqual(self.age("up to 40 years"), (None, None, 40, True))
        self.assertEqual(self.age("below 35 years"), (None, None, 35, False))
        self.assertEqual(self.age("40 वर्ष से कम"), (None, None, 40, False))

    def test_both_limits(self):
        self.assertEqual(self.age("above 18 years and below 40 years"), (18, False, 40, False))

    def test_duration_and_money_are_not_ages(self):
        self.assertEqual(self.age("loan tenure up to 5 years"), (None, None, None, None))
        self.assertEqual(self.age("assistance of ₹5000+ available"), (None, None, None, None))
        self.assertEqual(self.age("income below 2 lakh"), (None, None, None, None))

    def test_invalid_range_warns(self):
        r = extract("aged between 40 and 18")
        self.assertIsNone(r["age_limits"]["min_age"])
        self.assertTrue(any("minimum above maximum" in w for w in r["warnings"]))

    def test_conflicting_values_warn(self):
        r = extract("minimum age 18. Another line: minimum age 21")
        self.assertEqual(r["age_limits"]["min_age"], 18)
        self.assertTrue(any("Conflicting" in w for w in r["warnings"]))


class EligibilityTests(unittest.TestCase):
    def conditions(self, text):
        return extract(text)["eligibility"]

    def by_category(self, text):
        return {c["category"]: c for c in self.conditions(text)}

    def test_income_condition_with_limit(self):
        c = self.by_category("Students with family income below ₹2 lakh per year")
        self.assertEqual((c["income"]["limit_value"], c["income"]["bound"]), (200000, "upper"))
        self.assertEqual(c["occupation"]["values"], ["student"])

    def test_hindi_income_condition(self):
        c = self.by_category("जिन परिवारों की आय ₹2 लाख से कम है")
        self.assertEqual((c["income"]["limit_value"], c["income"]["bound"]), (200000, "upper"))

    def test_landholding_limit(self):
        c = self.by_category("Farmers with landholding up to 2 hectares")
        self.assertEqual((c["landholding"]["limit_value"], c["landholding"]["limit_unit"], c["landholding"]["bound"]),
                         (2, "hectare", "upper"))

    def test_beneficiary_and_residence(self):
        c = self.by_category("Open to women and senior citizens living in rural areas")
        self.assertEqual(c["beneficiary_category"]["values"], ["senior_citizen", "women"])
        self.assertEqual(c["residence"]["values"], ["rural"])

    def test_original_condition_text_preserved(self):
        text = "Eligible: women above 18 years"
        self.assertEqual(self.conditions(text)[0]["condition_text"], text)

    def test_scheme_name_is_not_an_eligibility_condition(self):
        self.assertEqual(self.conditions("PM Kisan Samman Nidhi update"), [])

    def test_no_invented_rules(self):
        self.assertEqual(self.conditions("The weather is nice"), [])

    def test_clause_splitting_keeps_rupee_abbreviation_together(self):
        text = "Pays Rs. 6000 yearly. Second clause here"
        spans = split_clauses(text)
        self.assertEqual(len(spans), 2)
        self.assertIn("Rs. 6000", text[spans[0][0]:spans[0][1]])


class CombinedExtractionTests(unittest.TestCase):
    def test_empty_text_returns_empty_structure(self):
        r = extract("")
        self.assertEqual((r["schemes"], r["amounts"], r["dates"], r["eligibility"]), ([], [], [], []))

    def test_extraction_is_json_serializable(self):
        r = extract("PM Kisan: ₹6,000 per year, apply by 31 March 2026, aged 18-40, farmers with income below ₹2 lakh")
        json.dumps(r, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
