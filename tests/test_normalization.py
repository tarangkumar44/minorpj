"""Tests for normalization.py (sample values are allowed here, not in production code)."""

import unittest

from shivansh_work.normalization import (
    annual_equivalent,
    build_iso_date,
    financial_year_bounds,
    month_number,
    normalize_text,
    parse_number,
    to_amount_value,
)


class NormalizeTextTests(unittest.TestCase):
    def test_devanagari_digits_become_ascii(self):
        self.assertEqual(normalize_text("₹६,०००"), "₹6,000")

    def test_spaces_collapse_and_zero_width_removed(self):
        self.assertEqual(normalize_text("a \u00a0  b\u200bc"), "a bc")

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            normalize_text(None)


class AmountTests(unittest.TestCase):
    def test_plain_and_grouped_numbers(self):
        self.assertEqual(to_amount_value("6000"), 6000)
        self.assertEqual(to_amount_value("6,000"), 6000)
        self.assertEqual(to_amount_value("1,00,000"), 100000)

    def test_magnitude_words(self):
        self.assertEqual(to_amount_value("2.5", "lakh"), 250000)
        self.assertEqual(to_amount_value("1", "crore"), 10_000_000)
        self.assertEqual(to_amount_value("2", "लाख"), 200000)
        self.assertEqual(to_amount_value("do", "lakh"), 200000)

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(to_amount_value("abc"))
        self.assertIsNone(to_amount_value("5", "bogus"))
        self.assertIsNone(parse_number(None))

    def test_annual_equivalent_only_when_safe(self):
        self.assertEqual(annual_equivalent(500, "per_month"), 6000)
        self.assertEqual(annual_equivalent(6000, "per_year"), 6000)
        self.assertIsNone(annual_equivalent(2000, "per_instalment"))
        self.assertIsNone(annual_equivalent(2000, None))


class DateTests(unittest.TestCase):
    def test_iso_building_validates_calendar(self):
        self.assertEqual(build_iso_date(2026, 3, 31), "2026-03-31")
        self.assertIsNone(build_iso_date(2026, 2, 30))
        self.assertIsNone(build_iso_date(2026, 13, 1))

    def test_month_names_english_and_hindi(self):
        self.assertEqual(month_number("March"), 3)
        self.assertEqual(month_number("sept"), 9)
        self.assertEqual(month_number("मार्च"), 3)
        self.assertIsNone(month_number("notamonth"))

    def test_financial_year_bounds(self):
        self.assertEqual(financial_year_bounds(2025, "26"),
                         {"label": "2025-26", "start_iso": "2025-04-01", "end_iso": "2026-03-31"})
        self.assertEqual(financial_year_bounds(2025, "2026")["end_iso"], "2026-03-31")
        self.assertIsNone(financial_year_bounds(2025, "28"))
        self.assertIsNone(financial_year_bounds(2025, "2028"))


if __name__ == "__main__":
    unittest.main()
