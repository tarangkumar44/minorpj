"""Tests for input_validation.py."""

import unittest

from shivansh_work.input_validation import (
    extract_records,
    parse_json_payload,
    validate_claim_record,
)


class ParsingTests(unittest.TestCase):
    def test_invalid_json(self):
        payload, error = parse_json_payload("{not json")
        self.assertIsNone(payload)
        self.assertIn("Invalid JSON", error)

    def test_empty_and_non_string_input(self):
        self.assertIsNotNone(parse_json_payload("   ")[1])
        self.assertIsNotNone(parse_json_payload(None)[1])

    def test_bytes_input(self):
        payload, error = parse_json_payload('{"a": "है"}'.encode("utf-8"))
        self.assertIsNone(error)
        self.assertEqual(payload, {"a": "है"})

    def test_record_shapes(self):
        self.assertEqual(extract_records([{"a": 1}])[0], [{"a": 1}])
        self.assertEqual(extract_records({"claims": [{"a": 1}]})[0], [{"a": 1}])
        self.assertEqual(extract_records({"text": "x"})[0], [{"text": "x"}])
        self.assertIsNotNone(extract_records({"claims": "nope"})[1])
        self.assertIsNotNone(extract_records(None)[1])
        self.assertIsNotNone(extract_records(42)[1])


class RecordValidationTests(unittest.TestCase):
    def test_valid_record(self):
        r = validate_claim_record({"claim_id": "c1", "text": "abc", "language": "EN"})
        self.assertEqual((r.claim.claim_id, r.claim.text, r.claim.language, r.errors), ("c1", "abc", "en", []))

    def test_non_object_record(self):
        r = validate_claim_record("just a string")
        self.assertIsNone(r.claim)
        self.assertTrue(r.errors)

    def test_missing_null_and_blank_text(self):
        for record in ({}, {"text": None}, {"text": "   "}, {"claim_id": "c9"}):
            r = validate_claim_record(record)
            self.assertIsNone(r.claim, record)
            self.assertIn("Claim text is missing or empty", r.errors)

    def test_missing_claim_id_is_generated_deterministically(self):
        a = validate_claim_record({"text": "same text"}).claim.claim_id
        b = validate_claim_record({"text": "same text"}).claim.claim_id
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("auto-"))

    def test_text_preference_and_fallback(self):
        record = {"text": "orig", "normalized_text": "norm"}
        self.assertEqual(validate_claim_record(record).claim.text, "orig")
        self.assertEqual(validate_claim_record(record, text_preference="normalized_text").claim.text, "norm")
        fallback = validate_claim_record({"normalized_text": "norm"})
        self.assertEqual((fallback.claim.text, fallback.claim.text_field_used), ("norm", "normalized_text"))
        self.assertTrue(any("missing" in w for w in fallback.warnings))

    def test_unsupported_language_warns(self):
        r = validate_claim_record({"text": "x", "language": "fr"})
        self.assertTrue(any("Language 'fr'" in w for w in r.warnings))

    def test_custom_field_map_with_nested_path(self):
        record = {"id": 7, "claim": {"body": "hello"}, "lang": "hi"}
        mapping = {"claim_id": "id", "text": "claim.body", "language": "lang"}
        r = validate_claim_record(record, field_map=mapping)
        self.assertEqual((r.claim.claim_id, r.claim.text, r.claim.language), ("7", "hello", "hi"))

    def test_entity_hints_filtering(self):
        record = {"text": "x", "entities": [
            {"type": "SCHEME", "text": "A Scheme"},
            {"type": "scheme", "text": "a scheme"},        # duplicate
            {"type": "ORG", "text": "Ministry"},           # unsupported
            {"type": "MONEY", "text": None},               # missing text
            "not an object",
        ]}
        r = validate_claim_record(record)
        self.assertEqual([h["text"] for h in r.claim.entity_hints], ["A Scheme"])
        joined = " | ".join(r.warnings)
        for fragment in ("Duplicate NLP entity", "Unsupported entity type 'ORG'", "missing its type or text", "not an object"):
            self.assertIn(fragment, joined)

    def test_entities_not_a_list(self):
        r = validate_claim_record({"text": "x", "entities": "oops"})
        self.assertEqual(r.claim.entity_hints, [])
        self.assertTrue(any("not a list" in w for w in r.warnings))

    def test_metadata_handling(self):
        self.assertEqual(validate_claim_record({"text": "x", "metadata": {"k": 1}}).claim.metadata, {"k": 1})
        self.assertEqual(validate_claim_record({"text": "x", "metadata": "str"}).claim.metadata, {"value": "str"})


if __name__ == "__main__":
    unittest.main()
