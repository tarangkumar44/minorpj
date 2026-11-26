"""Integration tests: representative NLP output -> pipeline -> final structured output."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from shivansh_work.pipeline_interface import (
    SCHEMA_VERSION,
    EntityMutationPipeline,
    PipelineConfig,
    main,
    to_json,
)

REQUIRED_KEYS = {
    "schema_version", "claim_id", "status", "language", "text_used", "processed_text", "schemes", "scheme",
    "unknown_scheme_mentions", "entities", "reference_comparison", "mutations", "mutation_summary",
    "warnings", "errors", "nlp_input", "provenance",
}

# Representative output of the NLP module (proposed contract), used only in tests.
NLP_RECORD = {
    "claim_id": "claim-001",
    "text": "PM Kisan ke tahat kisanon ko ₹8,000 per year milenge, apply by 31 March 2026.",
    "normalized_text": "pm kisan ke tahat kisanon ko 8000 per year milenge apply by 31 march 2026",
    "language": "hinglish",
    "entities": [{"type": "SCHEME", "text": "PM Kisan"}, {"type": "ORG", "text": "Ministry"}],
    "metadata": {"source": "unit-test", "nlp_version": "x"},
}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = EntityMutationPipeline()

    def test_end_to_end_output(self):
        out = self.pipeline.process_claim(NLP_RECORD)
        self.assertEqual(set(out), REQUIRED_KEYS)
        self.assertEqual(out["schema_version"], SCHEMA_VERSION)
        self.assertEqual(out["claim_id"], "claim-001")
        self.assertEqual(out["language"], "hinglish")
        self.assertEqual(out["scheme"]["scheme_id"], "pm_kisan")
        self.assertEqual(out["scheme"]["canonical_name"], "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)")
        amounts = out["entities"]["amounts"]
        self.assertEqual((amounts[0]["amount"], amounts[0]["frequency"]), (8000, "per_year"))
        deadlines = [d for d in out["entities"]["dates"] if d["role"] == "deadline"]
        self.assertEqual(deadlines[0]["iso"], "2026-03-31")
        self.assertTrue(out["reference_comparison"]["performed"])
        self.assertEqual(out["mutation_summary"]["mutation_types"], ["amount_increased"])
        self.assertTrue(out["mutation_summary"]["any_change_detected"])
        # deadline is incomparable because the illustrative reference has none
        self.assertIn("deadline", out["mutation_summary"]["attributes_incomparable"])
        self.assertEqual(out["provenance"]["text_field_used"], "text")
        self.assertEqual(out["provenance"]["input_metadata"], {"source": "unit-test", "nlp_version": "x"})

    def test_nlp_input_is_preserved_untouched(self):
        original = copy.deepcopy(NLP_RECORD)
        out = self.pipeline.process_claim(NLP_RECORD)
        self.assertEqual(out["nlp_input"], original)
        self.assertEqual(NLP_RECORD, original)  # input not mutated

    def test_unsupported_entity_type_warns_but_processes(self):
        out = self.pipeline.process_claim(NLP_RECORD)
        self.assertTrue(any("Unsupported entity type 'ORG'" in w for w in out["warnings"]))
        self.assertEqual(out["status"], "ok_with_warnings")

    def test_output_is_json_serializable_and_hindi_readable(self):
        out = self.pipeline.process_claim({"claim_id": "h1", "text": "पीएम किसान के तहत सालाना ₹6000", "language": "hi"})
        text = to_json(out)
        self.assertIn("पीएम किसान", text)
        self.assertEqual(json.loads(text)["claim_id"], "h1")

    def test_unchanged_claim_has_no_mutation(self):
        out = self.pipeline.process_claim({"claim_id": "u1", "text": "PM Kisan pays ₹6,000 per year", "language": "en"})
        self.assertFalse(out["mutation_summary"]["any_change_detected"])
        self.assertIn("amount", out["mutation_summary"]["attributes_unchanged"])

    def test_empty_claim(self):
        out = self.pipeline.process_claim({"claim_id": "e1", "text": ""})
        self.assertEqual(out["status"], "invalid_input")
        self.assertEqual(out["claim_id"], "e1")
        self.assertTrue(out["errors"])
        self.assertEqual(set(out), REQUIRED_KEYS)

    def test_null_and_non_dict_records(self):
        for record in (None, "text", 5, [1, 2]):
            out = self.pipeline.process_claim(record)
            self.assertEqual(out["status"], "invalid_input", record)

    def test_unknown_scheme_skips_comparison_with_reason(self):
        out = self.pipeline.process_claim({"claim_id": "k1", "text": "Some Unregistered Yojana gives ₹100 per year",
                                           "language": "en",
                                           "entities": [{"type": "SCHEME", "text": "Some Unregistered Yojana"}]})
        self.assertEqual(out["unknown_scheme_mentions"][0]["text"], "Some Unregistered Yojana")
        self.assertFalse(out["reference_comparison"]["performed"])
        self.assertEqual(out["reference_comparison"]["reason"], "no scheme identified in the claim")
        self.assertEqual(out["mutations"], [])

    def test_scheme_without_reference_record(self):
        out = self.pipeline.process_claim({"claim_id": "r1", "text": "Ujjwala Yojana gives ₹100 once", "language": "en"})
        self.assertFalse(out["reference_comparison"]["performed"])
        self.assertIn("no reference record", out["reference_comparison"]["reason"])

    def test_several_schemes_are_not_compared(self):
        out = self.pipeline.process_claim({"claim_id": "m1", "language": "en",
                                           "text": "PM Kisan and Atal Pension Yojana give ₹6,000 per year"})
        self.assertFalse(out["reference_comparison"]["performed"])
        self.assertIsNone(out["scheme"])
        self.assertEqual(len(out["schemes"]), 2)

    def test_normalized_text_preference(self):
        config = PipelineConfig(text_preference="normalized_text")
        out = EntityMutationPipeline(config).process_claim(NLP_RECORD)
        self.assertEqual(out["provenance"]["text_field_used"], "normalized_text")

    def test_custom_field_map(self):
        config = PipelineConfig(field_map={"claim_id": "id", "text": "payload.claim"})
        out = EntityMutationPipeline(config).process_claim({"id": "z", "payload": {"claim": "PM Kisan ₹6,000 per year"}})
        self.assertEqual((out["claim_id"], out["scheme"]["scheme_id"]), ("z", "pm_kisan"))

    def test_batch_keeps_order_and_isolates_bad_records(self):
        outs = self.pipeline.process_batch([NLP_RECORD, None, {"claim_id": "b3", "text": "PM Kisan"}])
        self.assertEqual([o["status"] for o in outs], ["ok_with_warnings", "invalid_input", "ok_with_warnings"])
        self.assertEqual(outs[2]["claim_id"], "b3")

    def test_process_json_variants(self):
        wrapped = json.dumps({"claims": [NLP_RECORD, NLP_RECORD]})
        self.assertEqual(len(self.pipeline.process_json(wrapped)), 2)
        self.assertEqual(len(self.pipeline.process_json(json.dumps(NLP_RECORD))), 1)

    def test_invalid_json_does_not_crash(self):
        out = self.pipeline.process_json("{broken")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "invalid_input")
        self.assertIn("Invalid JSON", out[0]["errors"][0])

    def test_missing_input_file(self):
        out = self.pipeline.process_file(Path("no_such_file.json"))
        self.assertEqual(out[0]["status"], "invalid_input")

    def test_missing_reference_file_degrades_gracefully(self):
        config = PipelineConfig(reference_path=Path("missing_reference.json"))
        out = EntityMutationPipeline(config).process_claim(NLP_RECORD)
        self.assertFalse(out["reference_comparison"]["performed"])
        self.assertTrue(any("Reference records unavailable" in w for w in out["warnings"]))
        self.assertTrue(out["entities"]["amounts"])  # extraction still works

    def test_missing_registry_is_a_configuration_error(self):
        from shivansh_work.entity_extraction import ConfigError
        with self.assertRaises(ConfigError):
            EntityMutationPipeline(PipelineConfig(registry_path=Path("missing_registry.json")))


class CommandLineTests(unittest.TestCase):
    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "in.json", Path(folder) / "out.json"
            source.write_text(json.dumps([NLP_RECORD]), encoding="utf-8")
            exit_code = main(["--input", str(source), "--output", str(target)])
            self.assertEqual(exit_code, 0)
            result = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(result[0]["scheme"]["scheme_id"], "pm_kisan")

    def test_cli_reports_bad_input_with_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "bad.json", Path(folder) / "out.json"
            source.write_text("{oops", encoding="utf-8")
            self.assertEqual(main(["--input", str(source), "--output", str(target)]), 1)

    def test_cli_bad_registry_path(self):
        self.assertEqual(main(["--registry", "nope.json", "--input", "x.json"]), 2)


if __name__ == "__main__":
    unittest.main()
