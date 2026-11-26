# VERILENS – Member 3: Entity Extraction + Rule-Based Mutation Detection

Deterministic module of the VERILENS pipeline:

```
NLP module (Member 2)  ->  THIS MODULE (Member 3)  ->  next VERILENS module
                           extraction -> normalization -> rule-based comparison
```

It does **not** do language detection, preprocessing or NLI. Semantic mutation
comparison belongs to Member 6.

## Project structure

```
minorpj/
├── shivansh_work/
│   ├── __init__.py            exports EntityMutationPipeline
│   ├── normalization.py       text / amount / date helpers
│   ├── input_validation.py    validates NLP records, configurable field names
│   ├── entity_extraction.py   scheme registry, rules loader, all extractors
│   ├── mutation_detection.py  reference store + rule-based comparison
│   └── pipeline_interface.py  integration class + command line
├── data/
│   ├── scheme_registry.json   schemes + aliases (en / hi / hinglish)  – ILLUSTRATIVE
│   ├── extraction_rules.json  keyword lists (eligibility, roles, bounds) – extendable
│   └── reference_records.json reference values for comparison – ILLUSTRATIVE, verified=false
├── tests/                     unittest test files (pytest can run them too)
├── pytest.ini                 lets pytest import the package from the project root
├── requirements.txt
└── README.md
```

Two files were added to the proposed structure: `data/extraction_rules.json`
(so Hindi/English/Hinglish keywords are not scattered in Python) and
`data/reference_records.json` (the "available reference records" the
comparison needs). `tests/test_input_validation.py` and `pytest.ini` are extras.

## Run

Python 3.9+, standard library only.

```bash
# tests
python -m unittest discover -s tests -v

# process NLP output (file or stdin)
python -m shivansh_work.pipeline_interface --input nlp_output.json --output result.json
```

From Python:

```python
from shivansh_work import EntityMutationPipeline
pipeline = EntityMutationPipeline()
result = pipeline.process_claim(nlp_record)      # dict, JSON-serializable
results = pipeline.process_json(json_string)     # list of dicts
```

## Input contract (PROPOSED – to be agreed with Member 2)

The real NLP output format was not available, so this contract is a proposal.
Only a text field is mandatory. Field names are configurable
(`PipelineConfig.field_map`, dot notation allowed, e.g. `"claim.text"`).

```json
{
  "claim_id": "string",
  "text": "original claim text",
  "normalized_text": "text after NLP preprocessing",
  "language": "en | hi | hinglish",
  "entities": [{"type": "SCHEME | AMOUNT | DATE | AGE | ELIGIBILITY", "text": "..."}],
  "metadata": {}
}
```

Accepted payloads: one record, a list of records, or `{"claims": [...]}`.
`text` is used for extraction by default (`normalized_text` is the fallback;
switch with `text_preference`). NLP entities are only *hints*: a `SCHEME`
hint that is missing from the registry is reported as an unknown alias.
Other entity types (e.g. `ORG`) raise an "unsupported entity type" warning.

## Output schema (`schema_version` 1.0, every key always present)

| key | meaning |
|---|---|
| `claim_id`, `status` | `ok`, `ok_with_warnings`, `invalid_input`, `error` |
| `language`, `text_used`, `processed_text` | `processed_text` is the normalized text all `span` offsets refer to |
| `schemes[]`, `scheme` | `scheme_id`, `canonical_name`, `matched_text`, `matched_alias`, `alias_language`, `span`, `mentions`, `source`; `scheme` is set only when exactly one scheme is found |
| `unknown_scheme_mentions[]` | NLP scheme hints not found in the registry |
| `entities.amounts[]` | `text`, `amount`, `currency`, `currency_explicit`, `frequency`, `annual_equivalent`, `instalment_count`, `role` (`benefit` / `income_limit` / `unspecified`), `context`, `occurrences` |
| `entities.dates[]` | `text`, `iso`, `precision` (`day`, `month`, `day_month`, `financial_year`), `role` (`deadline`, `start`, `end`, `mentioned`); financial years carry `label`, `start_iso`, `end_iso` |
| `entities.validity_periods[]` | `start`, `end` (ISO) |
| `entities.age_limits` | `min_age`, `min_inclusive`, `max_age`, `max_inclusive`, `matches[]` |
| `entities.eligibility[]` | `category`, `values`, `matched_terms`, `condition_text` (original clause), `limit_value`, `limit_unit`, `bound` |
| `reference_comparison` | `performed`, `reason`, `scheme_id`, `reference_source`, `reference_verified` |
| `mutations[]` | one record per compared attribute: `attribute`, `result`, `mutation_type`, `reason`, values |
| `mutation_summary` | `any_change_detected`, `mutation_types`, `attributes_unchanged`, `attributes_incomparable` |
| `warnings[]`, `errors[]` | validation warnings / blocking errors |
| `nlp_input` | the untouched NLP record |
| `provenance` | module name, versions, `text_field_used`, input metadata, `processed_at` |

## Mutation rules

| mutation_type | when |
|---|---|
| `amount_increased` / `amount_decreased` | claim and reference benefit amounts have the **same frequency** and differ |
| `deadline_changed` | claim deadline (full ISO date) differs from reference deadline |
| `validity_period_changed` | claim start or end differs from reference |
| `eligibility_condition_changed` | claim names a value absent from the reference, or a numeric limit (income, land) differs |
| `min_age_changed` / `max_age_changed` | value differs, or same value with different inclusive/exclusive boundary |

`result: "incomparable"` (with a reason) is returned whenever values cannot be
compared reliably: no reference record, different frequency, partial date,
different bounds, several schemes in one claim, etc. No change is claimed then.

## Adding data

* New scheme or alias: edit `data/scheme_registry.json` (no code change).
* New keywords (occupations, categories, benefit words): edit `data/extraction_rules.json`.
* Reference values: edit `data/reference_records.json` and set `verified` to true
  only after checking an official notification (record source name, URL, date).

## Known limitations

* The registry and reference files are illustrative, not complete.
* Pattern-based extraction: unusual phrasings can be missed. Age numbers followed by "years"
  with no age keyword may in rare cases be durations (a duration-word filter reduces this).
* Dates are read in Indian order (DD/MM/YYYY). Two-digit years are not guessed.
* Amounts without a currency symbol/word are assumed INR and flagged with a warning.
