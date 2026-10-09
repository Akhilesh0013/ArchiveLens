# Dataset, schema, and evaluation

## Source corpus

The workspace contains two Baltimore City marriage-certificate PDF volumes from
the 1930s–1940s. After blank-back filtering, the corpus contains 225 certificate
pages: 131 from the first processed volume and 94 from the second.

The documents combine printed boilerplate and handwritten or typed values. The
forms record historical race/color terminology such as `WHT` and `COL`. The
project preserves those source values for faithful transcription and research.

## Structured record

One certificate maps to one `MarriageRecord`.

### Header and identifiers

- `jurisdiction_note`
- `state`
- `city`
- `license_number`
- `application_number`

### Marriage event

- `marriage_day`
- `marriage_month`
- `marriage_year`
- `marriage_year_text`
- `marriage_location`

`marriage_year` is populated only when four digits are explicitly present under
the gold-labeling convention. A phrase such as “Nineteen hundred and thirty-nine”
belongs in `marriage_year_text`; inferring `1939` in the numeric field is treated
as a hallucination during evaluation.

### Groom and bride blocks

Both parties have:

- `full_name`
- `residence`
- `age`
- `color`
- `nativity`
- `occupation`
- `marital_status`

### Consent, relationship, and officiant

- `groom_consent_by`
- `bride_consent_by`
- `parents_consent_filed`
- `relationship_between_parties`
- `officiant.signature`
- `officiant.official_character`
- `officiant.address`

### Provenance

- `source_pdf`
- `source_page`
- `ocr_engine`

Provenance is assigned by the pipeline and is never trusted from model output.

## Gold set

The benchmark contains 14 page images with human-created JSON labels, split across
both processed volumes. Labelers transcribe what is visible, use null for blank or
illegible values, preserve `NONE` as a real occupation, and avoid inferring values
that are not explicitly written.

The evaluator scores these 22 fields:

```text
license_number
application_number
marriage_day
marriage_month
marriage_year
marriage_location
groom.full_name
groom.residence
groom.age
groom.color
groom.nativity
groom.occupation
groom.marital_status
bride.full_name
bride.residence
bride.age
bride.color
bride.nativity
bride.occupation
bride.marital_status
relationship_between_parties
officiant.signature
```

Across 14 pages, 294 of the 308 page-field positions contain a gold value. The
remaining 14 are true-null positions for `marriage_year`.

## Outcome definitions

For every scored field, `app/pipeline/eval.py` assigns one outcome:

- **correct:** the prediction matches under that field's comparison rule;
- **wrong:** both values exist but do not match;
- **missed:** gold has a value and the prediction is null;
- **hallucinated:** gold is null and the prediction has a value; or
- **true null:** both values are null.

Field accuracy is:

```text
correct / (correct + wrong + missed)
```

Hallucinations are disclosed as a separate count. True-null positions do not
inflate accuracy.

## Comparison rules

All values are uppercased, trimmed, and whitespace-normalized. Numeric identifiers
compare their digit sequences so a label prefix does not make `No. 34178` differ
from `34178`.

These handwritten/free-text fields use `SequenceMatcher` with a threshold of 0.85:

- groom/bride name
- groom/bride residence
- groom/bride nativity
- groom/bride occupation
- officiant signature

Other fields require an exact match after basic normalization. This is why
representation cleanup such as `COL.` → `COL` matters.

## Current evaluator results

| Pipeline | Correct | Wrong | Missed | Hallucinated | Accuracy |
|---|---:|---:|---:|---:|---:|
| TrOCR baseline, rescored | 181 | 60 | 53 | 0 | 61.6% |
| Docling/RapidOCR + agent | 189 | 51 | 54 | 0 | 64.3% |
| Qwen2-VL 2B transcript + agent | 229 | 14 | 51 | 0 | 77.9% |
| Qwen2.5-VL 7B direct + normalization | 268 | 19 | 7 | 14 | 91.2% |

The TrOCR row was recomputed from
`artifacts/extracted/marriage_records_trocr.jsonl`. The other rows correspond to
the saved published summaries:

- `artifacts/agent/docling-gold-results/plots/2026-07-05_135849/`
- `artifacts/agent/vlm-gold-results/plots/2026-07-06_063003/`
- `artifacts/agent/qwen-2.5-vl-7b/plots/color-fix/`

## What improved

The shift from Docling/RapidOCR to Qwen2-VL primarily reduced wrong values:
51 → 14. Misses remained almost flat: 54 → 51. The VLM transcript was cleaner,
but still omitted information.

Direct Qwen2.5-VL extraction changed that error profile. Compared with Qwen2-VL,
misses fell from 51 to 7 while 39 more populated values were recovered correctly.
The largest gains appear in dates, identifiers, relationship text, and officiant
signatures.

Several party fields reached 100% on the 14 pages, including groom name,
residence, age, nativity, occupation, and marital status. Hard remaining fields
include marriage location and relationship between parties.

## Why two headline score pairs exist

The original evaluation notebook removes punctuation and performs
word-order-insensitive fuzzy matching. The current application evaluator retains
more punctuation and uses direct sequence similarity. Therefore, scores from the
two rubrics must not be mixed.

Consistent comparisons are:

- original notebook rubric: **54.1% → 89.8%**;
- current project evaluator: **61.6% → 91.2%**.

The README uses 54% → 90% as the narrative headline because it is a like-for-like
comparison under the original rubric, then publishes the current evaluator table
for reproducibility.

## Limitations and next evaluation steps

The 14 pages are a development set. Model selection, prompts, and normalization
were informed by their errors, so they are not an untouched test set. A stronger
evaluation would:

1. freeze prompts, normalization, and scoring rules;
2. label a larger stratified set from both volumes;
3. hold back pages until the pipeline is frozen;
4. report confidence intervals and per-form-condition slices;
5. measure character/word error rate for transcription separately from field
   extraction accuracy; and
6. evaluate retrieval and answer faithfulness independently from extraction.
