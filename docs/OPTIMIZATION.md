# Optimization experiments

Optimization is the organizing theme of ArchiveLens. The project treats a document
pipeline as a sequence of measurable information bottlenecks rather than a single
OCR model call.

## Optimization objectives

The primary objective is structured-field accuracy. Runtime, memory, input tokens,
and operational complexity are secondary objectives constrained by accuracy.

```text
maximize: field accuracy and evidence coverage
minimize: wrong values, missed values, latency, memory, and model input
subject to: page provenance and reproducible evaluation
```

## Completed experiment ladder

### Experiment A: line recognition baseline

**Hypothesis:** a dedicated recognizer should read clean line crops well.

**Design:** RapidOCR detects lines; TrOCR Base Printed recognizes crops; a
structured LLM maps the reassembled text to the schema.

**Finding:** line recognition loses page-level relationships and struggles with a
form containing both print and handwriting. Original-rubric field accuracy was
54.1%.

### Experiment B: full-page OCR plus targeted cleanup

**Hypothesis:** full-page OCR will preserve more form context, and conservative
domain cleanup will make label/value boundaries recoverable.

**Design:** Docling forces RapidOCR over the full scanned page. Cleanup repairs
recurring fused labels and a short list of observed whole-word confusions. The
same schema extraction logic consumes the result.

**Finding:** the saved current-evaluator run reached 64.3%. Improvements were real
but date, location, nativity, color, and signature fields remained weak.

### Experiment C: local VLM transcription

**Hypothesis:** a VLM will use page layout and visual context to produce a cleaner
transcript while preserving the downstream agent for controlled comparison.

**Design:** Qwen2-VL 2B Instruct (4-bit MLX) transcribes each page. The same
LangGraph extraction agent and schema remain downstream.

**Finding:** accuracy reached 77.9%. Wrong values fell sharply, but misses did not.
This isolated completeness—not just recognition—as the next bottleneck.

### Experiment D: direct image-to-schema extraction

**Hypothesis:** supplying the schema with the page image will avoid information
loss at the transcript boundary.

**Design:** Qwen2.5-VL 7B Instruct (4-bit MLX) emits the complete JSON record from
one image. Application code parses, normalizes, validates, and assigns provenance.

**Finding:** accuracy reached 91.2% under the current evaluator. Misses fell to 7.
The gain includes both a stronger visual model and an architectural change, so it
should be described as a pipeline improvement rather than attributed solely to
removing OCR.

### Experiment E: representation normalization

**Hypothesis:** some residual errors are semantic matches expressed differently.

**Design:** normalize only fields with a clear contract: integer-like values,
boolean consent phrases, and short color codes.

**Finding:** trimming punctuation from color codes fixed five saved corpus errors,
increasing the current score from 89.5% to 91.2% without another model pass.

## DBNet + heavy recognizer pilot

### Question

Can a very small text detector remove blank page area before Qwen2.5-VL, reducing
visual tokens and latency while retaining field accuracy?

### Implementation

`app/ocr/dbnet_regions.py` loads only the PP-OCRv4 mobile DB detection model:

- model size: 4,745,517 bytes (4.75 MB decimal);
- backend: ONNX Runtime on CPU;
- pinned SHA-256:
  `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9`;
- no RapidOCR recognition or orientation model is loaded.

The detector runs on a reduced image and maps quadrilateral boxes back to original
coordinates. Nearby detections become horizontal bands. Bands retain their shared
horizontal extent so groom/bride columns stay aligned, and pixels are copied at
their original scale.

An early preview exposed a critical failure: the detector missed part of a
handwritten signature, and naïve gap removal cut through it. The corrected packer
scans internal gaps for dark pixels and merges bands when any ink is present. Empty
detection results fall back to the full page.

### Benchmark protocol

`scripts/benchmark_dbnet_qwen.py` compares two modes with the same:

- Qwen2.5-VL 7B model and quantization;
- extraction prompt and schema;
- temperature (`0.0`) and seed;
- maximum generation length;
- page set and evaluator; and
- one image per model call.

It records:

- field outcomes for every page;
- detector, generation, and end-to-end time;
- prompt and generation tokens;
- MLX peak memory;
- retained pixel ratio;
- detector overlays and packed previews;
- hashes of labels, images, code, and detector weights; and
- parse failures, which are scored as misses rather than silently dropped.

Mode order alternates by page to reduce warm-cache/order bias. Model load and a
one-token warmup are excluded from per-page timing.

### Preliminary findings

Detection completed on all 14 gold pages:

- 41–59 text boxes per page;
- approximately 0.20 seconds median detection time in the first detection-only
  run;
- 56.6% median retained pixels after the ink-preservation guard.

The full inference run was interrupted and is not a completed benchmark. The
partial guarded run contains three full-page and four packed outputs, so its
accuracy and latency aggregates are not directly comparable. Its packed images
used a median 1,757 prompt tokens versus 2,823 for full pages, a reduction of about
38%. However, packed latency did not consistently improve, peak MLX memory was not
lower, and field regressions appeared—including a marriage day changing from 20
to 28.

No speedup or accuracy-preservation claim should be made from this pilot.

### Interpretation

The experiment shows that the simple optimization assumption—fewer pixels means a
cheaper heavy recognizer—is incomplete:

1. the processor discretizes images into model-specific visual tokens;
2. a tall packed image may still be expensive despite fewer raw pixels;
3. layout changes can harm spatial interpretation;
4. a detector trained for text boxes can miss long handwritten strokes; and
5. autoregressive JSON generation can dominate total latency.

The detector is still useful as a routing signal. A stronger next design would use
it to decide between full-page and region-based processing instead of always
rewriting the page layout.

## Recommended next experiments

### 1. Region-aware prompting without repacking

Keep the original page and add detector boxes or coarse field regions as visual
guidance. This preserves global layout while testing whether grounding improves
attention.

### 2. Fixed form-template crops

These certificates share a stable layout. Use broad semantic regions—header,
marriage event, party blocks, relationship, and officiant—instead of individual
text boxes. Broad regions are less likely to clip handwriting and can be evaluated
independently.

### 3. Cascaded recognition

Use a lightweight recognizer for high-confidence printed fields and route only
low-confidence or handwritten regions to Qwen2.5-VL. This better matches the goal
of combining cheap detection with expensive recognition.

### 4. Dynamic routing

Run full-page Qwen only when detector coverage, ink density, or lightweight OCR
confidence falls below a threshold. Report the Pareto frontier across accuracy,
latency, and percentage of pages escalated.

### 5. Complete the paired benchmark

Before changing the architecture again, finish all 14 paired pages and report:

- macro field accuracy and per-field deltas;
- paired latency distribution, not only medians;
- prompt prefill versus generation time;
- total process memory in addition to MLX peak memory;
- detector recall on manually marked field regions; and
- the percentage of pages where the fallback path activates.

The optimization should be considered successful only if it moves the
accuracy–cost frontier, not merely one metric in isolation.
