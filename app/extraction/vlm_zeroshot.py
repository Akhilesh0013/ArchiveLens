"""Zero-shot structured extraction with Qwen2.5-VL (notebooks/vlm.ipynb — Model 2).

Renders each processed PDF page and prompts the VLM to emit a MarriageRecord
JSON directly from the image — no OCR text step and no LangGraph agent.
Results are checkpointed to a JSONL and persisted to a dedicated SQLite DB,
mirroring how the qwen2-vl-2b and docling corpora were ingested.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import (
    GOLD_DIR,
    PROCESSED_DIR,
    PROCESSED_PDF_GLOB,
    QWEN25_DB_PATH,
    QWEN25_ENGINE_TAG,
    QWEN25_ERRORS_JSONL,
    QWEN25_RECORDS_JSONL,
    VLM_DIR,
    ZEROSHOT_VLM_PROMPT,
    settings,
)
from app.db.repository import RecordRepository
from app.ocr.render import pdf_page_numbers, render_pdf_page
from app.ocr.vlm import resolve_pdf
from app.schema import MarriageRecord, normalize_marriage_record

ZEROSHOT_ENGINE = f"zeroshot+{QWEN25_ENGINE_TAG}"


def _quiet_transformers() -> None:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*processor_kwargs.*",
        category=UserWarning,
    )


@lru_cache(maxsize=1)
def _load_model():
    _quiet_transformers()
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model_path = settings.zeroshot_vlm_model_path
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config, generate, apply_chat_template


# Integer fields where the VLM sometimes emits ordinals ("5th") or stray text.
_INT_FIELD_PATHS = (
    ("marriage_day",),
    ("marriage_year",),
    ("groom", "age"),
    ("bride", "age"),
)


def _coerce_int_like(value):
    """Pull the first integer out of strings like '5th' or 'age 23'."""
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        return int(match.group()) if match else None
    return value


def _coerce_bool_like(value):
    """Map VLM strings like 'PARENTS CONSENT FILED' / 'NO' onto booleans."""
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if not text:
        return None
    if text in {"no", "none", "false"} or "not" in text:
        return False
    if text in {"yes", "true"} or "filed" in text:
        return True
    return None


def _strip_color_code(value):
    """Clean short color codes ('COL.', ' wht ') → 'COL' / 'WHT'.

    The VLM often appends a trailing period to the coded color value. Since
    ``color`` is scored by exact match (not fuzzy), 'COL.' vs 'COL' counts as
    wrong, so we trim trailing punctuation/whitespace. Only applied to color —
    other fields legitimately keep periods (e.g. residence 'BALTIMORE MD.').
    """
    if not isinstance(value, str):
        return value
    cleaned = value.strip().strip(".").strip()
    return cleaned or None


def _coerce_payload_ints(payload: dict) -> dict:
    for path in _INT_FIELD_PATHS:
        obj = payload
        for key in path[:-1]:
            obj = obj.get(key) if isinstance(obj, dict) else None
            if obj is None:
                break
        if isinstance(obj, dict) and path[-1] in obj:
            obj[path[-1]] = _coerce_int_like(obj[path[-1]])
    if "parents_consent_filed" in payload:
        payload["parents_consent_filed"] = _coerce_bool_like(
            payload["parents_consent_filed"]
        )
    for party in ("groom", "bride"):
        block = payload.get(party)
        if isinstance(block, dict) and "color" in block:
            block["color"] = _strip_color_code(block["color"])
    return payload


def _parse_record_json(raw: str) -> dict:
    """Extract the JSON object from raw VLM output (fences / stray text tolerated)."""
    text = raw.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in model output: {raw[:200]!r}")
    return json.loads(text[start : end + 1])


def extract_record_from_image(image_path: Path) -> MarriageRecord:
    """Run Qwen2.5-VL zero-shot on one page image → validated MarriageRecord."""
    model, processor, config, generate, apply_chat_template = _load_model()
    formatted_prompt = apply_chat_template(
        processor,
        config,
        ZEROSHOT_VLM_PROMPT,
        num_images=1,
    )
    result = generate(
        model,
        processor,
        formatted_prompt,
        [str(image_path)],
        max_tokens=settings.zeroshot_vlm_max_tokens,
        temperature=0.0,
        verbose=False,
    )
    raw = result.text if hasattr(result, "text") else str(result)
    payload = _parse_record_json(raw)
    # Provenance is set by the pipeline, never trusted from the model.
    for key in ("source_pdf", "source_page", "ocr_engine"):
        payload.pop(key, None)
    payload = _coerce_payload_ints(payload)
    return normalize_marriage_record(MarriageRecord.model_validate(payload))


def _load_checkpoint_keys(jsonl_path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add((rec["source_pdf"], int(rec["source_page"])))
    return keys


def _append_jsonl(jsonl_path: Path, payload: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


@dataclass
class ZeroShotScanResult:
    processed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def scan_processed_corpus(
    *,
    force: bool = False,
    render_dir: Path | None = None,
    limit: int | None = None,
) -> ZeroShotScanResult:
    """Zero-shot scan all processed PDF pages → JSONL checkpoint + SQLite DB.

    Resumable: pages already in the checkpoint JSONL are skipped unless
    ``force`` is set (which restarts the checkpoint and DB rows are upserted).
    """
    if force and QWEN25_RECORDS_JSONL.exists():
        QWEN25_RECORDS_JSONL.unlink()

    repository = RecordRepository(QWEN25_DB_PATH)
    done = _load_checkpoint_keys(QWEN25_RECORDS_JSONL)
    render_root = render_dir or (VLM_DIR / "_rendered_corpus")
    result = ZeroShotScanResult()

    pdf_paths = sorted(PROCESSED_DIR.glob(PROCESSED_PDF_GLOB))
    pending: list[tuple[Path, int]] = []
    for pdf_path in pdf_paths:
        for page_no in pdf_page_numbers(pdf_path):
            if (pdf_path.name, page_no) in done:
                result.skipped += 1
                continue
            pending.append((pdf_path, page_no))

    if limit is not None:
        pending = pending[:limit]

    total = len(done) + len(pending)
    print(
        f"Zero-shot scan ({settings.zeroshot_vlm_model_path}): "
        f"{len(done)}/{total} pages done, {len(pending)} to process",
        flush=True,
    )

    for index, (pdf_path, page_no) in enumerate(pending, start=1):
        pdf_name = pdf_path.name
        print(f"  [{index}/{len(pending)}] {pdf_name} p{page_no} …", flush=True)
        try:
            image_path = render_pdf_page(pdf_path, page_no, render_root)
            record = extract_record_from_image(image_path)
            record.source_pdf = pdf_name
            record.source_page = page_no
            record.ocr_engine = ZEROSHOT_ENGINE

            existing = repository.find_by_provenance(pdf_name, page_no, ZEROSHOT_ENGINE)
            if existing is not None:
                upload_id = existing.upload_id
            else:
                upload_id = repository.create_upload(
                    filename=f"{pdf_name}#p{page_no}",
                    stored_path=str(pdf_path),
                    ocr_engine=ZEROSHOT_ENGINE,
                    status="extracted",
                )
            record_id = repository.save_record(upload_id, record)
            _append_jsonl(QWEN25_RECORDS_JSONL, record.model_dump(mode="json"))

            groom = record.groom.full_name or "?"
            bride = record.bride.full_name or "?"
            print(f"       saved record #{record_id} — {groom} / {bride}", flush=True)
            result.processed += 1
        except Exception as exc:
            message = f"{pdf_name} p{page_no}: {exc}"
            result.errors.append(message)
            _append_jsonl(
                QWEN25_ERRORS_JSONL,
                {"source_pdf": pdf_name, "source_page": page_no, "error": str(exc)},
            )
            print(f"       failed — {exc}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Gold evaluation (14 labeled pages) — mirrors docling/vlm gold runs
# ---------------------------------------------------------------------------

GOLD_SOURCE = "qwen25"


def _gold_page_image(source_pdf: str, source_page: int, render_dir: Path) -> Path:
    """Prefer the pre-rendered gold PNG; fall back to rendering the PDF page."""
    stem = Path(source_pdf).stem
    candidate = GOLD_DIR / "images" / f"{stem}_p{source_page:03d}.png"
    if candidate.exists():
        return candidate
    pdf_path = resolve_pdf(source_pdf)
    if pdf_path is None:
        raise FileNotFoundError(
            f"No gold image or source PDF for {source_pdf} p{source_page}"
        )
    return render_pdf_page(pdf_path, source_page, render_dir)


@dataclass
class ZeroShotGoldResult:
    metrics: dict
    predictions: dict[tuple[str, int], MarriageRecord]
    output_path: Path
    run_dir: Path
    report_paths: dict[str, Path]


def run_gold_eval(*, run_name: str | None = None, render_dir: Path | None = None) -> ZeroShotGoldResult:
    """Zero-shot Qwen2.5-VL on the 14 gold pages → predictions + CSVs + plots.

    Writes to ``artifacts/agent/qwen-2.5-vl-7b/`` via the shared gold-run
    machinery, exactly like the docling and vlm gold pipelines.
    """
    from app.config import gold_predictions_path
    from app.pipeline.eval import evaluate_predictions
    from app.pipeline.gold_labels import iter_gold_label_paths
    from app.pipeline.gold_plots import publish_gold_run

    render_root = render_dir or (VLM_DIR / "_rendered_gold")
    output_path = gold_predictions_path(GOLD_SOURCE)
    predictions: dict[tuple[str, int], MarriageRecord] = {}

    gold_paths = iter_gold_label_paths()
    print(
        f"Zero-shot gold eval ({settings.zeroshot_vlm_model_path}) on "
        f"{len(gold_paths)} labeled pages …",
        flush=True,
    )

    with output_path.open("w") as handle:
        for index, gold_path in enumerate(gold_paths, start=1):
            label = json.loads(gold_path.read_text())
            source_pdf = label["source_pdf"]
            source_page = int(label["source_page"])
            key = (source_pdf, source_page)

            print(f"  [{index}/{len(gold_paths)}] {source_pdf} p{source_page} …", flush=True)
            image_path = _gold_page_image(source_pdf, source_page, render_root)
            record = extract_record_from_image(image_path)
            record.source_pdf = source_pdf
            record.source_page = source_page
            record.ocr_engine = ZEROSHOT_ENGINE

            predictions[key] = record
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    metrics = evaluate_predictions(predictions)
    run_dir, report_paths = publish_gold_run(
        predictions, metrics, ocr_source=GOLD_SOURCE, run_name=run_name
    )
    return ZeroShotGoldResult(
        metrics=metrics,
        predictions=predictions,
        output_path=output_path,
        run_dir=run_dir,
        report_paths=report_paths,
    )
