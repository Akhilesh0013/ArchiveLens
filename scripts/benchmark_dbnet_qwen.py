"""Paired full-page vs DBNet-guided Qwen benchmark; writes isolated artifacts.

Run from the repository root with:
  .venv/bin/python -B -m scripts.benchmark_dbnet_qwen --output artifacts/optimization/run-name
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import time

from PIL import Image, ImageDraw

from app.config import EVAL_FIELDS, GOLD_DIR, ZEROSHOT_VLM_PROMPT, settings
from app.extraction.vlm_zeroshot import _coerce_payload_ints, _parse_record_json
from app.ocr.dbnet_regions import DBNetDetector, DETECTOR_SHA256, pack_regions, preserve_inked_gaps, regions_from_boxes
from app.pipeline.eval import classify, get_field
from app.pipeline.gold_labels import iter_gold_label_paths
from app.schema import MarriageRecord, normalize_marriage_record


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(gold, record):
    return dict(Counter(
        classify(get_field(gold, f), get_field(record, f), f) for f in EVAL_FIELDS
    ))


def summarize(rows):
    summary = {}
    for mode in ("full_page", "dbnet_packed"):
        selected = [r for r in rows if r["mode"] == mode]
        if not selected:
            continue
        counts = Counter()
        for row in selected:
            counts.update(row["outcomes"])
        denominator = sum(counts[k] for k in ("correct", "wrong", "missed"))
        summary[mode] = {
            "n_pages": len(selected),
            "field_accuracy": counts["correct"] / denominator if denominator else None,
            **dict(counts),
            "failed_pages": sum(r["error"] is not None for r in selected),
            "median_pipeline_seconds": statistics.median(r["pipeline_seconds"] for r in selected),
            "total_pipeline_seconds": sum(r["pipeline_seconds"] for r in selected),
            "median_detector_seconds": statistics.median(r["detector_seconds"] for r in selected),
            "median_prompt_tokens": statistics.median(r["prompt_tokens"] for r in selected),
            "total_generation_tokens": sum(r["generation_tokens"] for r in selected),
            "max_mlx_peak_gb": max(r["mlx_peak_gb"] for r in selected),
            "median_pixel_ratio": statistics.median(r["pixel_ratio"] for r in selected),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detector", type=Path, default=Path("artifacts/optimization/models/ch_PP-OCRv4_det_mobile.onnx"))
    parser.add_argument("--model", default=settings.zeroshot_vlm_model_path)
    parser.add_argument("--limit", type=int, help="Interleave volumes when selecting a pilot subset")
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--merge-gap", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.padding < 0 or args.merge_gap < 0 or args.max_tokens < 1:
        parser.error("invalid crop or generation settings")
    paths = iter_gold_label_paths()
    # Alternate documents for pilots, avoiding a subset drawn only from volume 1.
    groups = {}
    for path in paths:
        label = json.loads(path.read_text())
        groups.setdefault(label["source_pdf"], []).append(path)
    paths = [group[i] for i in range(max(map(len, groups.values()), default=0))
             for group in groups.values() if i < len(group)]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("no gold labels found")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "hardware": platform.platform() + " " + platform.machine(),
        "versions": {n: version(n) for n in ("rapidocr", "onnxruntime", "mlx-vlm", "mlx", "transformers")},
        "detector_sha256": DETECTOR_SHA256,
        "prompt": ZEROSHOT_VLM_PROMPT,
        "temperature": 0.0,
        "seed": 0,
        "evaluation_fields": EVAL_FIELDS,
        "labels": {p.name: digest(p) for p in paths},
        "code_sha256": {p: digest(p) for p in (
            "app/ocr/dbnet_regions.py", "scripts/benchmark_dbnet_qwen.py",
            "app/pipeline/eval.py", "app/config.py", "app/extraction/vlm_zeroshot.py")},
        "timing": "Warm model, alternate mode order by page; includes detection/packing, model image preprocessing, generation and parsing; excludes image file reads, model load, warmup, artifact writes and evaluation.",
        "memory": "Per-call MLX peak GB, includes loaded VLM; excludes ONNX CPU allocations. Not total process memory.",
        "coverage": "Failed generation produces an empty record and is scored as misses; no pages silently omitted.",
    }
    started = time.perf_counter()
    detector = DBNetDetector(args.detector, max_side=args.max_side)
    manifest["detector_load_seconds"] = time.perf_counter() - started

    if not args.detect_only:
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        started = time.perf_counter()
        model, processor = load(args.model)
        config = load_config(args.model)
        prompt = apply_chat_template(processor, config, ZEROSHOT_VLM_PROMPT, num_images=1)
        manifest["model_load_seconds"] = time.perf_counter() - started
        first_image = GOLD_DIR / "images" / (paths[0].stem + ".png")
        with Image.open(first_image) as im:
            warmup_image = im.convert("RGB")
        # Both modes use one image, identical prompt, model and generation settings.
        generate(model, processor, prompt, [warmup_image], max_tokens=1, temperature=0.0, verbose=False)
        mx.synchronize()
        manifest["warmup"] = "One unscored full-page prefill and one output token"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    detections = []
    for index, label_path in enumerate(paths):
        gold = json.loads(label_path.read_text())
        image_path = GOLD_DIR / "images" / (label_path.stem + ".png")
        with Image.open(image_path) as im:
            source = im.convert("RGB")
        manifest.setdefault("images", {})[image_path.name] = digest(image_path)
        # Preparation is included in packed mode and excluded from full-page mode.
        start = time.perf_counter()
        boxes = detector(source)
        detector_seconds = time.perf_counter() - start
        regions = regions_from_boxes(boxes, source.size, padding=args.padding, merge_gap=args.merge_gap)
        regions = preserve_inked_gaps(source, regions)
        packed, fallback = pack_regions(source, regions)
        preparation_seconds = time.perf_counter() - start
        pixel_ratio = packed.width * packed.height / (source.width * source.height)
        info = {
            "page": label_path.stem, "n_boxes": len(boxes), "regions": [asdict(r) for r in regions],
            "source_size": source.size, "packed_size": packed.size, "pixel_ratio": pixel_ratio,
            "fallback": fallback, "detector_seconds": detector_seconds,
        }
        detections.append(info)
        page_dir = args.output / label_path.stem
        page_dir.mkdir()
        packed.save(page_dir / "packed.png")
        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            points = [tuple(point) for point in box]
            draw.line(points + points[:1], fill="red", width=2)
        for r in regions:
            draw.rectangle((r.left, r.top, r.right, r.bottom), outline="blue", width=3)
        overlay.save(page_dir / "detections.png")
        print(f"[{index + 1}/{len(paths)}] {label_path.stem}: {len(boxes)} boxes, {len(regions)} bands, {pixel_ratio:.1%} pixels", flush=True)
        if args.detect_only:
            continue
        order = ("full_page", "dbnet_packed") if index % 2 == 0 else ("dbnet_packed", "full_page")
        for mode in order:
            image = source if mode == "full_page" else packed
            mx.clear_cache()
            mx.reset_peak_memory()
            mx.random.seed(0)
            start = time.perf_counter()
            raw, error = "", None
            prompt_tokens, generation_tokens = 0, 0
            try:
                result = generate(model, processor, prompt, [image], max_tokens=args.max_tokens, temperature=0.0, verbose=False)
                mx.synchronize()
                raw = result.text
                prompt_tokens = int(result.prompt_tokens)
                generation_tokens = int(result.generation_tokens)
                payload = _coerce_payload_ints(_parse_record_json(raw))
                for key in ("source_pdf", "source_page", "ocr_engine"):
                    payload.pop(key, None)
                record = normalize_marriage_record(MarriageRecord.model_validate(payload))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record = MarriageRecord()
            elapsed = time.perf_counter() - start
            record.source_pdf = gold["source_pdf"]
            record.source_page = gold["source_page"]
            record.ocr_engine = mode + "+qwen2.5-vl-7b"
            row = {
                "page": label_path.stem, "mode": mode, "outcomes": score(gold, record),
                "pipeline_seconds": elapsed + (preparation_seconds if mode == "dbnet_packed" else 0),
                "generation_and_parse_seconds": elapsed,
                "detector_seconds": detector_seconds if mode == "dbnet_packed" else 0,
                "prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                "mlx_peak_gb": mx.get_peak_memory() / 1e9,
                "pixel_ratio": pixel_ratio if mode == "dbnet_packed" else 1.0,
                "fallback": fallback if mode == "dbnet_packed" else None,
                "error": error, "record": record.model_dump(mode="json"),
            }
            (page_dir / (mode + ".txt")).write_text(raw)
            rows.append(row)
            with (args.output / "pages.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            (args.output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
            print(f"  {mode}: {row['pipeline_seconds']:.2f}s, {prompt_tokens} prompt tokens, {row['outcomes']}, error={error}", flush=True)
    (args.output / "detections.json").write_text(json.dumps(detections, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if rows:
        print(json.dumps(summarize(rows), indent=2), flush=True)
    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
