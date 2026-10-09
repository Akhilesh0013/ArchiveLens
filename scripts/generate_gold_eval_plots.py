"""Regenerate gold eval CSVs and PNG plots from existing agent predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import gold_predictions_path, settings  # noqa: E402
from app.ocr.registry import resolve_backend  # noqa: E402
from app.pipeline.eval import evaluate_predictions  # noqa: E402
from app.pipeline.gold_plots import publish_gold_run  # noqa: E402
from app.schema import MarriageRecord  # noqa: E402


def _load_predictions(path: Path) -> dict[tuple[str, int], MarriageRecord]:
    predictions: dict[tuple[str, int], MarriageRecord] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = MarriageRecord.model_validate_json(line)
            key = (record.source_pdf or "", record.source_page or 0)
            predictions[key] = record
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate gold eval CSVs and PNG plots from marriage_records_agent.jsonl",
    )
    parser.add_argument(
        "--ocr",
        choices=["docling", "vlm"],
        help="Gold results backend folder (default: OCR_BACKEND from .env or docling)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Predictions JSONL (overrides --ocr default path)",
    )
    parser.add_argument(
        "--name",
        help="Optional run folder name (default: UTC timestamp YYYY-MM-DD_HHMMSS)",
    )
    args = parser.parse_args()

    backend = resolve_backend(args.ocr)
    path = args.path or gold_predictions_path(backend)

    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: make run-gold-{backend}")

    predictions = _load_predictions(path)
    metrics = evaluate_predictions(predictions)
    run_dir, report_paths = publish_gold_run(
        predictions, metrics, ocr_source=backend, run_name=args.name,
    )

    print(f"Saved run to {run_dir}/")
    print(f"  field_accuracy: {metrics['field_accuracy']:.1%}")
    print(f"  Summary:  {run_dir / 'gold_eval_summary.json'}")
    print(f"  Errors:   {report_paths['errors']}")
    print(f"  Per field:{report_paths['per_field']}")
    print(f"  Per page: {report_paths['per_page']}")
    for png in sorted(run_dir.glob("*.png")):
        print(f"  {png.name}")


if __name__ == "__main__":
    main()
