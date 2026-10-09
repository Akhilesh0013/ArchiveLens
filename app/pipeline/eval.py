"""Evaluate agent predictions against gold labels."""

from __future__ import annotations

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from app.config import (
    EVAL_FIELDS,
    FUZZY_FIELDS,
    FUZZY_THRESHOLD,
    NUMERIC_ID_FIELDS,
)
from app.pipeline.gold_labels import load_gold_labels
from app.schema import MarriageRecord


def get_field(record: dict | MarriageRecord, dotted: str):
    if isinstance(record, MarriageRecord):
        record = record.model_dump(mode="json")
    obj = record
    for part in dotted.split("."):
        if obj is None:
            return None
        obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
    return obj


def norm(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _digits_only(value) -> str | None:
    """Strip non-digits for license/application # comparison."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    return digits or None


def _numeric_id_match(gold_v, pred_v) -> bool:
    gold_digits = _digits_only(gold_v)
    pred_digits = _digits_only(pred_v)
    if gold_digits is None or pred_digits is None:
        return False
    return gold_digits == pred_digits


def classify(gold_v, pred_v, field: str | None = None) -> str:
    g, p = norm(gold_v), norm(pred_v)
    if g is None and p is None:
        return "true_null"
    if g is None and p is not None:
        return "hallucinated"
    if g is not None and p is None:
        return "missed"
    if g == p:
        return "correct"
    if field in NUMERIC_ID_FIELDS and _numeric_id_match(gold_v, pred_v):
        return "correct"
    if field in FUZZY_FIELDS:
        ratio = SequenceMatcher(None, g, p).ratio()
        if ratio >= FUZZY_THRESHOLD:
            return "correct"
    return "wrong"


def load_gold() -> dict[tuple[str, int], dict]:
    return load_gold_labels()


def evaluate_predictions(
    preds: dict[tuple[str, int], MarriageRecord],
) -> dict[str, float | int]:
    gold = load_gold()
    keys = sorted(set(gold) & set(preds))
    if not keys:
        return {"field_accuracy": 0.0, "n_pages": 0, "correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0}

    totals = {"correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0}
    for key in keys:
        for field in EVAL_FIELDS:
            outcome = classify(get_field(gold[key], field), get_field(preds[key], field), field)
            if outcome in totals:
                totals[outcome] += 1

    scored = totals["correct"] + totals["wrong"] + totals["missed"]
    accuracy = totals["correct"] / scored if scored else 0.0
    return {
        "field_accuracy": accuracy,
        "n_pages": len(keys),
        **totals,
    }


def _page_label(source_pdf: str, source_page: int) -> str:
    return f"{source_pdf} p{source_page}"


def build_field_scores(
    preds: dict[tuple[str, int], MarriageRecord],
) -> list[dict[str, str | int | float | None]]:
    gold = load_gold()
    keys = sorted(set(gold) & set(preds))
    rows: list[dict[str, str | int | float | None]] = []

    for key in keys:
        source_pdf, source_page = key
        page = _page_label(source_pdf, source_page)
        for field in EVAL_FIELDS:
            gold_v = get_field(gold[key], field)
            pred_v = get_field(preds[key], field)
            outcome = classify(gold_v, pred_v, field)
            rows.append(
                {
                    "page": page,
                    "source_pdf": source_pdf,
                    "source_page": source_page,
                    "field": field,
                    "outcome": outcome,
                    "gold": gold_v,
                    "prediction": pred_v,
                    "source": "agent",
                }
            )
    return rows


def write_eval_reports(
    preds: dict[tuple[str, int], MarriageRecord],
    output_dir: Path,
    *,
    prefix: str = "gold_eval",
) -> dict[str, Path]:
    """Write predictions, errors, and per-field breakdown CSVs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_field_scores(preds)

    errors_path = output_dir / f"{prefix}_errors.csv"
    with errors_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page",
                "field",
                "outcome",
                "gold",
                "prediction",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            if row["outcome"] in {"wrong", "missed", "hallucinated"}:
                writer.writerow(
                    {
                        "page": row["page"],
                        "field": row["field"],
                        "outcome": row["outcome"],
                        "gold": row["gold"],
                        "prediction": row["prediction"],
                        "source": row["source"],
                    }
                )

    per_field_path = output_dir / f"{prefix}_per_field.csv"
    field_stats: dict[str, dict[str, int]] = {}
    for row in rows:
        field = str(row["field"])
        stats = field_stats.setdefault(
            field,
            {"correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0, "total": 0},
        )
        outcome = str(row["outcome"])
        if outcome in stats:
            stats[outcome] += 1
        if outcome != "true_null":
            stats["total"] += 1

    with per_field_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["field", "correct", "wrong", "missed", "hallucinated", "accuracy"],
        )
        writer.writeheader()
        for field in EVAL_FIELDS:
            stats = field_stats.get(field, {})
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            accuracy = correct / total if total else 0.0
            writer.writerow(
                {
                    "field": field,
                    "correct": stats.get("correct", 0),
                    "wrong": stats.get("wrong", 0),
                    "missed": stats.get("missed", 0),
                    "hallucinated": stats.get("hallucinated", 0),
                    "accuracy": round(accuracy, 4),
                }
            )

    per_page_path = output_dir / f"{prefix}_per_page.csv"
    page_stats: dict[str, dict[str, int | str]] = {}
    for row in rows:
        page = str(row["page"])
        stats = page_stats.setdefault(
            page,
            {
                "page": page,
                "source_pdf": row["source_pdf"],
                "source_page": row["source_page"],
                "correct": 0,
                "wrong": 0,
                "missed": 0,
                "hallucinated": 0,
                "total": 0,
            },
        )
        outcome = str(row["outcome"])
        if outcome in stats:
            stats[outcome] = int(stats[outcome]) + 1  # type: ignore[operator]
        if outcome != "true_null":
            stats["total"] = int(stats["total"]) + 1  # type: ignore[operator]

    with per_page_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page",
                "source_pdf",
                "source_page",
                "correct",
                "wrong",
                "missed",
                "hallucinated",
                "total",
                "accuracy",
            ],
        )
        writer.writeheader()
        for page in sorted(page_stats):
            stats = page_stats[page]
            total = int(stats["total"])
            correct = int(stats["correct"])
            accuracy = correct / total if total else 0.0
            writer.writerow(
                {
                    "page": page,
                    "source_pdf": stats["source_pdf"],
                    "source_page": stats["source_page"],
                    "correct": correct,
                    "wrong": int(stats["wrong"]),
                    "missed": int(stats["missed"]),
                    "hallucinated": int(stats["hallucinated"]),
                    "total": total,
                    "accuracy": round(accuracy, 4),
                }
            )

    return {
        "errors": errors_path,
        "per_field": per_field_path,
        "per_page": per_page_path,
    }
