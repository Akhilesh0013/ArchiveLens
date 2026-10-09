"""Write gold eval CSV reports and PNG plots to a timestamped run folder."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from app.config import gold_plots_root
from app.pipeline.eval import write_eval_reports
from app.schema import MarriageRecord

FIELD_GROUPS = {
    "license_number": "Header",
    "application_number": "Header",
    "marriage_day": "Header",
    "marriage_month": "Header",
    "marriage_location": "Header",
    "relationship_between_parties": "Other",
    "officiant.signature": "Other",
}

# Fields in EVAL_FIELDS but not labeled in gold — omit from per-field plots.
PLOT_EXCLUDE_FIELDS = frozenset({"marriage_year"})


def new_run_dir(ocr_source: str, name: str | None = None) -> Path:
    stamp = name or datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    return gold_plots_root(ocr_source) / stamp


def _short_label(field: str) -> str:
    return field.replace("groom.", "G: ").replace("bride.", "B: ")


def _group(field: str) -> str:
    if field in FIELD_GROUPS:
        return FIELD_GROUPS[field]
    if field.startswith("groom."):
        return "Groom"
    if field.startswith("bride."):
        return "Bride"
    return "Other"


def _load_per_field_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "gold_eval_per_field.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_field_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["field"] not in PLOT_EXCLUDE_FIELDS]


def plot_per_field_accuracy(rows: list[dict], out: Path, overall_pct: float) -> None:
    rows = _plot_field_rows(rows)
    sorted_rows = sorted(rows, key=lambda r: float(r["accuracy"]))
    labels = [_short_label(r["field"]) for r in sorted_rows]
    values = [float(r["accuracy"]) * 100 for r in sorted_rows]
    colors = ["#8B4513" if v >= overall_pct else "#C4A484" for v in values]

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(labels, values, color=colors, edgecolor="#333", linewidth=0.4)
    ax.axvline(
        overall_pct,
        color="#B8860B",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall {overall_pct:.1f}%",
    )
    ax.set_xlabel("Field accuracy (%)")
    ax.set_title("Agent gold eval — per-field accuracy (14 pages)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_outcome_mix(summary: dict, out: Path) -> None:
    labels = ["Correct", "Wrong", "Missed"]
    values = [summary["correct"], summary["wrong"], summary["missed"]]
    colors = ["#2E7D32", "#C62828", "#F9A825"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title(
        f"Outcome mix — {summary['n_pages']} gold pages\n"
        f"Field accuracy: {summary['field_accuracy']:.1%}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accuracy_by_block(rows: list[dict], out: Path, overall_pct: float) -> None:
    blocks: dict[str, list[float]] = {}
    for row in rows:
        if float(row["accuracy"]) == 0 and int(row["correct"]) + int(row["wrong"]) + int(row["missed"]) == 0:
            continue
        group = _group(row["field"])
        blocks.setdefault(group, []).append(float(row["accuracy"]))

    order = ["Groom", "Bride", "Header", "Other"]
    labels = [g for g in order if g in blocks]
    means = [sum(blocks[g]) / len(blocks[g]) * 100 for g in labels]
    colors = ["#5C4033", "#8B6914", "#C62828", "#6D4C41"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, color=colors[: len(labels)], edgecolor="#333", linewidth=0.5)
    ax.axhline(
        overall_pct,
        color="#B8860B",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall {overall_pct:.1f}%",
    )
    ax.set_ylabel("Mean field accuracy (%)")
    ax.set_title("Accuracy by form block")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_error_stacked(rows: list[dict], out: Path) -> None:
    rows = _plot_field_rows(rows)
    sorted_rows = sorted(rows, key=lambda r: float(r["accuracy"]))
    labels = [_short_label(r["field"]) for r in sorted_rows]
    correct = [int(r["correct"]) for r in sorted_rows]
    wrong = [int(r["wrong"]) for r in sorted_rows]
    missed = [int(r["missed"]) for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(labels, correct, label="Correct", color="#2E7D32")
    ax.barh(labels, wrong, left=correct, label="Wrong", color="#C62828")
    left = [c + w for c, w in zip(correct, wrong)]
    ax.barh(labels, missed, left=left, label="Missed", color="#F9A825")
    ax.set_xlabel("Count (14 gold pages per field)")
    ax.set_title("Error type by field — stacked counts")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_page(run_dir: Path, out: Path, overall_pct: float) -> None:
    per_page = run_dir / "gold_eval_per_page.csv"
    if not per_page.exists():
        return

    with per_page.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    labels = [r["page"].replace("_content.pdf", "") for r in rows]
    acc = [float(r["accuracy"]) * 100 for r in rows]
    colors = [
        "#2E7D32" if a >= overall_pct else "#C62828" if a < overall_pct - 10 else "#B8860B"
        for a in acc
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), acc, color=colors, edgecolor="#333", linewidth=0.4)
    ax.axhline(
        overall_pct,
        color="#333",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall {overall_pct:.1f}%",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Field accuracy (%)")
    ax.set_title("Agent gold eval — accuracy by page")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_plot_pngs(run_dir: Path, summary: dict) -> None:
    rows = _load_per_field_rows(run_dir)
    overall_pct = float(summary["field_accuracy"]) * 100

    plot_per_field_accuracy(rows, run_dir / "01_per_field_accuracy.png", overall_pct)
    plot_outcome_mix(summary, run_dir / "02_outcome_mix.png")
    plot_accuracy_by_block(rows, run_dir / "03_accuracy_by_block.png", overall_pct)
    plot_error_stacked(rows, run_dir / "04_error_type_stacked.png")
    plot_per_page(run_dir, run_dir / "05_per_page_accuracy.png", overall_pct)


def publish_gold_run(
    predictions: dict[tuple[str, int], MarriageRecord],
    metrics: dict[str, float | int],
    *,
    ocr_source: str = "docling",
    run_name: str | None = None,
) -> tuple[Path, dict[str, Path]]:
    """Write CSV reports, summary JSON, PNG plots, and manifest to a new run folder."""
    run_dir = new_run_dir(ocr_source, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    report_paths = write_eval_reports(predictions, run_dir, prefix="gold_eval")
    (run_dir / "gold_eval_summary.json").write_text(json.dumps(metrics, indent=2))
    generate_plot_pngs(run_dir, metrics)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "field_accuracy": metrics["field_accuracy"],
        "n_pages": metrics["n_pages"],
        "correct": metrics["correct"],
        "wrong": metrics["wrong"],
        "missed": metrics["missed"],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    return run_dir, report_paths
