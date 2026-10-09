"""Create two compact figures from saved evaluations, without loading any models.

Run plotting with a Python environment containing matplotlib, then run --pdf-only
with an environment containing reportlab. Outputs are deterministic saved-data
visualizations; the historical run folders are never changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "images" / "results"
RUNS = (
    ("Docling + RapidOCR", "OCR + agent", "#71869E",
     "docling-gold-results/plots/2026-07-05_135849"),
    ("Qwen2-VL 2B", "Transcript + agent", "#3971B8",
     "vlm-gold-results/plots/2026-07-06_063003"),
    ("Qwen2.5-VL 7B", "Direct image + normalization", "#007F70",
     "qwen-2.5-vl-7b/plots/color-fix"),
)
INK = "#172B3A"
MUTED = "#586A79"
LINE = "#E5EBEF"
BG = "#FFFFFF"
SIZE = (12, 7.2)
STEMS = ("pipeline-comparison", "selected-field-comparison")


def read_data():
    data = []
    for name, route, color, folder in RUNS:
        path = ROOT / "artifacts" / "agent" / folder
        summary = json.loads((path / "gold_eval_summary.json").read_text())
        with (path / "gold_eval_per_field.csv").open() as handle:
            fields = {r["field"]: r for r in csv.DictReader(handle)}
        assert summary["n_pages"] == 14
        assert sum(summary[k] for k in ("correct", "wrong", "missed")) == 294
        assert abs(summary["field_accuracy"] - summary["correct"] / 294) < 1e-9
        data.append(dict(name=name, route=route, color=color, summary=summary, fields=fields))
    return data


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": INK,
        "svg.fonttype": "none", "svg.hashsalt": "archivelens-results",
        "figure.facecolor": BG, "axes.facecolor": BG,
    })
    return plt


def frame(plt, title, subtitle, number):
    fig = plt.figure(figsize=SIZE)
    fig.text(.045, .945, "ARCHIVELENS  /  EXTRACTION BENCHMARK", color=MUTED,
             fontsize=10, weight="bold")
    fig.text(.955, .945, number, color=MUTED, fontsize=10, ha="right")
    fig.text(.045, .875, title, fontsize=25, weight="bold")
    fig.text(.045, .825, subtitle, fontsize=11, color=MUTED)
    return fig


def save(plt, fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180, facecolor=BG)
    fig.savefig(OUTPUT / f"{stem}.svg", facecolor=BG, metadata={"Date": None})
    plt.close(fig)


def overview(plt, data):
    fig = frame(plt, "Better extraction, fewer missing fields",
                "Same 14 gold pages  /  294 populated values  /  Current evaluation rubric", "01")
    gain = 100 * (data[-1]["summary"]["field_accuracy"] - data[0]["summary"]["field_accuracy"])
    fig.text(.045, .745, f"+{gain:.1f} percentage points", fontsize=15, weight="bold", color=RUNS[-1][2])
    fig.text(.405, .745, "Docling pipeline to direct Qwen2.5-VL extraction", fontsize=11, color=MUTED)
    ax = fig.add_axes((.30, .24, .43, .42))
    ax.set_xlim(0, 104)
    ax.set_ylim(-.6, 2.6)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=LINE, linewidth=.8)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Field accuracy (%)", fontsize=11, labelpad=14)
    fig.text(.815, .685, "WRONG", ha="center", fontsize=10, color=MUTED, weight="bold")
    fig.text(.925, .685, "MISSED", ha="center", fontsize=10, color=MUTED, weight="bold")
    for i, item in enumerate(data):
        y = 2 - i
        summary = item["summary"]
        pct = 100 * summary["field_accuracy"]
        fy = .24 + .42 * ((y + .6) / 3.2)
        ax.barh(y, 100, height=.47, color="#F0F4F6", zorder=1)
        ax.barh(y, pct, height=.47, color=item["color"], zorder=2)
        ax.text(pct - 2, y, f"{pct:.1f}%", ha="right", va="center", fontsize=14,
                weight="bold", color="white")
        fig.text(.045, fy + .010, item["name"], fontsize=13, weight="bold", va="center")
        fig.text(.045, fy - .025, item["route"], fontsize=9.5, color=MUTED, va="center")
        fig.text(.815, fy, str(summary["wrong"]), ha="center", va="center", fontsize=19)
        fig.text(.925, fy, str(summary["missed"]), ha="center", va="center", fontsize=19,
                 color=item["color"], weight="bold")
    fig.text(.045, .125, "Accuracy = correct / (correct + wrong + missed).", fontsize=10, color=MUTED)
    fig.text(.045, .09, "14 extra year predictions in the final run are flagged separately; they are outside this accuracy denominator.",
             fontsize=10, color="#8C4E3C")
    fig.text(.045, .042, "Source: saved gold_eval_summary.json files, July 5-7, 2026. Small development set; not a held-out test.",
             fontsize=9, color=MUTED)
    save(plt, fig, STEMS[0])


def field_comparison(plt, data):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    selected = [
        ("marriage_day", "Marriage day"),
        ("marriage_month", "Marriage month"),
        ("license_number", "License number"),
        ("officiant.signature", "Officiant signature"),
        ("marriage_location", "Marriage location"),
        ("relationship_between_parties", "Relationship"),
    ]
    values = []
    for key, _ in selected:
        row = []
        for item in data:
            stats = item["fields"][key]
            total = sum(int(stats[k]) for k in ("correct", "wrong", "missed"))
            assert total == 14 and int(stats["hallucinated"]) == 0
            row.append(100 * int(stats["correct"]) / total)
        values.append(row)
    values = np.array(values)
    fig = frame(plt, "Where the gains came from",
                "Six selected fields  /  14 populated values per field  /  Accuracy (%)", "02")
    ax = fig.add_axes((.285, .215, .67, .48))
    cmap = LinearSegmentedColormap.from_list("archive", ["#F3F6F7", "#BDDED8", "#007F70"])
    ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(7):
        ax.axhline(i - .5, color=BG, linewidth=5)
    for j in range(4):
        ax.axvline(j - .5, color=BG, linewidth=6)
    for j, item in enumerate(data):
        x = .285 + .67 * (j + .5) / 3
        fig.text(x, .748, item["name"], ha="center", fontsize=12, color=item["color"], weight="bold")
        fig.text(x, .719, item["route"], ha="center", fontsize=8.5, color=MUTED)
    for i, (_, label) in enumerate(selected):
        y = .215 + .48 * (1 - (i + .5) / 6)
        fig.text(.045, y, label, fontsize=12, va="center")
        for j, value in enumerate(values[i]):
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=15,
                    weight="bold" if j == 2 else "normal", color="white" if value >= 75 else INK)
    fig.text(.045, .13, "Dates and signatures improve; location and relationship remain difficult.",
             fontsize=12, weight="bold")
    fig.text(.045, .088, "Selection includes gains, a non-monotonic identifier result, and remaining weaknesses. Darker cells = higher accuracy.",
             fontsize=9.5, color=MUTED)
    fig.text(.045, .042, "Source: saved gold_eval_per_field.csv files, July 5-7, 2026. See docs/DATA_AND_EVALUATION.md for the full results.",
             fontsize=9, color=MUTED)
    save(plt, fig, STEMS[1])


def pdf():
    from reportlab.pdfgen import canvas
    # A landscape page sized for the chart's typography. Each chart is one page.
    report = ROOT / "gold-ocr-comparison-report.pdf"
    if report.exists():
        backup = ROOT / "tmp" / "pdfs" / "gold-ocr-comparison-report.original.pdf"
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(report.read_bytes())
    c = canvas.Canvas(str(report), pagesize=(864, 518.4))
    c.setTitle("ArchiveLens | Gold-set extraction comparison")
    c.setAuthor("ArchiveLens")
    for stem in STEMS:
        c.drawImage(str(OUTPUT / f"{stem}.png"), 0, 0, width=864, height=518.4)
        c.showPage()
    c.save()
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()
    if args.pdf_only:
        pdf()
    else:
        data = read_data()
        plt = setup()
        overview(plt, data)
        field_comparison(plt, data)
        print(OUTPUT)
