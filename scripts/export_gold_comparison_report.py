"""Export a gold eval comparison report (Docling vs Qwen2-VL vs Qwen2.5-VL) as standalone HTML."""

from __future__ import annotations

import base64
import csv
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AGENT_DIR = PROJECT_ROOT / "artifacts/agent"
DOCLING_RUN = AGENT_DIR / "docling-gold-results/plots/2026-07-05_135849"
VLM_RUN = AGENT_DIR / "vlm-gold-results/plots/2026-07-06_063003"
QWEN25_RUN = AGENT_DIR / "qwen-2.5-vl-7b/plots/color-fix"
OUTPUT = AGENT_DIR / "gold-ocr-comparison-report.html"
OUTPUT_PDF = AGENT_DIR / "gold-ocr-comparison-report.pdf"

PLOT_FILES = (
    "01_per_field_accuracy.png",
    "02_outcome_mix.png",
    "03_accuracy_by_block.png",
    "05_per_page_accuracy.png",
    "04_error_type_stacked.png",
)

CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)


@dataclass
class Engine:
    key: str
    label: str
    sublabel: str
    run: Path


def _read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "gold_eval_summary.json").read_text())


def _read_per_field(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (run_dir / "gold_eval_per_field.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "field": row["field"],
                    "accuracy": float(row["accuracy"]),
                    "correct": int(row["correct"]),
                    "wrong": int(row["wrong"]),
                    "missed": int(row["missed"]),
                }
            )
    return rows


def _read_errors(run_dir: Path, *, limit: int = 10) -> list[dict]:
    path = run_dir / "gold_eval_errors.csv"
    rows: list[dict] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows[:limit]


def _short_field(field: str) -> str:
    return (
        field.replace("groom.", "g: ")
        .replace("bride.", "b: ")
        .replace("relationship_between_parties", "relationship")
        .replace("officiant.signature", "officiant")
    )


def _img_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _plot_img(run_dir: Path, filename: str) -> str:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return _img_data_uri(path)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summary_row(label: str, values: list[str], *, highlight_max: bool = False) -> str:
    best_idx = -1
    if highlight_max:
        numeric = []
        for value in values:
            try:
                numeric.append(float(value.rstrip("%")))
            except ValueError:
                numeric.append(float("nan"))
        if any(n == n for n in numeric):  # not all NaN
            best_idx = max(range(len(numeric)), key=lambda i: numeric[i])
    cells = "".join(
        f'<td class="num{" best" if i == best_idx else ""}">{html.escape(v)}</td>'
        for i, v in enumerate(values)
    )
    return f"<tr><td>{html.escape(label)}</td>{cells}</tr>"


def _plot_row(engines: list[Engine], filename: str) -> str:
    figures = "".join(
        f"""
      <figure>
        <figcaption>{html.escape(eng.label)}</figcaption>
        <img src="{_plot_img(eng.run, filename)}" alt="{html.escape(eng.label)}" />
      </figure>"""
        for eng in engines
    )
    return f'<div class="plot-row cols-{len(engines)}">{figures}</div>'


def _accuracy_matrix(engines: list[Engine]) -> str:
    per_engine = {eng.key: {r["field"]: r["accuracy"] for r in _read_per_field(eng.run)} for eng in engines}
    field_order = [r["field"] for r in _read_per_field(engines[0].run)]

    header_cells = "".join(f"<th>{html.escape(eng.label)}</th>" for eng in engines)
    body_rows = []
    for field in field_order:
        accs = [per_engine[eng.key].get(field, 0.0) for eng in engines]
        best = max(accs) if accs else 0.0
        cells = "".join(
            f'<td class="num{" best" if a == best and best > 0 else ""}">{_pct(a)}</td>'
            for a in accs
        )
        body_rows.append(f"<tr><td>{html.escape(_short_field(field))}</td>{cells}</tr>")

    return f"""
    <table class="matrix-table">
      <thead><tr><th>Field</th>{header_cells}</tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>"""


def _error_table(engine: Engine) -> str:
    rows = []
    for err in _read_errors(engine.run):
        gold = err["gold"] or "—"
        pred = err["prediction"] or "—"
        if len(gold) > 24:
            gold = gold[:21] + "…"
        if len(pred) > 24:
            pred = pred[:21] + "…"
        rows.append(
            "<tr>"
            f'<td>{html.escape(err["page"].replace("_content.pdf", ""))}</td>'
            f'<td>{html.escape(_short_field(err["field"]))}</td>'
            f'<td><span class="tag {html.escape(err["outcome"])}">{html.escape(err["outcome"])}</span></td>'
            f'<td>{html.escape(gold)}</td>'
            f'<td>{html.escape(pred)}</td>'
            "</tr>"
        )
    return f"""
        <div>
          <h3>{html.escape(engine.label)}<span class="sub"> · {html.escape(engine.sublabel)}</span></h3>
          <table class="errors-table">
            <thead>
              <tr><th>Pg</th><th>Field</th><th>Type</th><th>Gold</th><th>Predicted</th></tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>"""


def build_html(engines: list[Engine]) -> str:
    summaries = {eng.key: _read_summary(eng.run) for eng in engines}

    header_cells = "".join(
        f"<th>{html.escape(eng.label)}<br><span class='th-sub'>{html.escape(eng.sublabel)}</span></th>"
        for eng in engines
    )
    summary_rows = "".join(
        [
            _summary_row(
                "Field accuracy",
                [_pct(summaries[e.key]["field_accuracy"]) for e in engines],
                highlight_max=True,
            ),
            _summary_row("Correct", [str(summaries[e.key]["correct"]) for e in engines], highlight_max=True),
            _summary_row("Wrong", [str(summaries[e.key]["wrong"]) for e in engines]),
            _summary_row("Missed", [str(summaries[e.key]["missed"]) for e in engines]),
            _summary_row(
                "Hallucinated",
                [str(summaries[e.key].get("hallucinated", 0)) for e in engines],
            ),
            _summary_row("Pages evaluated", [str(summaries[e.key]["n_pages"]) for e in engines]),
        ]
    )

    best = max(engines, key=lambda e: summaries[e.key]["field_accuracy"])
    worst = min(engines, key=lambda e: summaries[e.key]["field_accuracy"])
    spread_pp = (summaries[best.key]["field_accuracy"] - summaries[worst.key]["field_accuracy"]) * 100

    per_field_plots = _plot_row(engines, "01_per_field_accuracy.png")
    outcome_plots = _plot_row(engines, "02_outcome_mix.png")
    block_plots = _plot_row(engines, "03_accuracy_by_block.png")
    page_plots = _plot_row(engines, "05_per_page_accuracy.png")
    error_stack_plots = _plot_row(engines, "04_error_type_stacked.png")
    matrix = _accuracy_matrix(engines)
    error_tables = "".join(_error_table(eng) for eng in engines)
    run_footer = " · ".join(
        f"{html.escape(eng.key)}: {html.escape(eng.run.parent.parent.name)}/plots/{html.escape(eng.run.name)}"
        for eng in engines
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gold Eval — OCR &amp; extraction pipelines</title>
  <style>
    @page {{ size: letter; margin: 0.5in; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-size: 10pt;
      line-height: 1.45;
      color: #1a1a1a;
      background: #fff;
    }}
    .doc {{ max-width: 7.5in; margin: 0 auto; }}
    .page {{ padding: 0.1in 0 0.2in; }}
    .page-break {{ page-break-after: always; break-after: page; }}
    .masthead {{ border-bottom: 2px solid #1a1a1a; padding-bottom: 10px; margin-bottom: 14px; }}
    .masthead h1 {{ font-size: 18pt; font-weight: 600; letter-spacing: -0.02em; }}
    .masthead p {{ margin-top: 4px; color: #555; font-size: 9.5pt; }}
    .lede {{ margin-bottom: 14px; color: #333; font-size: 10pt; }}
    .lede strong {{ color: #111; }}
    h2 {{
      font-size: 11pt;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #444;
      margin: 18px 0 10px;
      padding-bottom: 4px;
      border-bottom: 1px solid #ddd;
    }}
    h3 {{ font-size: 9.5pt; font-weight: 600; color: #333; margin-bottom: 6px; }}
    h3 .sub {{ font-weight: 400; color: #888; font-size: 8.5pt; }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 6px;
      font-size: 9.5pt;
    }}
    .summary-table th, .summary-table td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e5e5;
      text-align: left;
    }}
    .summary-table th {{
      font-weight: 600;
      color: #555;
      font-size: 8.5pt;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #f7f7f7;
    }}
    .summary-table th .th-sub {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: #999; }}
    .summary-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .summary-table td.best {{ background: #e7f5ec; color: #1a7f37; font-weight: 600; }}
    .summary-table tr:last-child td {{ border-bottom: none; }}
    .plot-row {{ display: grid; gap: 10px; margin-bottom: 4px; }}
    .plot-row.cols-2 {{ grid-template-columns: 1fr 1fr; }}
    .plot-row.cols-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
    figure {{ margin: 0; }}
    figcaption {{
      font-size: 8.5pt;
      font-weight: 600;
      color: #555;
      margin-bottom: 4px;
      text-align: center;
    }}
    img {{ width: 100%; height: auto; display: block; }}
    .matrix-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 8.5pt;
    }}
    .matrix-table th, .matrix-table td {{
      padding: 4px 6px;
      border-bottom: 1px solid #eee;
      text-align: left;
    }}
    .matrix-table th {{
      font-size: 8pt;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #666;
      background: #fafafa;
    }}
    .matrix-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .matrix-table td.best {{ color: #1a7f37; font-weight: 600; }}
    .errors-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
    .errors-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 7.5pt;
      margin-top: 4px;
    }}
    .errors-table th, .errors-table td {{
      padding: 3px 4px;
      border-bottom: 1px solid #eee;
      vertical-align: top;
    }}
    .errors-table th {{
      font-size: 7pt;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #666;
      background: #fafafa;
    }}
    .tag {{
      display: inline-block;
      font-size: 6.5pt;
      font-weight: 600;
      padding: 1px 4px;
      border-radius: 3px;
      text-transform: uppercase;
    }}
    .tag.wrong {{ background: #ffebe9; color: #cf222e; }}
    .tag.missed {{ background: #fff8c5; color: #9a6700; }}
    .tag.hallucinated {{ background: #ede7f6; color: #6b21a8; }}
    .footer {{
      margin-top: 14px;
      padding-top: 8px;
      border-top: 1px solid #ddd;
      font-size: 7.5pt;
      color: #888;
    }}
    .no-print {{ color: #888; font-size: 9pt; margin-bottom: 12px; }}
    @media print {{
      .no-print {{ display: none; }}
      .page {{ padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="doc">
    <p class="no-print">Gold eval comparison report · print or save as PDF</p>

    <section class="page page-break">
      <header class="masthead">
        <h1>Gold Eval — OCR &amp; Extraction Pipelines</h1>
        <p>Baltimore marriage certificates · 14 labeled pages · 21 fields / page</p>
      </header>

      <p class="lede">
        Three pipelines on identical gold labels. <strong>Docling + RapidOCR</strong> and
        <strong>Qwen2-VL 2B</strong> feed OCR text into the same LangGraph extraction agent;
        <strong>Qwen2.5-VL 7B</strong> extracts structured fields <strong>zero-shot</strong>
        from the page image (no OCR text, no agent). Field accuracy rises
        {_pct(summaries[worst.key]["field_accuracy"])} → {_pct(summaries[best.key]["field_accuracy"])}
        (<strong>{best.label} leads by {spread_pp:.1f} pp</strong>). Note the zero-shot model
        trades a few hallucinated header fields for far fewer missed values.
      </p>

      <table class="summary-table">
        <thead>
          <tr><th>Metric</th>{header_cells}</tr>
        </thead>
        <tbody>{summary_rows}</tbody>
      </table>

      <h2>Per-field accuracy</h2>
      {per_field_plots}
      <p class="footer">Runs — {run_footer}</p>
    </section>

    <section class="page page-break">
      <header class="masthead">
        <h1>Outcomes &amp; per-page accuracy</h1>
        <p>Correct / wrong / missed distribution, form-block accuracy, and page-by-page results</p>
      </header>

      <h2>Outcome distribution</h2>
      {outcome_plots}

      <h2>Accuracy by form block</h2>
      {block_plots}

      <h2>Per-page field accuracy</h2>
      {page_plots}
    </section>

    <section class="page">
      <header class="masthead">
        <h1>Field-level comparison &amp; errors</h1>
        <p>Per-field accuracy across all three pipelines and representative extraction errors</p>
      </header>

      <h2>Per-field accuracy — all pipelines</h2>
      {matrix}

      <h2>Error type by field</h2>
      {error_stack_plots}

      <h2>Representative errors</h2>
      <div class="errors-grid">{error_tables}</div>

      <p class="footer">
        Generated from gold_eval_*.csv and plot PNGs in each run folder.
      </p>
    </section>
  </div>
</body>
</html>
"""


def _find_chrome() -> Path | None:
    for candidate in CHROME_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def export_pdf(html_path: Path, pdf_path: Path) -> Path:
    """Render HTML to PDF via headless Chrome/Chromium."""
    chrome = _find_chrome()
    if chrome is None:
        raise FileNotFoundError(
            "No Chrome/Chromium found. Install Google Chrome or print the HTML manually."
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(chrome),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=3000",
            "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not pdf_path.is_file():
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"PDF export failed: {detail}")
    return pdf_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docling-run", type=Path, default=DOCLING_RUN)
    parser.add_argument("--vlm-run", type=Path, default=VLM_RUN)
    parser.add_argument("--qwen25-run", type=Path, default=QWEN25_RUN)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=OUTPUT_PDF,
        help="PDF output path (default: alongside HTML)",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF export")
    args = parser.parse_args()

    engines = [
        Engine("docling", "Docling + RapidOCR", "OCR → agent", args.docling_run),
        Engine("vlm", "Qwen2-VL 2B", "OCR → agent", args.vlm_run),
        Engine("qwen25", "Qwen2.5-VL 7B", "zero-shot", args.qwen25_run),
    ]
    for eng in engines:
        if not eng.run.is_dir():
            raise SystemExit(f"Missing {eng.key} run: {eng.run}")

    report = build_html(engines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote {args.output}")

    if not args.no_pdf:
        try:
            pdf_path = export_pdf(args.output, args.pdf)
            print(f"Wrote {pdf_path}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"PDF export skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
