"""Command-line interface for ingestion and database management."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import gold_predictions_path
from app.ingestion.service import IngestionService
from app.ocr.registry import resolve_backend
from app.pipeline.eval import evaluate_predictions
from app.pipeline.gold_plots import publish_gold_run
from app.schema import MarriageRecord


def cmd_import_jsonl(args: argparse.Namespace) -> int:
    service = IngestionService()
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    count = service.import_jsonl(path, skip_existing=not args.force)
    print(f"Imported {count} record(s) from {path}")
    print(f"Total records in database: {service.repository.count_records()}")
    return 0


def cmd_import_batch(args: argparse.Namespace) -> int:
    service = IngestionService()
    counts = service.import_default_batch(skip_existing=not args.force)
    if not counts:
        print("No marriage_records_*.jsonl files found in artifacts/extracted/")
        return 1

    for name, count in counts.items():
        print(f"  {name}: {count} imported")
    print(f"Total records in database: {service.repository.count_records()}")
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    service = IngestionService()
    backend = resolve_backend(args.ocr)
    print(f"Processing {path.name} (OCR={backend} + agent pipeline)...")
    result = service.ingest_file(path, path.name, ocr_backend=backend)

    print(f"Upload id: {result.upload_id}")
    print(f"Stored at: {result.stored_path}")
    for page in result.pages:
        groom = page.record.groom.full_name or "?"
        bride = page.record.bride.full_name or "?"
        print(
            f"  page {page.page_no}: record #{page.record_id} — "
            f"{groom} / {bride}"
        )
        if page.validation_errors:
            print(f"    validation: {page.validation_errors}")

    if result.skipped_pages:
        print(f"Skipped empty pages: {result.skipped_pages}")

    print(f"Total records in database: {service.repository.count_records()}")
    return 0


def cmd_run_gold(args: argparse.Namespace) -> int:
    from app.config import DOCLING_DIR, VLM_GOLD_PAGES_JSONL

    service = IngestionService()
    backend = resolve_backend(args.ocr)
    ocr_ref = str(DOCLING_DIR) if backend == "docling" else str(VLM_GOLD_PAGES_JSONL)
    print(f"Running agent on gold-labeled pages (cached {backend} OCR from {ocr_ref})...")
    result = service.run_gold_eval(persist=args.persist, ocr_source=backend)

    metrics = result.metrics
    print(f"\nGold eval ({metrics['n_pages']} pages):")
    print(f"  field_accuracy: {metrics['field_accuracy']:.1%}")
    print(f"  correct: {metrics['correct']}  wrong: {metrics['wrong']}  "
          f"missed: {metrics['missed']}  hallucinated: {metrics['hallucinated']}")
    print(f"\nPredictions: {result.output_path}")
    if result.run_dir:
        print(f"Run folder:  {result.run_dir}/")
        print(f"  (CSVs, summary JSON, and PNG plots)")
    if result.report_paths:
        print(f"Errors:      {result.report_paths['errors']}")
        print(f"Per field:   {result.report_paths['per_field']}")
        print(f"Per page:    {result.report_paths['per_page']}")
    return 0


def cmd_build_vlm_cache(args: argparse.Namespace) -> int:
    from app.ocr.vlm import VLM_PAGES_JSONL, build_gold_vlm_cache

    print("Building gold VLM cache (14 labeled pages → artifacts/vlm/pages.jsonl)...")
    count = build_gold_vlm_cache()
    if count == 0:
        print(
            "No pages written. Add gold/images/*.png or PDFs under data/baltimore-marriage-certificates/",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {count} page(s) to {VLM_PAGES_JSONL}")
    return 0


def cmd_build_vlm_corpus_cache(args: argparse.Namespace) -> int:
    from app.config import PROCESSED_DIR, VLM_CORPUS_PAGES_JSONL

    service = IngestionService()
    print(
        f"Running Qwen2-VL on PDFs in {PROCESSED_DIR} "
        "(this may take a long time on Apple Silicon)..."
    )
    count = service.build_vlm_corpus_cache(force=args.force)
    if count == 0 and not VLM_CORPUS_PAGES_JSONL.exists():
        print(f"No pages written. Add PDFs matching {PROCESSED_DIR}/*_content.pdf", file=sys.stderr)
        return 1
    print(f"Wrote {count} new page(s) to {VLM_CORPUS_PAGES_JSONL}")
    from app.pipeline.ocr_loader import load_vlm_pages

    total_cached = len(load_vlm_pages(VLM_CORPUS_PAGES_JSONL))
    print(f"Corpus cache total: {total_cached} page(s)")
    return 0


def cmd_ingest_vlm_corpus(args: argparse.Namespace) -> int:
    from app.config import VLM_CORPUS_PAGES_JSONL, VLM_DB_PATH
    from app.pipeline.ocr_loader import load_vlm_pages

    service = IngestionService()
    cached = len(load_vlm_pages(VLM_CORPUS_PAGES_JSONL))
    in_db = service.repository_for_backend("vlm").count_records()
    print(
        f"Running agent extraction on {cached} cached VLM OCR page(s) "
        f"({in_db} already in {VLM_DB_PATH.name})..."
    )
    try:
        result = service.ingest_vlm_corpus(skip_existing=not args.force)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"\nDone — processed: {result.processed}  skipped: {result.skipped}  "
        f"errors: {len(result.errors)}"
    )
    if result.errors:
        print(f"Errors ({len(result.errors)}):", file=sys.stderr)
        for message in result.errors[:10]:
            print(f"  {message}", file=sys.stderr)
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more", file=sys.stderr)
    print(f"VLM database: {VLM_DB_PATH}")
    print(f"Total VLM records: {service.repository_for_backend('vlm').count_records()}")
    return 0 if not result.errors or result.processed else 1


def cmd_scan_qwen25_corpus(args: argparse.Namespace) -> int:
    from app.config import QWEN25_DB_PATH, QWEN25_RECORDS_JSONL
    from app.extraction.vlm_zeroshot import scan_processed_corpus

    result = scan_processed_corpus(force=args.force, limit=args.limit)
    print(
        f"\nDone — processed: {result.processed}  skipped: {result.skipped}  "
        f"errors: {len(result.errors)}"
    )
    if result.errors:
        print(f"Errors ({len(result.errors)}):", file=sys.stderr)
        for message in result.errors[:10]:
            print(f"  {message}", file=sys.stderr)
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more", file=sys.stderr)
    print(f"Records JSONL: {QWEN25_RECORDS_JSONL}")

    from app.db.repository import RecordRepository

    print(f"Zero-shot database: {QWEN25_DB_PATH}")
    print(f"Total zero-shot records: {RecordRepository(QWEN25_DB_PATH).count_records()}")
    return 0 if result.processed or not result.errors else 1


def cmd_run_gold_qwen25(args: argparse.Namespace) -> int:
    from app.extraction.vlm_zeroshot import run_gold_eval

    result = run_gold_eval(run_name=args.name)
    metrics = result.metrics
    print(f"\nZero-shot gold eval ({metrics['n_pages']} pages):")
    print(f"  field_accuracy: {metrics['field_accuracy']:.1%}")
    print(
        f"  correct: {metrics['correct']}  wrong: {metrics['wrong']}  "
        f"missed: {metrics['missed']}  hallucinated: {metrics['hallucinated']}"
    )
    print(f"\nPredictions: {result.output_path}")
    print(f"Run folder:  {result.run_dir}/")
    print("  (CSVs, summary JSON, and PNG plots)")
    print(f"Errors:      {result.report_paths['errors']}")
    print(f"Per field:   {result.report_paths['per_field']}")
    print(f"Per page:    {result.report_paths['per_page']}")
    return 0


def cmd_retranscribe_vlm_page(args: argparse.Namespace) -> int:
    from app.config import VLM_CORPUS_PAGES_JSONL
    from app.ocr.vlm import retranscribe_corpus_page

    try:
        text = retranscribe_corpus_page(args.pdf, args.page)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Updated {VLM_CORPUS_PAGES_JSONL}")
    print(f"Transcription length: {len(text)} chars")
    print("---")
    print(text[:2000] + ("…" if len(text) > 2000 else ""))
    return 0


def cmd_gold_report(args: argparse.Namespace) -> int:
    backend = resolve_backend(args.ocr)
    path = Path(args.path) if args.path else gold_predictions_path(backend)
    if not path.exists():
        print(f"Predictions file not found: {path}", file=sys.stderr)
        return 1

    predictions: dict[tuple[str, int], MarriageRecord] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = MarriageRecord.model_validate_json(line)
            key = (record.source_pdf or "", record.source_page or 0)
            predictions[key] = record

    metrics = evaluate_predictions(predictions)
    run_dir, report_paths = publish_gold_run(
        predictions, metrics, ocr_source=backend, run_name=args.name,
    )

    print(f"Generated reports from {path}")
    print(f"  field_accuracy: {metrics['field_accuracy']:.1%}")
    print(f"  correct: {metrics['correct']}  wrong: {metrics['wrong']}  "
          f"missed: {metrics['missed']}  hallucinated: {metrics['hallucinated']}")
    print(f"Run folder:  {run_dir}/")
    print(f"Errors:      {report_paths['errors']}")
    print(f"Per field:   {report_paths['per_field']}")
    print(f"Per page:    {report_paths['per_page']}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    service = IngestionService()
    records = service.list_records(limit=args.limit)

    if not records:
        print("No records in database.")
        return 0

    print(f"{'ID':>4}  {'Groom':<24} {'Bride':<24} {'Year':>4}  Source")
    print("-" * 80)
    for item in records:
        rec = item.record
        year = rec.marriage_year or ""
        source = f"{rec.source_pdf or item.filename} p{rec.source_page or '?'}"
        print(
            f"{item.id:>4}  "
            f"{(rec.groom.full_name or '-'):<24} "
            f"{(rec.bride.full_name or '-'):<24} "
            f"{year!s:>4}  "
            f"{source}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    service = IngestionService()
    item = service.repository.get_record(args.id)
    if item is None:
        print(f"Record {args.id} not found.", file=sys.stderr)
        return 1

    print(json.dumps(item.record.model_dump(mode="json"), indent=2))
    return 0


def cmd_rag_index(args: argparse.Namespace) -> int:
    from app.config import corpus_rag_index_dir, normalize_corpus
    from app.rag.service import RAGService

    corpus = normalize_corpus(args.corpus)
    service = RAGService(corpus=corpus)
    db_count = service.db_count()
    if db_count == 0:
        hints = {
            "docling": "Run `make import-batch` first.",
            "vlm": "Run `make build-vlm-corpus-cache` then `make ingest-vlm-corpus` first.",
            "qwen25": "Run `make scan-qwen25-corpus` first.",
        }
        hint = hints.get(corpus, "")
        print(f"No records in {corpus} database. {hint}", file=sys.stderr)
        return 1

    print(f"Indexing {db_count} record(s) into Chroma ({corpus})...")
    count = service.rebuild_index()
    print(f"Indexed {count} document(s) at {corpus_rag_index_dir(corpus)}")
    print(f"Vector index count: {service.indexed_count()}")
    return 0


def cmd_rag_ask(args: argparse.Namespace) -> int:
    from app.config import normalize_corpus
    from app.rag.service import RAGService

    corpus = normalize_corpus(args.corpus)
    service = RAGService(corpus=corpus)
    if service.db_count() == 0:
        print(f"No records in {corpus} database.", file=sys.stderr)
        return 1

    result = service.ask(args.question, k=args.k)
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for src in result.sources:
            print(
                f"  [{src['record_id']}] {src.get('groom') or '?'} / "
                f"{src.get('bride') or '?'} ({src.get('year') or '?'})"
            )
    return 0


def cmd_rag_search(args: argparse.Namespace) -> int:
    from app.config import normalize_corpus
    from app.rag.service import RAGService

    corpus = normalize_corpus(args.corpus)
    service = RAGService(corpus=corpus)
    if service.db_count() == 0:
        print(f"No records in {corpus} database.", file=sys.stderr)
        return 1

    hits = service.search(args.query, k=args.k)
    if not hits:
        print("No matches.")
        return 0

    for hit in hits:
        print(
            f"[{hit['record_id']}] {hit.get('groom') or '?'} / "
            f"{hit.get('bride') or '?'} ({hit.get('year') or '?'}) — "
            f"{hit.get('source_pdf') or '?'} p{hit.get('source_page') or '?'}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Marriage certificate ingestion and database tools"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser(
        "import-jsonl",
        help="Import pre-extracted records from a JSONL file (no LLM re-run)",
    )
    p_import.add_argument("path", type=Path, help="Path to marriage_records_*.jsonl")
    p_import.add_argument(
        "--force",
        action="store_true",
        help="Re-import even if source_pdf + source_page already exists",
    )
    p_import.set_defaults(func=cmd_import_jsonl)

    p_batch = sub.add_parser(
        "import-batch",
        help="Import all artifacts/extracted/marriage_records_*.jsonl files",
    )
    p_batch.add_argument(
        "--force",
        action="store_true",
        help="Re-import even if records already exist",
    )
    p_batch.set_defaults(func=cmd_import_batch)

    p_upload = sub.add_parser(
        "upload",
        help="Upload a new certificate (OCR + agent pipeline + save to DB)",
    )
    p_upload.add_argument("path", type=Path, help="PDF or image file")
    p_upload.add_argument(
        "--ocr",
        choices=["docling", "vlm"],
        help="OCR backend (default: OCR_BACKEND from .env or docling)",
    )
    p_upload.set_defaults(func=cmd_upload)

    p_gold = sub.add_parser(
        "run-gold",
        help="Run agent on gold-labeled pages and print field accuracy",
    )
    p_gold.add_argument(
        "--persist",
        action="store_true",
        help="Also save agent outputs to the database",
    )
    p_gold.add_argument(
        "--ocr",
        choices=["docling", "vlm"],
        help="Cached OCR source (default: OCR_BACKEND from .env or docling)",
    )
    p_gold.set_defaults(func=cmd_run_gold)

    p_vlm_cache = sub.add_parser(
        "build-vlm-cache",
        help="Transcribe gold pages with Qwen2-VL → artifacts/vlm/pages.jsonl",
    )
    p_vlm_cache.set_defaults(func=cmd_build_vlm_cache)

    p_vlm_corpus = sub.add_parser(
        "build-vlm-corpus-cache",
        help="Transcribe processed PDFs with Qwen2-VL → artifacts/vlm/corpus_pages.jsonl",
    )
    p_vlm_corpus.add_argument(
        "--force",
        action="store_true",
        help="Delete existing corpus cache and re-transcribe all pages",
    )
    p_vlm_corpus.set_defaults(func=cmd_build_vlm_corpus_cache)

    p_ingest_vlm = sub.add_parser(
        "ingest-vlm-corpus",
        help="Agent-extract cached VLM OCR into records_vlm.db",
    )
    p_ingest_vlm.add_argument(
        "--force",
        action="store_true",
        help="Re-process pages even if already in the VLM database",
    )
    p_ingest_vlm.set_defaults(func=cmd_ingest_vlm_corpus)

    p_qwen25 = sub.add_parser(
        "scan-qwen25-corpus",
        help="Zero-shot Qwen2.5-VL-7B scan of processed PDFs → records_qwen25_zeroshot.db (no agent)",
    )
    p_qwen25.add_argument(
        "--force",
        action="store_true",
        help="Restart the checkpoint JSONL and re-scan all pages",
    )
    p_qwen25.add_argument(
        "--limit",
        type=int,
        help="Only process the first N pending pages (smoke test)",
    )
    p_qwen25.set_defaults(func=cmd_scan_qwen25_corpus)

    p_gold_qwen25 = sub.add_parser(
        "run-gold-qwen25",
        help="Zero-shot Qwen2.5-VL-7B on the 14 gold pages → artifacts/agent/qwen-2.5-vl-7b/",
    )
    p_gold_qwen25.add_argument(
        "--name",
        help="Optional run folder name under artifacts/agent/qwen-2.5-vl-7b/plots/",
    )
    p_gold_qwen25.set_defaults(func=cmd_run_gold_qwen25)

    p_retranscribe = sub.add_parser(
        "retranscribe-vlm-page",
        help="Re-run Qwen2-VL on one corpus page and update corpus_pages.jsonl",
    )
    p_retranscribe.add_argument("--pdf", required=True, help="e.g. 1_content.pdf")
    p_retranscribe.add_argument("--page", type=int, required=True, help="1-based page number")
    p_retranscribe.set_defaults(func=cmd_retranscribe_vlm_page)

    p_gold_report = sub.add_parser(
        "gold-report",
        help="Build error CSVs from an existing agent predictions JSONL",
    )
    p_gold_report.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Predictions JSONL (default: artifacts/agent/<ocr>-gold-results/marriage_records_agent.jsonl)",
    )
    p_gold_report.add_argument(
        "--ocr",
        choices=["docling", "vlm"],
        help="Gold results folder (default: OCR_BACKEND from .env or docling)",
    )
    p_gold_report.add_argument(
        "--name",
        help="Optional run folder name under artifacts/agent/<backend>-gold-results/plots/",
    )
    p_gold_report.set_defaults(func=cmd_gold_report)

    p_list = sub.add_parser("list", help="List records in the database")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show one record as JSON")
    p_show.add_argument("id", type=int)
    p_show.set_defaults(func=cmd_show)

    p_rag_index = sub.add_parser(
        "rag-index",
        help="Build/rebuild vector index from ingested DB records (no OCR/agent)",
    )
    p_rag_index.add_argument(
        "--corpus",
        choices=["docling", "vlm", "qwen25"],
        default="docling",
        help="Which database/index to rebuild (default: docling)",
    )
    p_rag_index.set_defaults(func=cmd_rag_index)

    p_rag_ask = sub.add_parser(
        "rag-ask",
        help="Ask a question over indexed marriage records",
    )
    p_rag_ask.add_argument("question")
    p_rag_ask.add_argument("--k", type=int, default=5)
    p_rag_ask.add_argument(
        "--corpus",
        choices=["docling", "vlm", "qwen25"],
        default="docling",
    )
    p_rag_ask.set_defaults(func=cmd_rag_ask)

    p_rag_search = sub.add_parser(
        "rag-search",
        help="Semantic + structured search over ingested records",
    )
    p_rag_search.add_argument("query")
    p_rag_search.add_argument("--k", type=int, default=5)
    p_rag_search.add_argument(
        "--corpus",
        choices=["docling", "vlm", "qwen25"],
        default="docling",
    )
    p_rag_search.set_defaults(func=cmd_rag_search)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
