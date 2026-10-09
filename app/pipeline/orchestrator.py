"""Orchestrate OCR → LangGraph agent → database for marriage certificates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.agents import run_extraction_agent
from app.agents.state import GraphState
from app.config import (
    ARTIFACTS_DIR,
    GOLD_DIR,
    PROCESSED_DIR,
    UPLOADS_DIR,
    VLM_CORPUS_PAGES_JSONL,
    gold_predictions_path,
)
from app.db.repository import RecordRepository
from app.ocr.registry import ocr_document, ocr_engine_tag, resolve_backend
from app.ocr.render import render_pdf_page
from app.pipeline.eval import evaluate_predictions
from app.pipeline.gold_plots import publish_gold_run
from app.pipeline.gold_labels import iter_gold_label_paths
from app.pipeline.ocr_loader import load_cached_pages, load_vlm_pages
from app.ocr.vlm import resolve_pdf
from app.schema import MarriageRecord, normalize_marriage_record

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


@dataclass
class PageAgentResult:
    page_no: int
    record_id: int
    record: MarriageRecord
    upload_id: int
    judge_verdict: str | None = None
    judge_reason: str | None = None
    validation_errors: list[str] = field(default_factory=list)
    retries: int = 0
    image_path: str | None = None


@dataclass
class FileProcessResult:
    stored_path: Path
    pages: list[PageAgentResult] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)


@dataclass
class GoldEvalResult:
    predictions: dict[tuple[str, int], MarriageRecord]
    metrics: dict[str, float | int]
    output_path: Path
    run_dir: Path | None = None
    report_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class CorpusIngestResult:
    processed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class ExtractionOrchestrator:
    """Run the agent pipeline per page and persist results."""

    def __init__(
        self,
        repository: RecordRepository | None = None,
        *,
        render_dir: Path | None = None,
        agent_output_dir: Path | None = None,
    ) -> None:
        self.repository = repository or RecordRepository()
        self.render_dir = render_dir or (UPLOADS_DIR / "_rendered")
        self.agent_output_dir = agent_output_dir or (ARTIFACTS_DIR / "agent")
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.agent_output_dir.mkdir(parents=True, exist_ok=True)

    def gold_image_path(self, source_pdf: str, page_no: int) -> Path | None:
        stem = Path(source_pdf).stem
        candidate = GOLD_DIR / "images" / f"{stem}_p{page_no:03d}.png"
        return candidate if candidate.exists() else None

    def resolve_image_path(
        self,
        stored_path: Path,
        page_no: int,
        *,
        source_pdf: str | None = None,
    ) -> Path | None:
        if stored_path.suffix.lower() in IMAGE_SUFFIXES:
            return stored_path

        pdf_name = source_pdf or stored_path.name
        gold_image = self.gold_image_path(pdf_name, page_no)
        if gold_image is not None:
            return gold_image

        return self._render_page_image(stored_path, page_no)

    def process_page(
        self,
        ocr_text: str,
        *,
        source_pdf: str,
        source_page: int,
        image_path: Path | str | None = None,
        ocr_engine: str | None = None,
        ocr_backend: str | None = None,
        persist: bool = True,
        stored_path: str | None = None,
        use_judge: bool = True,
    ) -> tuple[GraphState, PageAgentResult | None]:
        backend = resolve_backend(ocr_backend)
        engine = ocr_engine or ocr_engine_tag(backend)
        state = run_extraction_agent(
            ocr_text,
            source_pdf=source_pdf,
            source_page=source_page,
            ocr_engine=engine,
            use_judge=use_judge,
        )

        record = state.get("final_record")
        if record is None:
            raise ValueError("Agent finished without a final_record")

        page_result: PageAgentResult | None = None
        if persist:
            upload_id = self.repository.create_upload(
                filename=f"{source_pdf}#p{source_page}",
                stored_path=stored_path or str(image_path or source_pdf),
                ocr_engine=engine,
                status=state.get("status", "extracted"),
            )
            record_id = self.repository.save_record(upload_id, record)
            page_result = PageAgentResult(
                page_no=source_page,
                record_id=record_id,
                record=record,
                upload_id=upload_id,
                judge_verdict=state.get("judge_verdict"),
                judge_reason=state.get("judge_reason"),
                validation_errors=list(state.get("validation_errors") or []),
                retries=int(state.get("retries") or 0),
                image_path=str(image_path) if image_path else None,
            )

        return state, page_result

    def process_file(
        self,
        file_path: Path,
        *,
        skip_empty_pages: bool = True,
        persist: bool = True,
        ocr_backend: str | None = None,
    ) -> FileProcessResult:
        backend = resolve_backend(ocr_backend)
        pages = ocr_document(file_path, backend=backend)
        if not pages:
            raise ValueError("OCR returned no pages")

        results: list[PageAgentResult] = []
        skipped: list[int] = []

        for page_no, ocr_text in pages:
            if skip_empty_pages and not ocr_text.strip():
                skipped.append(page_no)
                continue

            image = self.resolve_image_path(
                file_path, page_no, source_pdf=file_path.name
            )
            _, page_result = self.process_page(
                ocr_text,
                source_pdf=file_path.name,
                source_page=page_no,
                image_path=image,
                persist=persist,
                stored_path=str(file_path),
                ocr_backend=backend,
                use_judge=False,
            )
            if page_result is not None:
                results.append(page_result)

        if not results:
            raise ValueError("All pages were empty after OCR")

        return FileProcessResult(
            stored_path=file_path,
            pages=results,
            skipped_pages=skipped,
        )

    def process_gold_set(
        self,
        *,
        ocr_source: str | None = None,
        persist: bool = False,
    ) -> GoldEvalResult:
        """Run the agent on all gold-labeled pages using cached OCR text."""
        source = resolve_backend(ocr_source)
        ocr_pages = load_cached_pages(source)

        predictions: dict[tuple[str, int], MarriageRecord] = {}
        output_path = gold_predictions_path(source)

        with output_path.open("w") as handle:
            for gold_path in iter_gold_label_paths():
                label = json.loads(gold_path.read_text())
                key = (label["source_pdf"], int(label["source_page"]))
                ocr_text = ocr_pages[key]

                image = self.gold_image_path(key[0], key[1])
                state, _ = self.process_page(
                    ocr_text,
                    source_pdf=key[0],
                    source_page=key[1],
                    image_path=image,
                    ocr_engine=f"agent+{ocr_engine_tag(source)}",
                    ocr_backend=source,
                    persist=persist,
                    stored_path=str(image) if image else str(gold_path),
                    use_judge=True,
                )
                record = normalize_marriage_record(state["final_record"])
                predictions[key] = record
                handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

        metrics = evaluate_predictions(predictions)
        run_dir, report_paths = publish_gold_run(predictions, metrics, ocr_source=source)

        return GoldEvalResult(
            predictions=predictions,
            metrics=metrics,
            output_path=output_path,
            run_dir=run_dir,
            report_paths=report_paths,
        )

    def process_vlm_corpus(
        self,
        *,
        skip_existing: bool = True,
        jsonl_path: Path | None = None,
    ) -> CorpusIngestResult:
        """Run the agent on cached VLM OCR for all processed PDF pages."""
        cache_path = jsonl_path or VLM_CORPUS_PAGES_JSONL
        ocr_pages = load_vlm_pages(cache_path)
        if not ocr_pages:
            raise ValueError(
                f"No VLM corpus OCR cache at {cache_path}. "
                "Run: uv run python -m app.cli build-vlm-corpus-cache"
            )

        engine = f"agent+{ocr_engine_tag('vlm')}"
        result = CorpusIngestResult()
        total_pages = len(ocr_pages)
        already_in_db = 0
        pending: list[tuple[str, int, str, Path]] = []

        for (source_pdf, source_page), ocr_text in sorted(ocr_pages.items()):
            if skip_existing and self.repository.find_by_provenance(
                source_pdf, source_page, engine
            ):
                result.skipped += 1
                already_in_db += 1
                continue

            if not ocr_text.strip():
                result.skipped += 1
                continue

            pdf_path = resolve_pdf(source_pdf) or (PROCESSED_DIR / source_pdf)
            if not pdf_path.exists():
                result.errors.append(f"{source_pdf} p{source_page}: PDF not found")
                continue

            pending.append((source_pdf, source_page, ocr_text, pdf_path))

        print(
            f"VLM ingest: {already_in_db}/{total_pages} already in DB, "
            f"{len(pending)} to process (extract only — judge skipped)",
            flush=True,
        )

        for index, (source_pdf, source_page, ocr_text, pdf_path) in enumerate(
            pending, start=1
        ):
            print(
                f"  [{index}/{len(pending)}] {source_pdf} p{source_page} …",
                flush=True,
            )
            try:
                image = self.resolve_image_path(
                    pdf_path, source_page, source_pdf=source_pdf
                )
                _, page_result = self.process_page(
                    ocr_text,
                    source_pdf=source_pdf,
                    source_page=source_page,
                    image_path=image,
                    ocr_engine=engine,
                    ocr_backend="vlm",
                    persist=True,
                    stored_path=str(pdf_path),
                    use_judge=False,
                )
                result.processed += 1
                if page_result is not None:
                    groom = page_result.record.groom.full_name or "?"
                    bride = page_result.record.bride.full_name or "?"
                    print(
                        f"       saved record #{page_result.record_id} — "
                        f"{groom} / {bride}",
                        flush=True,
                    )
            except Exception as exc:
                result.errors.append(f"{source_pdf} p{source_page}: {exc}")
                print(f"       failed — {exc}", flush=True)

        return result

    def _render_page_image(self, pdf_path: Path, page_no: int) -> Path:
        return render_pdf_page(pdf_path, page_no, self.render_dir)
