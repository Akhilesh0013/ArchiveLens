"""Upload and import pipelines that persist records to SQLite."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import EXTRACTED_DIR, UPLOADS_DIR, corpus_db_path
from app.db.repository import RecordRepository, StoredRecord
from app.pipeline.orchestrator import CorpusIngestResult, ExtractionOrchestrator, PageAgentResult
from app.schema import MarriageRecord

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


# Backwards-compatible alias used by CLI / Streamlit
PageResult = PageAgentResult


@dataclass
class IngestionResult:
    upload_id: int
    stored_path: Path
    pages: list[PageAgentResult] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)


class IngestionService:
    """OCR + agent extraction + store marriage certificates."""

    def __init__(
        self,
        repository: RecordRepository | None = None,
        orchestrator: ExtractionOrchestrator | None = None,
        uploads_dir: Path | None = None,
    ) -> None:
        self.repository = repository or RecordRepository()
        self.orchestrator = orchestrator or ExtractionOrchestrator(self.repository)
        self.uploads_dir = uploads_dir or UPLOADS_DIR
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def repository_for_backend(self, backend: str | None = None) -> RecordRepository:
        from app.ocr.registry import resolve_backend

        source = resolve_backend(backend)
        if source == "vlm":
            return RecordRepository(corpus_db_path("vlm"))
        return self.repository

    def orchestrator_for_backend(self, backend: str | None = None) -> ExtractionOrchestrator:
        return ExtractionOrchestrator(self.repository_for_backend(backend))

    def ingest_file(
        self,
        source: Path | bytes,
        filename: str,
        *,
        skip_empty_pages: bool = True,
        ocr_backend: str | None = None,
    ) -> IngestionResult:
        """Upload a certificate, run the agent pipeline per page, save to DB."""
        stored_path = self._persist_upload(source, filename)
        backend = ocr_backend
        orchestrator = self.orchestrator_for_backend(backend)
        result = orchestrator.process_file(
            stored_path,
            skip_empty_pages=skip_empty_pages,
            persist=True,
            ocr_backend=backend,
        )

        if not result.pages:
            raise ValueError("No pages extracted")

        return IngestionResult(
            upload_id=result.pages[0].upload_id,
            stored_path=result.stored_path,
            pages=result.pages,
            skipped_pages=result.skipped_pages,
        )

    def run_gold_eval(self, *, persist: bool = False, ocr_source: str | None = None):
        return self.orchestrator.process_gold_set(
            persist=persist,
            ocr_source=ocr_source,
        )

    def build_vlm_corpus_cache(self, *, force: bool = False) -> int:
        from app.ocr.vlm import build_corpus_vlm_cache

        return build_corpus_vlm_cache(force=force)

    def ingest_vlm_corpus(self, *, skip_existing: bool = True) -> CorpusIngestResult:
        orchestrator = ExtractionOrchestrator(RecordRepository(corpus_db_path("vlm")))
        return orchestrator.process_vlm_corpus(skip_existing=skip_existing)

    def import_jsonl(
        self,
        jsonl_path: Path,
        *,
        skip_existing: bool = True,
    ) -> int:
        """Import pre-extracted records from a notebook JSONL file (no LLM re-run)."""
        imported = 0

        with jsonl_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                record = MarriageRecord.model_validate(payload)

                source_pdf = record.source_pdf or jsonl_path.name
                source_page = record.source_page or 0

                if skip_existing and self.repository.find_by_provenance(
                    source_pdf, source_page, record.ocr_engine
                ):
                    continue

                upload_id = self.repository.create_upload(
                    filename=f"{source_pdf}#p{source_page}",
                    stored_path=str(jsonl_path),
                    ocr_engine=record.ocr_engine,
                    status="imported",
                )
                self.repository.save_record(upload_id, record)
                imported += 1

        return imported

    def import_default_batch(
        self,
        *,
        skip_existing: bool = True,
        docling_only: bool = False,
    ) -> dict[str, int]:
        """Import notebook JSONL batches into SQLite.

        By default imports all ``marriage_records_*.jsonl`` files. Set
        ``docling_only=True`` to load a single canonical OCR engine for RAG.
        """
        counts: dict[str, int] = {}
        paths = sorted(EXTRACTED_DIR.glob("marriage_records_*.jsonl"))
        if docling_only:
            paths = [p for p in paths if p.name == "marriage_records_docling.jsonl"]
        for path in paths:
            counts[path.name] = self.import_jsonl(path, skip_existing=skip_existing)
        return counts

    def list_records(self, limit: int = 100) -> list[StoredRecord]:
        return self.repository.list_records(limit=limit)

    def _persist_upload(self, source: Path | bytes, filename: str) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}"
            )

        dest = self.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
        if isinstance(source, Path):
            shutil.copy2(source, dest)
        else:
            dest.write_bytes(source)
        return dest
