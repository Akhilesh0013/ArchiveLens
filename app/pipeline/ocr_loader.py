"""Load cached OCR text for gold eval and corpus pipelines."""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from app.config import DOCLING_DIR, GOLD_DIR, VLM_GOLD_PAGES_JSONL
from app.ocr.cleanup import clean_ocr_text
from app.pipeline.gold_labels import gold_label_keys

SUPPORTED_OCR_SOURCES = ("docling", "vlm")


def load_docling_pages() -> dict[tuple[str, int], str]:
    """Return ``{(pdf_name, page_no): ocr_text}`` from Docling JSON exports."""
    pages: dict[tuple[str, int], str] = {}

    for json_path in sorted(DOCLING_DIR.glob("*_content.json")):
        doc = DoclingDocument.load_from_json(json_path)
        pdf_name = f"{json_path.stem}.pdf"
        for page_no in sorted(doc.pages):
            raw = doc.export_to_text(page_no=page_no)
            pages[(pdf_name, page_no)] = clean_ocr_text(raw)

    return pages


def load_vlm_pages(jsonl_path: Path | None = None) -> dict[tuple[str, int], str]:
    """Return ``{(pdf_name, page_no): ocr_text}`` from one VLM cache JSONL."""
    path = jsonl_path or VLM_GOLD_PAGES_JSONL
    pages: dict[tuple[str, int], str] = {}
    if not path.exists():
        return pages

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["source_pdf"], int(rec["source_page"]))
            pages[key] = clean_ocr_text(rec["text"])
    return pages


def load_vlm_gold_pages() -> dict[tuple[str, int], str]:
    """Gold eval VLM OCR — ``artifacts/vlm/pages.jsonl`` (14 labeled pages)."""
    return load_vlm_pages(VLM_GOLD_PAGES_JSONL)


def load_gold_ocr_pages(source: str) -> dict[tuple[str, int], str]:
    """OCR text for gold eval only — mirrors docling subset-over-full-corpus pattern.

    Docling: reads ``artifacts/docling/*_content.json``, returns the 14 gold pages.
    VLM: ensures ``artifacts/vlm/pages.jsonl`` exists (auto-builds if needed).
    """
    if source == "docling":
        if not any(DOCLING_DIR.glob("*_content.json")):
            raise ValueError(
                f"No Docling OCR artifacts in {DOCLING_DIR}. "
                "Run the docling notebook export first."
            )
        pages = load_docling_pages()
        ocr_ref = str(DOCLING_DIR)
    elif source == "vlm":
        from app.ocr.vlm import ensure_vlm_gold_cache

        ensure_vlm_gold_cache()
        pages = load_vlm_gold_pages()
        ocr_ref = str(VLM_GOLD_PAGES_JSONL)
    else:
        raise ValueError(
            f"Unknown OCR source {source!r}. Supported: {', '.join(SUPPORTED_OCR_SOURCES)}"
        )

    keys = gold_label_keys()
    if not keys:
        raise ValueError(f"No gold labels found in {GOLD_DIR}/*.json")

    missing = [key for key in keys if not pages.get(key, "").strip()]
    if missing:
        sample = "\n".join(f"  {pdf} p{page}" for pdf, page in missing[:8])
        extra = f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise ValueError(
            f"Missing OCR for {len(missing)}/{len(keys)} gold page(s) ({ocr_ref}):\n"
            f"{sample}{extra}"
        )

    return {key: pages[key] for key in keys}


def load_cached_pages(source: str) -> dict[tuple[str, int], str]:
    """Load OCR pages for gold eval (alias for ``load_gold_ocr_pages``)."""
    return load_gold_ocr_pages(source)


def load_trocr_pages(trocr_jsonl: Path) -> dict[tuple[str, int], str]:
    pages: dict[tuple[str, int], str] = {}
    with trocr_jsonl.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pages[(rec["source_pdf"], rec["page"])] = rec["text"]
    return pages
