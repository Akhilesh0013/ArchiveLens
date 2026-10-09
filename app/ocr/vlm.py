"""Local VLM page transcription (Qwen2-VL via mlx-vlm) as an OCR backend."""

from __future__ import annotations

import json
import logging
import warnings
from functools import lru_cache
from pathlib import Path

from app.config import GOLD_DIR, PROCESSED_DIR, PROCESSED_PDF_GLOB, RAW_DIR, VLM_CORPUS_PAGES_JSONL, VLM_DIR, VLM_ENGINE_TAG, VLM_GOLD_PAGES_JSONL, settings
from app.ocr.cleanup import clean_ocr_text
from app.ocr.render import IMAGE_SUFFIXES, pdf_page_numbers, render_pdf_page

# Legacy alias — prefer ``VLM_GOLD_PAGES_JSONL`` from ``app.config``.
VLM_PAGES_JSONL = VLM_GOLD_PAGES_JSONL

VLM_TRANSCRIBE_PROMPT = """\
Transcribe all printed and handwritten text visible on this Maryland marriage \
record page (license or Department of Health Report of Marriage form).
Output plain text only. Preserve line breaks and top-to-bottom reading order.
Copy spelling, abbreviations, and punctuation exactly as shown on the form.
Include all filled-in names, dates, residences, and signatures — not just headers.
Do not summarize, structure into JSON, or add commentary.
"""


def _quiet_transformers() -> None:
    """mlx-vlm triggers harmless transformers processor_kwargs warnings each page."""
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*processor_kwargs.*",
        category=UserWarning,
    )


@lru_cache(maxsize=1)
def _load_model():
    _quiet_transformers()
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise ImportError(
            "mlx-vlm is required for OCR_BACKEND=vlm. Install with: uv sync"
        ) from exc

    model_path = settings.vlm_model_path
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config, generate, apply_chat_template


def ocr_engine_tag() -> str:
    return VLM_ENGINE_TAG


def transcribe_image(image_path: Path) -> str:
    """Run Qwen2-VL on one page image and return plain text."""
    model, processor, config, generate, apply_chat_template = _load_model()
    formatted_prompt = apply_chat_template(
        processor,
        config,
        VLM_TRANSCRIBE_PROMPT,
        num_images=1,
    )
    result = generate(
        model,
        processor,
        formatted_prompt,
        [str(image_path)],
        max_tokens=settings.vlm_max_tokens,
        temperature=0.0,
        verbose=False,
    )
    text = result.text if hasattr(result, "text") else str(result)
    return clean_ocr_text(text.strip())


def ocr_document(
    path: Path,
    *,
    page_range: tuple[int, int] | None = None,
) -> list[tuple[int, str]]:
    """OCR a PDF or image file. Returns ``[(page_no, text), ...]`` (1-based pages)."""
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [(1, transcribe_image(path))]

    page_numbers = pdf_page_numbers(path)

    if page_range is not None:
        start, end = page_range
        page_numbers = [p for p in page_numbers if start <= p <= end]

    render_dir = VLM_DIR / "_rendered"
    pages: list[tuple[int, str]] = []
    for page_no in page_numbers:
        image_path = render_pdf_page(path, page_no, render_dir)
        pages.append((page_no, transcribe_image(image_path)))
    return pages


def append_cache_row(
    source_pdf: str,
    source_page: int,
    text: str,
    *,
    jsonl_path: Path | None = None,
) -> None:
    path = jsonl_path or VLM_PAGES_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source_pdf": source_pdf,
        "source_page": source_page,
        "text": text,
        "engine": VLM_ENGINE_TAG,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def upsert_corpus_cache_row(
    source_pdf: str,
    source_page: int,
    text: str,
    *,
    jsonl_path: Path | None = None,
) -> None:
    """Replace one page's OCR text in the corpus JSONL (rewrite file)."""
    from app.pipeline.ocr_loader import load_vlm_pages

    path = jsonl_path or VLM_CORPUS_PAGES_JSONL
    pages = load_vlm_pages(path)
    pages[(source_pdf, source_page)] = clean_ocr_text(text.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for (pdf_name, page_no), page_text in sorted(pages.items()):
            row = {
                "source_pdf": pdf_name,
                "source_page": page_no,
                "text": page_text,
                "engine": VLM_ENGINE_TAG,
            }
            handle.write(json.dumps(row) + "\n")


def retranscribe_corpus_page(
    source_pdf: str,
    source_page: int,
    *,
    render_dir: Path | None = None,
    jsonl_path: Path | None = None,
    render_scale: float = 3.0,
    force_render: bool = True,
) -> str:
    """Re-run Qwen2-VL on one processed PDF page and update the corpus cache."""
    pdf_path = resolve_pdf(source_pdf) or (PROCESSED_DIR / source_pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {source_pdf}")

    render_root = render_dir or (VLM_DIR / "_rendered_corpus")
    image_path = render_pdf_page(
        pdf_path,
        source_page,
        render_root,
        scale=render_scale,
        force=force_render,
    )
    print(
        f"Transcribing {source_pdf} p{source_page} "
        f"(scale={render_scale}) …",
        flush=True,
    )
    text = transcribe_image(image_path)
    upsert_corpus_cache_row(
        source_pdf, source_page, text, jsonl_path=jsonl_path or VLM_CORPUS_PAGES_JSONL
    )
    return text


def resolve_pdf(source_pdf: str) -> Path | None:
    for directory in (PROCESSED_DIR, RAW_DIR):
        candidate = directory / source_pdf
        if candidate.exists():
            return candidate
    return None


def ensure_vlm_gold_cache() -> None:
    """Ensure ``artifacts/vlm/pages.jsonl`` covers all gold-labeled pages."""
    from app.pipeline.gold_labels import gold_label_keys
    from app.pipeline.ocr_loader import load_vlm_gold_pages

    keys = gold_label_keys()
    if not keys:
        return

    cached = load_vlm_gold_pages()
    missing = [key for key in keys if not cached.get(key, "").strip()]
    if not missing:
        return

    print(
        f"VLM gold cache: {len(keys) - len(missing)}/{len(keys)} pages — "
        f"building {len(missing)} missing (corpus copy or transcription)...",
        flush=True,
    )
    build_gold_vlm_cache()

    cached_after = load_vlm_gold_pages()
    still_missing = [key for key in keys if not cached_after.get(key, "").strip()]
    if still_missing:
        sample = "\n".join(f"  {pdf} p{page}" for pdf, page in still_missing[:8])
        raise ValueError(
            f"Could not build VLM gold cache for {len(still_missing)} page(s):\n{sample}"
        )


def build_gold_vlm_cache(*, render_dir: Path | None = None) -> int:
    """Build ``artifacts/vlm/pages.jsonl`` for the 14 gold-labeled pages only.

    Reuses matching rows from ``corpus_pages.jsonl`` when present; otherwise
    runs Qwen2-VL transcription for missing gold pages.
    """
    from app.pipeline.gold_labels import iter_gold_label_paths
    from app.pipeline.ocr_loader import load_vlm_pages

    corpus = load_vlm_pages(VLM_CORPUS_PAGES_JSONL)

    if VLM_GOLD_PAGES_JSONL.exists():
        VLM_GOLD_PAGES_JSONL.unlink()

    render_root = render_dir or (VLM_DIR / "_rendered")
    copied = 0
    transcribed = 0

    for gold_path in iter_gold_label_paths():
        label = json.loads(gold_path.read_text())
        source_pdf = label["source_pdf"]
        source_page = int(label["source_page"])
        key = (source_pdf, source_page)

        if key in corpus and corpus[key].strip():
            append_cache_row(
                source_pdf,
                source_page,
                corpus[key],
                jsonl_path=VLM_GOLD_PAGES_JSONL,
            )
            copied += 1
            continue

        stem = Path(source_pdf).stem
        image_candidate = GOLD_DIR / "images" / f"{stem}_p{source_page:03d}.png"
        if image_candidate.exists():
            image_path = image_candidate
        else:
            pdf_path = resolve_pdf(source_pdf)
            if pdf_path is None:
                continue
            image_path = render_pdf_page(pdf_path, source_page, render_root)

        print(f"  Transcribing gold page {source_pdf} p{source_page} …", flush=True)
        text = transcribe_image(image_path)
        append_cache_row(
            source_pdf,
            source_page,
            text,
            jsonl_path=VLM_GOLD_PAGES_JSONL,
        )
        transcribed += 1

    if copied:
        print(f"  Copied {copied} gold page(s) from {VLM_CORPUS_PAGES_JSONL.name}", flush=True)
    if transcribed:
        print(f"  Transcribed {transcribed} gold page(s) with Qwen2-VL", flush=True)

    return copied + transcribed


def build_corpus_vlm_cache(*, render_dir: Path | None = None, force: bool = False) -> int:
    """Transcribe all pages in processed PDFs → ``artifacts/vlm/corpus_pages.jsonl``."""
    from app.pipeline.ocr_loader import load_vlm_pages

    if force and VLM_CORPUS_PAGES_JSONL.exists():
        VLM_CORPUS_PAGES_JSONL.unlink()

    existing = load_vlm_pages(VLM_CORPUS_PAGES_JSONL)
    render_root = render_dir or (VLM_DIR / "_rendered_corpus")
    written = 0

    pdf_paths = sorted(PROCESSED_DIR.glob(PROCESSED_PDF_GLOB))
    total_pages = sum(len(pdf_page_numbers(path)) for path in pdf_paths)
    already_cached = len(existing)
    remaining = max(total_pages - already_cached, 0)

    print(
        f"VLM corpus: {already_cached}/{total_pages} pages cached, "
        f"{remaining} remaining (Qwen2-VL via mlx-vlm, ~1–2 min/page)",
        flush=True,
    )

    for pdf_path in pdf_paths:
        page_numbers = pdf_page_numbers(pdf_path)

        pdf_name = pdf_path.name
        for page_no in page_numbers:
            key = (pdf_name, page_no)
            if key in existing:
                continue

            done = already_cached + written + 1
            print(f"  [{done}/{total_pages}] {pdf_name} p{page_no} …", flush=True)
            image_path = render_pdf_page(pdf_path, page_no, render_root)
            text = transcribe_image(image_path)
            append_cache_row(
                pdf_name,
                page_no,
                text,
                jsonl_path=VLM_CORPUS_PAGES_JSONL,
            )
            existing[key] = text
            written += 1

    return written
