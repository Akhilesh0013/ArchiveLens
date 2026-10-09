"""Render PDF pages to PNG for VLM / vision pipelines."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def pdf_page_numbers(pdf_path: Path) -> list[int]:
    """Return 1-based page numbers for a PDF."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return list(range(1, len(pdf) + 1))
    finally:
        pdf.close()


def render_pdf_page(
    pdf_path: Path,
    page_no: int,
    dest_dir: Path,
    *,
    scale: float = 2.0,
    force: bool = False,
) -> Path:
    """Render a 1-based PDF page to ``dest_dir/{stem}_p{page:03d}.png``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pdf_path.stem}_p{page_no:03d}.png"
    if dest.exists() and not force:
        return dest

    pdf = pdfium.PdfDocument(str(pdf_path))
    page_index = page_no - 1
    if page_index < 0 or page_index >= len(pdf):
        pdf.close()
        raise ValueError(f"Page {page_no} out of range for {pdf_path.name}")

    image = pdf[page_index].render(scale=scale).to_pil()
    image.save(dest)
    pdf.close()
    return dest
