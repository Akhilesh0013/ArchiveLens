"""Docling + RapidOCR text extraction for uploaded documents."""

from __future__ import annotations

from pathlib import Path

from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.config import OCR_ENGINE_TAG, settings
from app.ocr.cleanup import clean_ocr_text


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions(
        lang=["english"],
        force_full_page_ocr=settings.docling_force_full_page_ocr,
        backend="onnxruntime",
    )
    return DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )


def ocr_document(
    path: Path,
    *,
    page_range: tuple[int, int] | None = None,
) -> list[tuple[int, str]]:
    """OCR a PDF or image file. Returns ``[(page_no, text), ...]`` (1-based pages)."""
    converter = build_converter()
    kwargs: dict = {}
    if page_range is not None:
        kwargs["page_range"] = page_range

    result = converter.convert(str(path), **kwargs)
    doc = result.document
    return [
        (page_no, clean_ocr_text(doc.export_to_text(page_no=page_no)))
        for page_no in sorted(doc.pages)
    ]


def ocr_engine_tag() -> str:
    return OCR_ENGINE_TAG
