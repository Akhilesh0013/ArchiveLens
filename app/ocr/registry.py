"""Select OCR backend (docling or local VLM) from settings / CLI."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType

from app.config import settings

BACKENDS = {
    "docling": "app.ocr.docling",
    "vlm": "app.ocr.vlm",
}


def resolve_backend(name: str | None = None) -> str:
    backend = (name or settings.ocr_backend).strip().lower()
    if backend not in BACKENDS:
        supported = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown OCR backend {backend!r}. Supported: {supported}")
    return backend


def get_ocr_module(name: str | None = None) -> ModuleType:
    return import_module(BACKENDS[resolve_backend(name)])


def ocr_document(
    path: Path,
    *,
    page_range: tuple[int, int] | None = None,
    backend: str | None = None,
) -> list[tuple[int, str]]:
    return get_ocr_module(backend).ocr_document(path, page_range=page_range)


def ocr_engine_tag(backend: str | None = None) -> str:
    return get_ocr_module(backend).ocr_engine_tag()
