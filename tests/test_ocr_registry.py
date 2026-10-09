"""OCR backend registry tests."""

from __future__ import annotations

import pytest

from app.ocr.registry import resolve_backend


def test_resolve_backend_defaults_to_docling():
    assert resolve_backend(None) == "docling"
    assert resolve_backend("docling") == "docling"
    assert resolve_backend("vlm") == "vlm"


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown OCR backend"):
        resolve_backend("trocr")
