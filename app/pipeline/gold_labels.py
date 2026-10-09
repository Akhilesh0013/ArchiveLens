"""Load hand-labeled gold ground truth from ``gold/*.json``."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import GOLD_DIR


def iter_gold_label_paths() -> list[Path]:
    """Paths to labeled page JSON files (excludes ``_template.json``)."""
    return [
        path
        for path in sorted(GOLD_DIR.glob("*.json"))
        if not path.name.startswith("_")
    ]


def gold_label_keys() -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for path in iter_gold_label_paths():
        payload = json.loads(path.read_text())
        keys.append((payload["source_pdf"], int(payload["source_page"])))
    return keys


def load_gold_labels() -> dict[tuple[str, int], dict]:
    """Return ``{(source_pdf, page_no): label_dict}`` for evaluation."""
    gold: dict[tuple[str, int], dict] = {}
    for path in iter_gold_label_paths():
        payload = json.loads(path.read_text())
        key = (payload["source_pdf"], int(payload["source_page"]))
        gold[key] = payload
    return gold
