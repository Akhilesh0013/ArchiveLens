"""Experimental DB text detection and layout-preserving region packing.

Only the detector is loaded. No RapidOCR recognizer or orientation classifier.
The original image supplies pixels; detection runs on a smaller image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from PIL import Image

DETECTOR_SHA256 = "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9"
DETECTOR_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/"
    "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx"
)


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    right: int
    bottom: int


def regions_from_boxes(boxes, size, *, padding=12, merge_gap=16):
    """Keep all valid detections; merge vertically overlapping text into bands.

    Every band uses the same horizontal extent, preserving column alignment.
    These are text regions, not semantic field labels. Detection can miss ink.
    """
    width, height = size
    if padding < 0 or merge_gap < 0:
        raise ValueError("padding and merge_gap must be nonnegative")
    bounds = []
    for box in boxes:
        points = np.asarray(box, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        left = max(0, math.floor(points[:, 0].min()) - padding)
        right = min(width, math.ceil(points[:, 0].max()) + padding)
        top = max(0, math.floor(points[:, 1].min()) - padding)
        bottom = min(height, math.ceil(points[:, 1].max()) + padding)
        if left < right and top < bottom:
            bounds.append(Region(left, top, right, bottom))
    if not bounds:
        return []
    left = min(b.left for b in bounds)
    right = max(b.right for b in bounds)
    bands = []
    for b in sorted(bounds, key=lambda r: (r.top, r.left)):
        if bands and b.top <= bands[-1][1] + merge_gap:
            bands[-1][1] = max(bands[-1][1], b.bottom)
        else:
            bands.append([b.top, b.bottom])
    return [Region(left, top, right, bottom) for top, bottom in bands]


def preserve_inked_gaps(image, regions, *, ink_threshold=245):
    """Bridge gaps containing any dark pixel so undetected strokes are retained.

    Conservative: form rules/noise may prevent compression. This checks only
    internal gaps; it is not a guarantee of detector recall at outer margins.
    """
    if not regions:
        return []
    gray = np.asarray(image.convert("L"))
    merged = [regions[0]]
    for region in regions[1:]:
        previous = merged[-1]
        left = min(previous.left, region.left)
        right = max(previous.right, region.right)
        gap = gray[previous.bottom:region.top, left:right]
        if region.top <= previous.bottom or (gap.size and np.any(gap < ink_threshold)):
            merged[-1] = Region(left, previous.top, right, max(previous.bottom, region.bottom))
        else:
            merged.append(region)
    return merged


def pack_regions(image, regions, *, gutter=12):
    """Remove blank margins/gaps without scaling text; return image and fallback."""
    if gutter < 0:
        raise ValueError("gutter must be nonnegative")
    if not regions:
        return image.copy(), "no_detections"
    width = max(r.right - r.left for r in regions)
    height = sum(r.bottom - r.top for r in regions) + gutter * (len(regions) - 1)
    if width * height >= image.width * image.height:
        return image.copy(), "packing_not_smaller"
    packed = Image.new("RGB", (width, height), "white")
    y = 0
    for r in regions:
        crop = image.crop((r.left, r.top, r.right, r.bottom))
        packed.paste(crop, (0, y))
        y += crop.height + gutter
    return packed, None


class DBNetDetector:
    """Pinned PP-OCRv4 mobile DB detector, ONNX Runtime CPU backend."""

    def __init__(self, model_path: Path, *, max_side=960, box_threshold=0.5):
        import hashlib
        import rapidocr
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.utils.parse_parameters import ParseParams
        from rapidocr.utils.typings import ModelType, OCRVersion

        if max_side < 32 or not 0 <= box_threshold <= 1:
            raise ValueError("invalid detector size or threshold")
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != DETECTOR_SHA256:
            raise ValueError("Detector checksum does not match pinned PP-OCRv4 mobile model")
        cfg = ParseParams.load(Path(rapidocr.__file__).parent / "config.yaml")
        cfg = ParseParams.update_batch(cfg, {
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.model_path": str(model_path.resolve()),
            "Det.limit_type": "min",
            "Det.limit_side_len": 32,
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.std": [0.229, 0.224, 0.225],
            "Det.box_thresh": box_threshold,
        })
        cfg.Det.engine_cfg = cfg.EngineConfig.onnxruntime
        cfg.Det.engine_cfg.intra_op_num_threads = 4
        self.detector = TextDetector(cfg.Det)
        self.max_side = max_side

    def __call__(self, image):
        scale = min(1.0, self.max_side / max(image.size))
        size = (max(32, round(image.width * scale)), max(32, round(image.height * scale)))
        small = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        # RapidOCR's image loader normally supplies BGR; we bypass that loader.
        result = self.detector(np.asarray(small)[:, :, ::-1].copy())
        if result.boxes is None:
            return np.empty((0, 4, 2), dtype=float)
        boxes = np.asarray(result.boxes, dtype=float).copy()
        boxes[:, :, 0] *= image.width / small.width
        boxes[:, :, 1] *= image.height / small.height
        return boxes
