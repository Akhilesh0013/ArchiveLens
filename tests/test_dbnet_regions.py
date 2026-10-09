"""Geometry and scoring checks for the isolated DBNet experiment."""
import numpy as np
from PIL import Image

from app.ocr.dbnet_regions import Region, pack_regions, preserve_inked_gaps, regions_from_boxes
from scripts.benchmark_dbnet_qwen import score, summarize
from app.schema import MarriageRecord


def quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_columns_remain_aligned_and_all_detected_ink_survives():
    boxes = [quad(15, 10, 30, 20), quad(70, 12, 90, 21), quad(60, 80, 80, 90)]
    regions = regions_from_boxes(boxes, (100, 120), padding=2, merge_gap=4)
    assert regions == [Region(13, 8, 92, 23), Region(13, 78, 92, 92)]
    image = Image.new("RGB", (100, 120), "white")
    image.putpixel((20, 15), (0, 0, 0))
    image.putpixel((75, 85), (0, 0, 0))
    packed, fallback = pack_regions(image, regions, gutter=3)
    assert fallback is None
    assert packed.getpixel((7, 7)) == (0, 0, 0)
    assert packed.getpixel((62, 25)) == (0, 0, 0)
    assert np.sum(np.asarray(packed) == 0) == 6


def test_missing_boxes_fall_back_without_discarding_page():
    image = Image.new("RGB", (100, 100), "gray")
    packed, reason = pack_regions(image, [])
    assert reason == "no_detections"
    assert packed.tobytes() == image.tobytes()


def test_clips_bounds_and_ignores_invalid_boxes():
    regions = regions_from_boxes(
        [quad(-5, -10, 110, 50), quad(150, 150, 160, 160), [[float("nan"), 0]] * 4],
        (100, 100), padding=0)
    assert regions == [Region(0, 0, 100, 50)]


def test_failed_page_is_scored_as_missing_not_excluded():
    gold = {"groom": {"full_name": "NAME"}, "bride": {"full_name": "NAME"}}
    outcomes = score(gold, MarriageRecord())
    assert outcomes["missed"] == 2
    rows = [{"mode": "full_page", "outcomes": outcomes, "error": "bad JSON",
             "pipeline_seconds": 1, "detector_seconds": 0, "prompt_tokens": 12,
             "generation_tokens": 1, "mlx_peak_gb": 2, "pixel_ratio": 1}]
    result = summarize(rows)["full_page"]
    assert result["n_pages"] == 1
    assert result["failed_pages"] == 1
    assert result["field_accuracy"] == 0


def test_undetected_signature_in_gap_is_preserved():
    image = Image.new("RGB", (100, 100), "white")
    image.putpixel((50, 45), (120, 120, 120))
    regions = [Region(10, 10, 90, 30), Region(10, 60, 90, 80)]
    safe = preserve_inked_gaps(image, regions)
    assert safe == [Region(10, 10, 90, 80)]
    packed, _ = pack_regions(image, safe)
    assert packed.getpixel((40, 35)) == (120, 120, 120)


def test_truly_blank_gaps_can_still_be_removed():
    image = Image.new("RGB", (100, 100), "white")
    regions = [Region(10, 10, 90, 30), Region(10, 60, 90, 80)]
    assert preserve_inked_gaps(image, regions) == regions
