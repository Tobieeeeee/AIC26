"""Measuring what the cheaper OCR sampling would cost in text.

Reading only the anchor frame makes OCR roughly three times cheaper, which on
the full batch is the difference between a day and a third of one. What it
costs is recall: the neighbour frames exist to catch text that motion blur
smears in the anchor. That trade is a competition-scoring decision, so it gets
measured on real output rather than argued from the code.
"""
from __future__ import annotations

import json
from pathlib import Path

from mvr.enrichment.ocr.offset_comparison import compare_offset_runs


def _write(root: Path, video_id: str, keyframe_id: str, texts: list[str]) -> None:
    path = root / video_id / f"{keyframe_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "keyframe_id": keyframe_id,
                "texts": [{"text": text, "conf": 0.9} for text in texts],
            }
        ),
        encoding="utf-8",
    )


def test_identical_output_loses_nothing(tmp_path: Path) -> None:
    for name in ("baseline", "candidate"):
        _write(tmp_path / name, "L21_V001", "k1", ["HÀ NỘI", "Xin chào"])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["recall"] == 1.0
    assert report["missing_texts"] == []
    assert report["baseline_text_count"] == 2


def test_text_only_the_neighbours_found_is_named(tmp_path: Path) -> None:
    """Naming the losses lets a human judge whether they mattered."""
    _write(tmp_path / "baseline", "L21_V001", "k1", ["HÀ NỘI", "Xin chào", "VTV1"])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["HÀ NỘI"])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["recall"] == round(1 / 3, 4)
    assert report["missing_texts"] == ["VTV1", "Xin chào"]


def test_comparison_ignores_case_and_surrounding_space(tmp_path: Path) -> None:
    """Two readings of one sign are the same finding, not a loss."""
    _write(tmp_path / "baseline", "L21_V001", "k1", ["  HÀ NỘI "])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["hà nội"])

    assert compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")["recall"] == 1.0


def test_keyframe_coverage_is_reported_separately_from_text_recall(tmp_path: Path) -> None:
    """A keyframe that keeps some text is still searchable; one that loses all is not."""
    _write(tmp_path / "baseline", "L21_V001", "k1", ["HÀ NỘI", "VTV1"])
    _write(tmp_path / "baseline", "L21_V001", "k2", ["Thời sự"])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["HÀ NỘI"])
    _write(tmp_path / "candidate", "L21_V001", "k2", [])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["baseline_keyframes_with_text"] == 2
    assert report["candidate_keyframes_with_text"] == 1
    assert report["keyframes_that_lost_all_text"] == ["L21_V001/k2"]


def test_text_the_cheaper_run_found_alone_is_reported_not_hidden(tmp_path: Path) -> None:
    """Fewer frames can also mean fewer false merges; the report must show it."""
    _write(tmp_path / "baseline", "L21_V001", "k1", ["HÀ NỘI"])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["HÀ NỘI", "Bản tin"])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["recall"] == 1.0
    assert report["extra_texts"] == ["Bản tin"]


def test_a_baseline_without_text_does_not_divide_by_zero(tmp_path: Path) -> None:
    _write(tmp_path / "baseline", "L21_V001", "k1", [])
    _write(tmp_path / "candidate", "L21_V001", "k1", [])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["recall"] is None
    assert report["baseline_text_count"] == 0


def test_a_near_duplicate_reading_is_not_a_loss(tmp_path: Path) -> None:
    """The same sign read twice is one finding, however the diacritics land.

    Neighbour frames mostly produce variant readings of text the anchor also
    carries -- "GIẤM GẠO LEN MEN" beside "GIẤM GẠO LÊN MEN". Counting those as
    lost would report a recall drop that costs lexical search nothing, and the
    pipeline itself already treats them as one: its manifest declares
    normalized_levenshtein at 0.80 for merging candidates.
    """
    _write(tmp_path / "baseline", "L21_V001", "k1", ["GIẤM GẠO LÊN MEN", "PHỞ BÒ"])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["GIẤM GẠO LEN MEN"])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["recall"] == 0.0
    assert report["effective_recall"] == 0.5
    assert report["unique_missing_texts"] == ["PHỞ BÒ"]


def test_text_with_no_close_reading_stays_missing(tmp_path: Path) -> None:
    _write(tmp_path / "baseline", "L21_V001", "k1", ["ĐÀI TRUYỀN HÌNH", "VTV1"])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["ĐÀI TRUYỀN HÌNH"])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["effective_recall"] == 0.5
    assert report["unique_missing_texts"] == ["VTV1"]


def test_effective_recall_is_never_below_exact_recall(tmp_path: Path) -> None:
    _write(tmp_path / "baseline", "L21_V001", "k1", ["MOT", "HAI", "BA"])
    _write(tmp_path / "candidate", "L21_V001", "k1", ["MOT", "HAl"])

    report = compare_offset_runs(tmp_path / "baseline", tmp_path / "candidate")

    assert report["effective_recall"] >= report["recall"]


def test_similarity_uses_the_threshold_the_pipeline_merges_at() -> None:
    from mvr.enrichment.ocr.offset_comparison import (
        TEXT_SIMILARITY_THRESHOLD,
        normalized_levenshtein,
    )

    assert TEXT_SIMILARITY_THRESHOLD == 0.80
    assert normalized_levenshtein("phở bò", "phở bò") == 1.0
    assert normalized_levenshtein("", "") == 1.0
    assert normalized_levenshtein("abc", "") == 0.0
    assert normalized_levenshtein("kitten", "sitting") == round(1 - 3 / 7, 4)
