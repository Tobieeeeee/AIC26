"""Folding a filled-in review sheet back into the ground truth.

Nothing here writes on its own: an unmarked sheet must change nothing, a
sheet marked twice for one query must stop rather than guess which row is
right, and a query the operator already confirmed must not be silently
overwritten by a later pass over `new_candidates.csv`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _module():
    path = REPOSITORY_ROOT / "scripts" / "apply_sublan1_review.py"
    spec = importlib.util.spec_from_file_location("_apply_sublan1_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(**overrides) -> dict:
    row = {
        "query": "query-p1-1-kis", "source": "sublan1", "video_id": "L2",
        "frame_idx": "200", "nearest_keyframe_frame": "200", "distance_frames": "0",
        "image_path": "x.jpg", "query_vi": "câu hỏi", "chon_dong_nay": "",
    }
    row.update(overrides)
    return row


def test_an_unmarked_row_changes_nothing() -> None:
    module = _module()
    ground_truth = {"query-p1-1-kis": {"video_id": "L1", "frame_idx": 100}}

    changes = module.apply_mismatch(ground_truth, [_row()])

    assert changes == []
    assert ground_truth["query-p1-1-kis"]["video_id"] == "L1"


def test_marking_the_sublan1_row_replaces_the_entry() -> None:
    module = _module()
    ground_truth = {
        "query-p1-1-kis": {"video_id": "L1", "frame_idx": 100, "verified_by": "human"}
    }

    changes = module.apply_mismatch(ground_truth, [_row(chon_dong_nay="x")])

    entry = ground_truth["query-p1-1-kis"]
    assert (entry["video_id"], entry["frame_idx"]) == ("L2", 200)
    assert entry["verified_by"] == module.VERIFIED_BY_SUBLAN1
    assert changes == ["query-p1-1-kis: ('L1', 100) -> (L2, 200)"]


def test_marking_the_operator_row_keeps_the_old_entry_untouched() -> None:
    """The person looked at both frames and confirmed the operator was
    right after all -- this must not tag a correct entry as re-verified
    or touch it in any way."""
    module = _module()
    ground_truth = {
        "query-p1-1-kis": {"video_id": "L1", "frame_idx": 100, "verified_by": "human"}
    }
    row = _row(source="ta (đã xác minh)", video_id="L1", frame_idx="100", chon_dong_nay="x")

    changes = module.apply_mismatch(ground_truth, [row])

    assert changes == []
    assert ground_truth["query-p1-1-kis"] == {
        "video_id": "L1", "frame_idx": 100, "verified_by": "human"
    }


def test_two_rows_marked_for_one_query_stops_rather_than_guessing() -> None:
    module = _module()
    rows = [_row(chon_dong_nay="x"), _row(video_id="L3", chon_dong_nay="x")]

    with pytest.raises(SystemExit, match="hơn một dòng"):
        module.apply_mismatch({"query-p1-1-kis": {}}, rows)


def test_a_new_candidate_is_added_when_marked(tmp_path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "QUERIES_DIR", tmp_path)
    ground_truth: dict = {}
    row = _row(
        query="query-p1-8-kis", video_id="L26_V171", frame_idx="5496",
        query_vi="mô tả món ăn", chon_dong_nay="x",
    )

    changes = module.apply_new(ground_truth, [row])

    entry = ground_truth["query-p1-8-kis"]
    assert entry["kind"] == "kis"
    assert (entry["video_id"], entry["frame_idx"]) == ("L26_V171", 5496)
    assert entry["verified_by"] == module.VERIFIED_BY_SUBLAN1
    assert changes == ["query-p1-8-kis: thêm mới (L26_V171, 5496)"]


def test_a_new_candidate_never_overwrites_an_existing_entry(tmp_path, monkeypatch) -> None:
    """new_candidates.csv is only for queries round 1 never verified.
    If one somehow already exists, keep the earlier verification."""
    module = _module()
    monkeypatch.setattr(module, "QUERIES_DIR", tmp_path)
    ground_truth = {"query-p1-8-kis": {"video_id": "L1", "frame_idx": 1}}
    row = _row(query="query-p1-8-kis", chon_dong_nay="x")

    changes = module.apply_new(ground_truth, [row])

    assert changes == []
    assert ground_truth["query-p1-8-kis"] == {"video_id": "L1", "frame_idx": 1}


def test_an_english_companion_is_attached_when_it_exists(tmp_path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "QUERIES_DIR", tmp_path)
    (tmp_path / "query-p1-8-kis.en.txt").write_text("a chef plating food", encoding="utf-8")
    row = _row(query="query-p1-8-kis", chon_dong_nay="x")

    module.apply_new({}, [row])  # exercised via the module's own read, not the return

    entry = {}
    module.apply_new(entry, [row])
    assert entry["query-p1-8-kis"]["query_en"] == "a chef plating food"


def test_a_missing_english_companion_omits_the_field(tmp_path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "QUERIES_DIR", tmp_path)
    row = _row(query="query-p1-8-kis", chon_dong_nay="x")

    entry: dict = {}
    module.apply_new(entry, [row])

    assert "query_en" not in entry["query-p1-8-kis"]
