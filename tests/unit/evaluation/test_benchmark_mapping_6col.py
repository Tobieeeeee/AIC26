"""The organizer keyframes as the team's single canonical identity.

Decided under a 10-hour deadline: the organizer already decoded 873 videos
(83 GB, ~130 hours) into 166,550 keyframes and shipped CLIP features for
every one of them. Re-deriving that costs ~13 hours of video decoding even
at 10x realtime -- a cost a GPU cannot remove, because it is I/O and
decoding, not matrix multiplication.

Everything downstream reads identity from one CSV:

    task05a.load_canonical_mapping(mapping_csv)
    docker_ocr.process_dataset_docker_ocr(mapping_csv, ...)

so adopting the organizer keyframes means emitting a different CSV, not
refactoring a pipeline. This widens the existing adapter from 4 columns to
the 6 those consumers require.

`keyframe_id` is the JPEG stem -- "034", not the CSV `n` column's "34".
Chosen because the stem is what exists on disk and what `image_path` must
agree with. The two spellings do not join, and a mismatch produces an empty
result silently rather than an error.
"""
from __future__ import annotations

import csv
from pathlib import Path

from mvr.evaluation.benchmark import build_aic_keyframe_mapping_csv

TASK05A_REQUIRED = {"keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp"}


def _organizer_root(tmp_path: Path, *, video_id: str = "L30_V072") -> Path:
    root = tmp_path / "root"
    keyframes = root / "keyframes" / video_id
    keyframes.mkdir(parents=True)
    for stem in ("033", "034", "035"):
        (keyframes / f"{stem}.jpg").write_bytes(b"jpg")

    mapping = root / "map_keyframes"
    mapping.mkdir(parents=True)
    with (mapping / f"{video_id}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": "33", "pts_time": "68.56", "fps": "25.0", "frame_idx": "1714"})
        writer.writerow({"n": "34", "pts_time": "69.8", "fps": "25.0", "frame_idx": "1745"})
        writer.writerow({"n": "35", "pts_time": "71.04", "fps": "25.0", "frame_idx": "1776"})
    return root


def _rows(output: Path) -> list[dict[str, str]]:
    with output.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_column_task05a_requires_is_present(tmp_path: Path) -> None:
    output = build_aic_keyframe_mapping_csv(
        _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
    )

    with output.open(newline="", encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])

    assert TASK05A_REQUIRED <= columns
    assert "image_path" in columns


def test_keyframe_id_is_the_jpeg_stem_not_the_csv_n_column(tmp_path: Path) -> None:
    """The decision. "034" and "34" do not join, and the failure is silent."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    )

    assert [r["keyframe_id"] for r in rows] == ["033", "034", "035"]


def test_image_path_agrees_with_keyframe_id(tmp_path: Path) -> None:
    """The reason the stem was chosen over `n`."""
    for row in _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    ):
        assert Path(row["image_path"]).stem == row["keyframe_id"]


def test_frame_idx_is_taken_from_the_organizer_not_recomputed(tmp_path: Path) -> None:
    """frame_idx is the number submitted to the organizer. They state it, so
    deriving it from fps and timestamp would be re-deriving a known fact and
    risking a rounding disagreement."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    )

    assert [r["frame_idx"] for r in rows] == ["1714", "1745", "1776"]


def test_timestamp_comes_from_pts_time(tmp_path: Path) -> None:
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    )

    assert [r["timestamp"] for r in rows] == ["68.56", "69.8", "71.04"]


def test_shot_id_and_ordinal_are_always_integers(tmp_path: Path) -> None:
    """Nine call sites across task05a, feature_extraction, docker_ocr,
    scene_text and the ASR mapping do `int(row["shot_id"])`. This column was
    blank when no boundaries were supplied, which made the mapping unloadable
    by every one of them -- see `test_task05a_can_load_what_this_writes`."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    )

    for row in rows:
        int(row["shot_id"])
        int(row["ordinal"])


def test_without_boundaries_each_organizer_keyframe_is_its_own_shot(
    tmp_path: Path,
) -> None:
    """The honest degradation. The organizer's keyframes are already
    shot-aware -- spacing runs 0.04s to 7.0s with 30-150 distinct values per
    video, which is cut-driven -- so this restates their sampling rather than
    inventing cuts. The cost is that OCR dedupe within a shot has nothing to
    group, so every keyframe is read independently; collapsing the video into
    one shot instead would make temporal OCR sample one anchor for ~190
    keyframes."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    )

    assert [r["shot_id"] for r in rows] == ["33", "34", "35"]
    assert [r["ordinal"] for r in rows] == ["1", "1", "1"]


def test_shot_boundaries_assign_shot_id_by_frame_idx(tmp_path: Path) -> None:
    """Keeps the team's TransNetV2 work: it supplies boundaries while the
    organizer supplies identity. A keyframe belongs to the shot whose frame
    range contains its frame_idx. Numbered from 1, like the team's own
    detectors, so a shot_id means the same thing on both sides."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path),
            tmp_path / "m.csv",
            ["L30_V072"],
            shot_boundaries={"L30_V072": [(0, 1720), (1721, 1800)]},
        )
    )

    assert [r["shot_id"] for r in rows] == ["1", "2", "2"]


def test_ordinal_counts_position_within_the_shot(tmp_path: Path) -> None:
    """`ingestion.keyframes` numbers keyframes 1..n inside each shot, and
    `feature_extraction.sort_keyframes_deterministically` sorts on
    (video_id, shot_id, ordinal). A mapping built from organizer data has to
    number them the same way or that sort is not deterministic."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path),
            tmp_path / "m.csv",
            ["L30_V072"],
            shot_boundaries={"L30_V072": [(0, 1720), (1721, 1800)]},
        )
    )

    assert [r["ordinal"] for r in rows] == ["1", "1", "2"]


def test_a_keyframe_outside_every_shot_gets_the_sentinel(tmp_path: Path) -> None:
    """Zero cannot collide with a real shot, because the team's detectors
    number from 1. Dropping the row would lose a frame the competition may
    be scored on; a blank would not load at all."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path),
            tmp_path / "m.csv",
            ["L30_V072"],
            shot_boundaries={"L30_V072": [(0, 1720)]},
        )
    )

    assert [r["shot_id"] for r in rows] == ["1", "0", "0"]


def test_the_split_organizer_layout_is_searched_too(tmp_path: Path) -> None:
    """Raw organizer data splits keyframes across Keyframes_L21/,
    Keyframes_L26_a/, _b/, ... rather than one flat directory."""
    root = tmp_path / "raw"
    keyframes = root / "Keyframes_L30" / "keyframes" / "L30_V072"
    keyframes.mkdir(parents=True)
    (keyframes / "034.jpg").write_bytes(b"jpg")
    mapping = root / "map_keyframes"
    mapping.mkdir(parents=True)
    with (mapping / "L30_V072.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": "34", "pts_time": "69.8", "fps": "25.0", "frame_idx": "1745"})

    rows = _rows(build_aic_keyframe_mapping_csv(root, tmp_path / "m.csv", ["L30_V072"]))

    assert [r["keyframe_id"] for r in rows] == ["034"]


def test_a_video_with_no_mapping_csv_is_skipped_not_guessed(tmp_path: Path) -> None:
    """23 of 873 videos ship no keyframes. Skipping is honest; fabricating a
    frame_idx would poison the one number the competition scores on."""
    root = _organizer_root(tmp_path)

    rows = _rows(
        build_aic_keyframe_mapping_csv(root, tmp_path / "m.csv", ["L30_V072", "L99_V999"])
    )

    assert {r["video_id"] for r in rows} == {"L30_V072"}


# -- the raw organizer layout, and refusing to fail silently ------------


def test_the_raw_organizer_map_keyframes_layout_is_found(tmp_path: Path) -> None:
    """Raw download nests it as `map-keyframes-aic25-b1/map-keyframes/`,
    not `map_keyframes/`. Running against the real corpus produced zero rows
    until this was handled."""
    root = tmp_path / "raw"
    keyframes = root / "Keyframes_L30" / "keyframes" / "L30_V072"
    keyframes.mkdir(parents=True)
    (keyframes / "034.jpg").write_bytes(b"jpg")
    mapping = root / "map-keyframes-aic25-b1" / "map-keyframes"
    mapping.mkdir(parents=True)
    with (mapping / "L30_V072.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerow({"n": "34", "pts_time": "69.8", "fps": "25.0", "frame_idx": "1745"})

    rows = _rows(build_aic_keyframe_mapping_csv(root, tmp_path / "m.csv", ["L30_V072"]))

    assert [(r["keyframe_id"], r["frame_idx"]) for r in rows] == [("034", "1745")]


def test_producing_nothing_at_all_raises_instead_of_writing_an_empty_file(
    tmp_path: Path,
) -> None:
    """A wrong dataset root is the likeliest cause, and an empty CSV would be
    accepted happily by everything downstream -- OCR, Task05A and the index
    builder would each report "0 documents" as if that were the data. The
    first version of this did exactly that against the real corpus.

    Scoped to "no mapping CSV found at all". A video that merely has no
    downloaded keyframes still has its CSV and legitimately yields no rows --
    the real L22-L30 situation -- so that case must keep succeeding.
    """
    import pytest

    empty = tmp_path / "nothing"
    empty.mkdir()

    with pytest.raises(ValueError, match="no map_keyframes CSV"):
        build_aic_keyframe_mapping_csv(empty, tmp_path / "m.csv", ["L30_V072"])


def test_a_partially_missing_corpus_still_succeeds(tmp_path: Path) -> None:
    """23 of 873 videos ship no keyframes; that is expected, not a failure."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072", "L99_V999"]
        )
    )

    assert len(rows) == 3


# -- the consumers, run for real ---------------------------------------
#
# Asserting on the columns proved nothing: the first version of this
# adapter satisfied every column assertion above and still could not be
# loaded by a single downstream module, because `shot_id` was blank and
# they all do `int(row["shot_id"])`. Reading a file's header is not the
# same as reading the file. These hand the output to the real readers.


def test_task05a_can_load_what_this_writes(tmp_path: Path) -> None:
    """The consumer the whole adapter exists to feed."""
    from mvr.indexing.task05a import load_canonical_mapping

    output = build_aic_keyframe_mapping_csv(
        _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
    )

    keyframes = load_canonical_mapping(output)

    assert [k.keyframe_id for k in keyframes] == ["033", "034", "035"]
    assert [k.frame_idx for k in keyframes] == [1714, 1745, 1776]


def test_feature_extraction_can_sort_what_this_writes(tmp_path: Path) -> None:
    """Sorts on (video_id, shot_id, ordinal) -- it needs `ordinal`, which
    the 6-column version did not emit at all."""
    from mvr.enrichment.feature_extraction import sort_keyframes_deterministically

    output = build_aic_keyframe_mapping_csv(
        _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
    )

    assert [k["keyframe_id"] for k in sort_keyframes_deterministically(output)] == [
        "033",
        "034",
        "035",
    ]


def test_the_column_set_matches_the_team_ingestion_exactly(tmp_path: Path) -> None:
    """A mapping built from organizer data and one built by the team's own
    ingestion have to be interchangeable, or every consumer needs to know
    which of the two it was handed."""
    from mvr.evaluation.benchmark import KEYFRAME_MAPPING_COLUMNS

    ingestion_columns = (
        "keyframe_id",
        "video_id",
        "shot_id",
        "ordinal",
        "frame_idx",
        "timestamp",
        "image_path",
    )

    assert KEYFRAME_MAPPING_COLUMNS == ingestion_columns
