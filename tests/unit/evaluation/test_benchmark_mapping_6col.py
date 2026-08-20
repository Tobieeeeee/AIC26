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


def test_shot_id_is_blank_when_no_boundaries_are_supplied(tmp_path: Path) -> None:
    """The organizer ships no shot identity. Blank is explicit and auditable;
    inventing one is not. `lexical_index` already accepts a null shot_id."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path), tmp_path / "m.csv", ["L30_V072"]
        )
    )

    assert {r["shot_id"] for r in rows} == {""}


def test_shot_boundaries_assign_shot_id_by_frame_idx(tmp_path: Path) -> None:
    """Keeps the team's TransNetV2 work: it supplies boundaries while the
    organizer supplies identity. A keyframe belongs to the shot whose frame
    range contains its frame_idx."""
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path),
            tmp_path / "m.csv",
            ["L30_V072"],
            shot_boundaries={"L30_V072": [(0, 1720), (1721, 1800)]},
        )
    )

    assert [r["shot_id"] for r in rows] == ["0", "1", "1"]


def test_a_keyframe_outside_every_shot_gets_no_shot_id(tmp_path: Path) -> None:
    rows = _rows(
        build_aic_keyframe_mapping_csv(
            _organizer_root(tmp_path),
            tmp_path / "m.csv",
            ["L30_V072"],
            shot_boundaries={"L30_V072": [(0, 1720)]},
        )
    )

    assert [r["shot_id"] for r in rows] == ["0", "", ""]


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
