import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_map_keyframes_csv(root: Path, video_id: str, rows: list[tuple[str, str, str, str]]) -> None:
    csv_dir = root / "map_keyframes"
    csv_dir.mkdir(parents=True, exist_ok=True)
    lines = ["n,pts_time,fps,frame_idx"]
    lines.extend(",".join(row) for row in rows)
    (csv_dir / f"{video_id}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _touch_video(root: Path, video_id: str) -> Path:
    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / f"{video_id}.mp4"
    video_path.write_bytes(b"")
    return video_path


def test_resolved_frame_idx_differs_from_keyframe_ordinal(tmp_path: Path) -> None:
    """The CSV's frame_idx (KIS frame_id) must differ from the keyframe ordinal `n`."""
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    _write_map_keyframes_csv(
        tmp_path,
        "L21_V001",
        [("3", "8.7", "30.0", "261")],
    )

    registry = AicDatasetRegistry(tmp_path)
    resolved = registry.resolve_frame("L21_V001", "003.jpg")

    assert resolved.frame_idx == 261
    assert resolved.frame_idx != 3
    assert resolved.timestamp == 8.7
    assert resolved.fps == 30.0


def test_same_keyframe_filename_in_two_videos_does_not_collide(tmp_path: Path) -> None:
    """Composite (video_id, keyframe_ref) keying must isolate per-video mappings."""
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    _write_map_keyframes_csv(tmp_path, "L21_V001", [("3", "8.7", "30.0", "261")])
    _write_map_keyframes_csv(tmp_path, "L21_V002", [("3", "1.5", "30.0", "45")])

    registry = AicDatasetRegistry(tmp_path)

    first = registry.resolve_frame("L21_V001", "003.jpg")
    second = registry.resolve_frame("L21_V002", "003.jpg")

    assert first.video_id == "L21_V001"
    assert second.video_id == "L21_V002"
    assert first.frame_idx == 261
    assert second.frame_idx == 45
    assert first.frame_idx != second.frame_idx


def test_optional_media_info_may_be_absent(tmp_path: Path) -> None:
    """Missing media_info must not raise and must not affect other lookups."""
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    _write_map_keyframes_csv(tmp_path, "L21_V001", [("3", "8.7", "30.0", "261")])
    _touch_video(tmp_path, "L21_V001")

    registry = AicDatasetRegistry(tmp_path)

    assert registry.media_info("L21_V001") is None
    # Frame resolution is unaffected by the absent descriptive metadata.
    assert registry.resolve_frame("L21_V001", "003.jpg").frame_idx == 261


def test_media_info_is_read_when_present_and_stays_descriptive_only(tmp_path: Path) -> None:
    media_info_dir = tmp_path / "media_info"
    media_info_dir.mkdir(parents=True, exist_ok=True)
    (media_info_dir / "L21_V001.json").write_text(
        json.dumps({"title": "sample", "watch_url": "https://example.invalid"}),
        encoding="utf-8",
    )

    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    registry = AicDatasetRegistry(tmp_path)
    info = registry.media_info("L21_V001")

    assert info == {"title": "sample", "watch_url": "https://example.invalid"}
    # media_info never carries fps/frame_idx style fields for frame mapping.
    assert "fps" not in info
    assert "frame_idx" not in info


def test_missing_required_frame_mapping_is_never_guessed(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry, MissingFrameMappingError

    _write_map_keyframes_csv(tmp_path, "L21_V001", [("3", "8.7", "30.0", "261")])
    registry = AicDatasetRegistry(tmp_path)

    with pytest.raises(MissingFrameMappingError):
        registry.resolve_frame("L21_V001", "999.jpg")

    with pytest.raises(MissingFrameMappingError):
        registry.resolve_frame("L21_V999", "003.jpg")


def test_raw_video_registry_path_works_without_organizer_support_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """video_info() must succeed from the video file alone: no keyframes,
    no map_keyframes CSV, and no media_info anywhere under the dataset root.
    """
    from mvr.ingestion import probe
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    video_path = _touch_video(tmp_path, "L21_V001")
    assert not (tmp_path / "keyframes").exists()
    assert not (tmp_path / "map_keyframes").exists()
    assert not (tmp_path / "media_info").exists()

    payload = {
        "streams": [
            {
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "nb_frames": "900",
                "nb_read_frames": "900",
                "duration": "30.0",
            }
        ],
        "format": {"duration": "30.0"},
    }
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    registry = AicDatasetRegistry(tmp_path)
    info = registry.video_info("L21_V001")

    assert info.video_id == "L21_V001"
    assert info.path == video_path.resolve()
    assert info.fps == 30.0
    assert info.frame_count == 900
    assert info.duration == 30.0

    # And resolving an organizer keyframe (which doesn't exist here) still
    # fails loudly instead of silently guessing.
    from mvr.ingestion.aic_dataset import MissingFrameMappingError

    with pytest.raises(MissingFrameMappingError):
        registry.resolve_frame("L21_V001", "003.jpg")


def test_keyframe_path_and_video_path_resolve_only_when_present(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    registry = AicDatasetRegistry(tmp_path)
    assert registry.video_path("L21_V001") is None
    assert registry.keyframe_path("L21_V001", "003") is None

    _touch_video(tmp_path, "L21_V001")
    keyframes_dir = tmp_path / "keyframes" / "L21_V001"
    keyframes_dir.mkdir(parents=True)
    (keyframes_dir / "003.jpg").write_bytes(b"")

    assert registry.video_path("L21_V001") is not None
    assert registry.keyframe_path("L21_V001", "3") == keyframes_dir / "003.jpg"


def test_registry_with_no_dataset_root_never_guesses(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    registry = AicDatasetRegistry(None)

    assert registry.video_path("L21_V001") is None
    assert registry.media_info("L21_V001") is None
    with pytest.raises(FileNotFoundError):
        registry.video_info("L21_V001")


# -- the raw organizer download layout ----------------------------------------
#
# The registry only ever read the normalized tree (`<root>/map_keyframes/`,
# `<root>/keyframes/`). What the organizer actually ships is one folder per
# asset kind, each with a batch suffix and an inner directory whose name uses
# hyphens where the normalized tree uses underscores:
#
#     <root>/map-keyframes-aic25-b1/map-keyframes/L21_V001.csv
#     <root>/Keyframes_L26_a/keyframes/L26_V001/003.jpg
#     <root>/media-info-aic25-b1/media-info/L21_V001.json
#     <root>/Videos_L21_a/video/L21_V001.mp4
#
# Measured consequence on the real 873-video corpus: every TRAKE run returned
# "(no candidate chains found)", because resolve_frame() found no mapping for
# any video. `build_keyframe_mapping.py` has read both layouts all along,
# which is why KIS worked and TRAKE did not.


def _write_raw_map_keyframes_csv(
    root: Path, video_id: str, rows: list[tuple[str, str, str, str]]
) -> None:
    csv_dir = root / "map-keyframes-aic25-b1" / "map-keyframes"
    csv_dir.mkdir(parents=True, exist_ok=True)
    lines = ["n,pts_time,fps,frame_idx"]
    lines.extend(",".join(row) for row in rows)
    (csv_dir / f"{video_id}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resolves_a_frame_from_the_raw_download_layout(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    _write_raw_map_keyframes_csv(tmp_path, "L21_V001", [("3", "8.7", "30.0", "261")])

    resolved = AicDatasetRegistry(tmp_path).resolve_frame("L21_V001", "003.jpg")

    assert resolved.frame_idx == 261


def test_finds_a_keyframe_image_inside_its_collection_folder(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    directory = tmp_path / "Keyframes_L26_a" / "keyframes" / "L26_V001"
    directory.mkdir(parents=True)
    (directory / "003.jpg").write_bytes(b"")

    found = AicDatasetRegistry(tmp_path).keyframe_path("L26_V001", "3")

    assert found == directory / "003.jpg"


def test_reads_media_info_from_the_raw_download_layout(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    directory = tmp_path / "media-info-aic25-b1" / "media-info"
    directory.mkdir(parents=True)
    (directory / "L21_V001.json").write_text(
        json.dumps({"title": "một video"}), encoding="utf-8"
    )

    assert AicDatasetRegistry(tmp_path).media_info("L21_V001") == {"title": "một video"}


def test_finds_a_video_in_the_raw_download_layout(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    directory = tmp_path / "Videos_L21_a" / "video"
    directory.mkdir(parents=True)
    (directory / "L21_V001.mp4").write_bytes(b"")

    assert AicDatasetRegistry(tmp_path).video_path("L21_V001") == directory / "L21_V001.mp4"


def test_the_normalized_layout_is_preferred_when_both_are_present(tmp_path: Path) -> None:
    """A normalized tree is something a person built deliberately, on purpose,
    from the raw one. Letting the raw copy win would silently discard that."""
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    _write_map_keyframes_csv(tmp_path, "L21_V001", [("3", "8.7", "30.0", "111")])
    _write_raw_map_keyframes_csv(tmp_path, "L21_V001", [("3", "8.7", "30.0", "999")])

    assert AicDatasetRegistry(tmp_path).resolve_frame("L21_V001", "3").frame_idx == 111


def test_a_keyframe_absent_from_every_collection_folder_is_still_none(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    (tmp_path / "Keyframes_L21" / "keyframes" / "L21_V001").mkdir(parents=True)
    (tmp_path / "Keyframes_L26_a" / "keyframes" / "L26_V001").mkdir(parents=True)

    assert AicDatasetRegistry(tmp_path).keyframe_path("L21_V001", "3") is None


def test_one_collection_folder_does_not_answer_for_another(tmp_path: Path) -> None:
    """Keyframes_L26_a and Keyframes_L26_b split ONE collection across two
    folders, so every folder has to be searched -- but a hit must still be
    the requested video, not whichever folder was globbed first."""
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    wanted = tmp_path / "Keyframes_L26_b" / "keyframes" / "L26_V300"
    wanted.mkdir(parents=True)
    (wanted / "007.jpg").write_bytes(b"")
    decoy = tmp_path / "Keyframes_L26_a" / "keyframes" / "L26_V001"
    decoy.mkdir(parents=True)
    (decoy / "007.jpg").write_bytes(b"")

    found = AicDatasetRegistry(tmp_path).keyframe_path("L26_V300", "7")

    assert found == wanted / "007.jpg"


def test_a_root_with_neither_layout_resolves_nothing_rather_than_raising(tmp_path: Path) -> None:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry, MissingFrameMappingError

    registry = AicDatasetRegistry(tmp_path)

    assert registry.keyframe_path("L21_V001", "3") is None
    assert registry.media_info("L21_V001") is None
    assert registry.video_path("L21_V001") is None
    with pytest.raises(MissingFrameMappingError):
        registry.resolve_frame("L21_V001", "3")
