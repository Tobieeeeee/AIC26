import csv

import numpy as np
import pytest

from scripts.build_qna_index_from_organizer_features import (
    VideoAssets,
    discover_videos,
    load_video,
)


def _make_dataset(tmp_path, video_id="L24_V011", n=3, *, n_vectors=None, n_keyframes=None, n_rows=None):
    features_root = tmp_path / "clip-features-32-aic25-b1" / "clip-features-32"
    mapping_root = tmp_path / "map-keyframes-aic25-b1" / "map-keyframes"
    keyframe_dir = tmp_path / "Keyframes_L24" / "keyframes" / video_id
    for d in (features_root, mapping_root, keyframe_dir):
        d.mkdir(parents=True, exist_ok=True)

    np.save(features_root / f"{video_id}.npy", np.ones((n_vectors or n, 512), dtype=np.float16))

    for i in range(1, (n_keyframes or n) + 1):
        (keyframe_dir / f"{i:03d}.jpg").write_bytes(b"jpeg")

    with (mapping_root / f"{video_id}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        for i in range(1, (n_rows or n) + 1):
            writer.writerow({"n": i, "pts_time": i * 2.5, "fps": 25.0, "frame_idx": i * 60})

    return tmp_path, VideoAssets(
        video_id=video_id,
        features_path=features_root / f"{video_id}.npy",
        keyframe_dir=keyframe_dir,
        mapping_csv=mapping_root / f"{video_id}.csv",
    )


def test_discover_finds_videos_with_complete_assets(tmp_path) -> None:
    root, _ = _make_dataset(tmp_path)

    found = discover_videos(root)

    assert [a.video_id for a in found] == ["L24_V011"]


def test_discover_skips_video_missing_a_mapping_csv(tmp_path) -> None:
    root, assets = _make_dataset(tmp_path)
    assets.mapping_csv.unlink()

    assert discover_videos(root) == []


def test_discover_skips_video_missing_keyframes(tmp_path) -> None:
    root, assets = _make_dataset(tmp_path)
    for jpg in assets.keyframe_dir.glob("*.jpg"):
        jpg.unlink()
    assets.keyframe_dir.rmdir()

    assert discover_videos(root) == []


def test_load_video_aligns_vectors_keyframes_and_mapping(tmp_path) -> None:
    _, assets = _make_dataset(tmp_path, n=3)

    features, metadata = load_video(assets)

    assert features.shape == (3, 512)
    assert features.dtype == np.float32
    assert [m["frame_id"] for m in metadata] == ["001", "002", "003"]
    assert [m["timestamp"] for m in metadata] == [2.5, 5.0, 7.5]
    assert all(m["video_id"] == "L24_V011" for m in metadata)


def test_load_video_returns_exactly_unit_norm_vectors(tmp_path) -> None:
    """The organizer ships float16, whose rounding leaves norms at ~0.9995 --
    outside FaissIndex's 1e-4 tolerance. They must come back exactly unit."""
    _, assets = _make_dataset(tmp_path, n=3)

    features, _ = load_video(assets)

    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_renormalize_preserves_direction(tmp_path) -> None:
    from scripts.build_qna_index_from_organizer_features import renormalize

    raw = np.array([[3.0, 4.0], [0.6, 0.8]], dtype=np.float32)

    out = renormalize(raw)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    # direction unchanged: both rows point the same way as their input
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.6, 0.8])


def test_renormalize_does_not_emit_nan_for_a_zero_row() -> None:
    from scripts.build_qna_index_from_organizer_features import renormalize

    out = renormalize(np.zeros((1, 4), dtype=np.float32))

    assert np.isfinite(out).all()


def test_load_video_rejects_more_vectors_than_keyframes(tmp_path) -> None:
    """A silent truncation here would attach every vector to the wrong
    frame, so the mismatch must be fatal."""
    _, assets = _make_dataset(tmp_path, n=3, n_vectors=5)

    with pytest.raises(ValueError, match="length mismatch"):
        load_video(assets)


def test_load_video_rejects_mapping_row_count_mismatch(tmp_path) -> None:
    _, assets = _make_dataset(tmp_path, n=3, n_rows=2)

    with pytest.raises(ValueError, match="length mismatch"):
        load_video(assets)


def test_load_video_orders_keyframes_by_sorted_filename(tmp_path) -> None:
    """The organizer's .npy row order is sorted filename order -- with
    zero-padded names that is also numeric order."""
    _, assets = _make_dataset(tmp_path, n=12)

    _, metadata = load_video(assets)

    ids = [m["frame_id"] for m in metadata]
    assert ids == sorted(ids)
    assert ids[:3] == ["001", "002", "003"]
    assert ids[-1] == "012"
