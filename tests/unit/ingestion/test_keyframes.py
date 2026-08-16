import pytest


def test_short_shot_uses_midpoint() -> None:
    from mvr.ingestion.keyframes import select_frame_indices
    from mvr.ingestion.models import Shot

    shot = Shot("V001", "pyscenedetect", 1, 100, 199, 25.0)

    assert select_frame_indices(shot, 12, 8) == [149]


def test_long_shot_sampling_is_sorted_unique_and_bounded() -> None:
    from mvr.ingestion.keyframes import select_frame_indices
    from mvr.ingestion.models import Shot

    shot = Shot("V001", "pyscenedetect", 1, 0, 999, 25.0)

    indices = select_frame_indices(shot, 12, 8)

    assert indices == [200, 400, 499, 600, 800]


def test_single_frame_shot_uses_its_only_frame() -> None:
    from mvr.ingestion.keyframes import select_frame_indices
    from mvr.ingestion.models import Shot

    shot = Shot("V001", "pyscenedetect", 1, 42, 42, 25.0)

    assert select_frame_indices(shot, 12, 8) == [42]


@pytest.mark.parametrize(
    ("shot", "long_seconds", "sample_seconds", "message"),
    [
        ((0, 1, 0.0), 12.0, 8.0, "shot.fps"),
        ((2, 1, 25.0), 12.0, 8.0, "end_frame"),
        ((0, 1, 25.0), -1.0, 8.0, "long_shot_seconds"),
        ((0, 1, 25.0), 12.0, 0.0, "sample_every_seconds"),
    ],
)
def test_keyframe_selection_rejects_invalid_parameters(
    shot: tuple[int, int, float],
    long_seconds: float,
    sample_seconds: float,
    message: str,
) -> None:
    from mvr.ingestion.keyframes import select_frame_indices
    from mvr.ingestion.models import Shot

    start, end, fps = shot
    value = Shot("V001", "pyscenedetect", 1, start, end, fps)

    with pytest.raises(ValueError, match=message):
        select_frame_indices(value, long_seconds, sample_seconds)


def test_make_keyframes_preserves_ids_timestamps_and_paths() -> None:
    from mvr.ingestion.keyframes import make_keyframes
    from mvr.ingestion.models import Shot

    shots = [Shot("V001", "pyscenedetect", 7, 100, 199, 25.0)]

    frames = make_keyframes(shots, long_shot_seconds=12, sample_every_seconds=8)

    assert [frame.to_dict() for frame in frames] == [
        {
            "keyframe_id": "V001_S0007_K01",
            "video_id": "V001",
            "shot_id": 7,
            "ordinal": 1,
            "frame_idx": 149,
            "timestamp": 5.96,
            "image_path": "keyframes/V001/V001_S0007_K01.jpg",
        }
    ]
