import pytest


@pytest.mark.parametrize(
    ("timestamp", "duration", "expected"),
    [
        (5.0, 20.0, (2.0, 8.0)),
        (1.0, 20.0, (0.0, 4.0)),
        (19.0, 20.0, (16.0, 20.0)),
    ],
)
def test_calculate_audio_window_clamps_to_video(
    timestamp: float, duration: float, expected: tuple[float, float]
) -> None:
    from mvr.enrichment.asr.audio import calculate_audio_window

    assert calculate_audio_window(timestamp, duration) == expected
