def test_shot_uses_inclusive_end_frame() -> None:
    from mvr.ingestion.models import Shot

    shot = Shot("V", "method", 1, 100, 124, 25.0)

    assert shot.frame_count == 25
    assert shot.duration == 1.0
