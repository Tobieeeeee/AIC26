from pathlib import Path


def test_extract_keyframes_is_noop_for_empty_frame_list(tmp_path: Path) -> None:
    from mvr.ingestion.extract import extract_keyframes

    output_root = tmp_path / "output"
    extract_keyframes(tmp_path / "missing.mp4", [], output_root)

    assert not output_root.exists()
