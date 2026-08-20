from pathlib import Path

import pytest


def test_ingestion_mapping_preserves_authoritative_metadata(tmp_path: Path) -> None:
    from mvr.enrichment.asr.mapping import load_keyframe_audio_sources

    video = tmp_path / "source.mp4"
    video.touch()
    (tmp_path / "video_metadata.csv").write_text(
        "video_id,video_path,duration_seconds,fps,frame_count,width,height\n"
        f"V001,{video},20.0,30.0,600,1920,1080\n",
        encoding="utf-8",
    )
    (tmp_path / "keyframe_mapping.csv").write_text(
        "keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path\n"
        "authoritative-id,V001,42,1,3690,5.0,keyframes/V001/a.jpg\n",
        encoding="utf-8",
    )

    [source] = load_keyframe_audio_sources(tmp_path)

    assert source.keyframe_id == "authoritative-id"
    assert source.frame_idx == 3690
    assert source.timestamp == 5.0
    assert source.shot_id == 42


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_organizer_mapping_does_not_invent_shot_id(
    monkeypatch, tmp_path: Path, encoding: str
) -> None:
    from mvr.enrichment.asr import mapping
    from mvr.ingestion.models import VideoInfo

    (tmp_path / "map_keyframes").mkdir()
    (tmp_path / "videos").mkdir()
    video = tmp_path / "videos" / "L21_V010.mp4"
    video.touch()
    (tmp_path / "map_keyframes" / "L21_V010.csv").write_text(
        "n,pts_time,fps,frame_idx\n000123,123.0,30.0,3690\n",
        encoding=encoding,
    )
    monkeypatch.setattr(
        mapping.AicDatasetRegistry,
        "video_info",
        lambda self, video_id: VideoInfo(video_id, video, 30.0, 6000, 200.0),
    )

    [source] = mapping.load_keyframe_audio_sources(tmp_path)

    assert source.keyframe_id == "000123"
    assert source.frame_idx == 3690
    assert source.timestamp == 123.0
    assert source.shot_id is None
