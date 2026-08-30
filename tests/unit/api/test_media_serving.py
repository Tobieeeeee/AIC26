"""Serving keyframe images out of the layout the organizer actually ships.

`MVR_KEYFRAME_DIR` names one directory, which is all the normalized layout
needs. The organizer's download splits the same corpus across fourteen roots
-- Keyframes_L21/keyframes, ... Keyframes_L30/keyframes -- so no single value
covers it and every thumbnail outside the chosen root 404s.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mvr.api.app import create_app
from mvr.ingestion.aic_dataset import AicDatasetRegistry

JPEG = b"\xff\xd8\xff\xe0jpeg-bytes"


def _write(path: Path, payload: bytes = JPEG) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _raw_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _write(root / "Keyframes_L21" / "keyframes" / "L21_V001" / "003.jpg")
    _write(root / "Keyframes_L30" / "keyframes" / "L30_V007" / "045.jpg")
    return root


def _client(root: Path, keyframe_dir: Path | None = None) -> TestClient:
    return TestClient(
        create_app(
            semantic_service=None,
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=AicDatasetRegistry(root),
            keyframe_dir=keyframe_dir,
            video_dir=None,
            qna_answer_reasoner=None,
            trake_query_translator=None,
        )
    )


def test_keyframes_serve_from_every_root_of_the_organizer_layout(tmp_path) -> None:
    client = _client(_raw_corpus(tmp_path))

    first = client.get("/keyframes/L21_V001/003.jpg")
    last = client.get("/keyframes/L30_V007/045.jpg")

    assert first.status_code == 200, first.text
    assert first.content == JPEG
    # A different batch folder entirely -- the case a single mounted directory
    # cannot reach.
    assert last.status_code == 200, last.text
    assert last.content == JPEG


def test_unknown_keyframe_is_404_not_500(tmp_path) -> None:
    client = _client(_raw_corpus(tmp_path))

    assert client.get("/keyframes/L21_V001/999.jpg").status_code == 404
    assert client.get("/keyframes/L99_V999/003.jpg").status_code == 404


def test_keyframe_route_refuses_to_walk_out_of_the_corpus(tmp_path) -> None:
    root = _raw_corpus(tmp_path)
    _write(tmp_path / "secret.txt", b"not a keyframe")

    client = _client(root)

    for attempt in (
        "/keyframes/L21_V001/..%2F..%2Fsecret.txt",
        "/keyframes/..%2F..%2Fsecret.txt/003.jpg",
        "/keyframes/L21_V001/%2E%2E%2Fsecret.txt",
    ):
        response = client.get(attempt)
        assert response.status_code in (400, 404), f"{attempt} -> {response.status_code}"
        assert b"not a keyframe" not in response.content


def test_a_mounted_keyframe_directory_still_works(tmp_path) -> None:
    """The normalized layout must keep working exactly as before."""
    keyframe_dir = tmp_path / "flat"
    _write(keyframe_dir / "video-1" / "frame-high.jpg")

    client = _client(_raw_corpus(tmp_path), keyframe_dir=keyframe_dir)

    response = client.get("/keyframes/video-1/frame-high.jpg")

    assert response.status_code == 200, response.text
    assert response.content == JPEG


MP4 = b"\x00\x00\x00\x18ftypmp42"


def _corpus_with_videos(tmp_path: Path) -> Path:
    """Both shipped video layouts under one root: the organizer's, and the one
    the R2 reorg replaced it with."""
    root = tmp_path / "videos-corpus"
    _write(root / "Videos_L21_a" / "video" / "L21_V001.mp4", MP4)
    _write(root / "batch1" / "videos" / "L30" / "L30_V031.mp4", MP4)
    return root


def test_videos_serve_from_the_organizer_and_the_r2_reorg_layouts(tmp_path) -> None:
    client = _client(_corpus_with_videos(tmp_path))

    organizer = client.get("/videos/L21_V001.mp4")
    mirrored = client.get("/videos/L30_V031.mp4")

    assert organizer.status_code == 200, organizer.text
    assert organizer.content == MP4
    assert mirrored.status_code == 200, mirrored.text
    assert mirrored.content == MP4


def test_unknown_video_is_404_not_500(tmp_path) -> None:
    client = _client(_corpus_with_videos(tmp_path))

    assert client.get("/videos/L99_V999.mp4").status_code == 404


def test_video_route_refuses_to_walk_out_of_the_corpus(tmp_path) -> None:
    root = _corpus_with_videos(tmp_path)
    _write(tmp_path / "secret.txt", b"not a video")

    client = _client(root)

    for attempt in ("/videos/..%2F..%2Fsecret.txt", "/videos/%2E%2E%2Fsecret.txt"):
        response = client.get(attempt)
        assert response.status_code in (400, 404), f"{attempt} -> {response.status_code}"
        assert b"not a video" not in response.content


def test_a_mounted_video_directory_still_works(tmp_path) -> None:
    video_dir = tmp_path / "flat-videos"
    _write(video_dir / "video-1.mp4", MP4)

    client = TestClient(
        create_app(
            semantic_service=None,
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=AicDatasetRegistry(_corpus_with_videos(tmp_path)),
            keyframe_dir=None,
            video_dir=video_dir,
            qna_answer_reasoner=None,
            trake_query_translator=None,
        )
    )

    response = client.get("/videos/video-1.mp4")

    assert response.status_code == 200, response.text
    assert response.content == MP4
