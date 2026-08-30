import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.ingestion.hybrid_selector import HybridSelectorConfig, candidate_frame_indices
from mvr.ingestion.models import Shot, VideoInfo
from mvr.ingestion.selector_cache import (
    SelectorCacheError,
    compute_edge_change_scores,
    load_or_build_selector_cache,
    run_hybrid_selector,
    run_semantic_dedup_selector,
    validate_selector_cache_manifest,
)
from mvr.ingestion.semantic_dedup import SemanticDedupConfig


def _spec():
    return load_provider_specs(Path("configs/embedding_providers.json")).get("metaclip2")


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, paths):
        self.calls += 1
        vectors = np.zeros((len(paths), 512), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def test_cache_manifest_validation_rejects_identity_change() -> None:
    expected = {
        "video_id": "V",
        "source_fingerprint": {"sha256": "a"},
        "shot_manifest_sha256": "b",
        "candidate_step_sec": 1.0,
        "metaclip": {"model_revision": "c"},
    }
    payload = {"schema_version": "1.0", **expected}
    validate_selector_cache_manifest(payload, expected)
    changed = {**expected, "candidate_step_sec": 2.0}
    with pytest.raises(SelectorCacheError, match="candidate_step_sec mismatch"):
        validate_selector_cache_manifest(payload, changed)


def test_candidate_extraction_and_embedding_cache_are_reused(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "V.mp4"
    video_path.write_bytes(b"deterministic video identity")
    video = VideoInfo("V", video_path, 1.0, 11, 11.0)
    shots = [Shot("V", "test", 1, 0, 10, 1.0)]
    extraction_calls = []

    def fake_extract(_video, candidates, output_root):
        extraction_calls.append(len(candidates))
        for candidate in candidates:
            path = output_root / candidate.image_path
            path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((16, 16, 3), candidate.frame_idx, dtype=np.uint8)
            assert cv2.imwrite(str(path), image)

    monkeypatch.setattr("mvr.ingestion.selector_cache.extract_keyframes", fake_extract)
    encoder = FakeEncoder()
    kwargs = dict(
        video=video,
        shots=shots,
        output_root=tmp_path / "output",
        config=HybridSelectorConfig(),
        candidate_step_sec=HybridSelectorConfig().candidate_step_sec,
        candidate_indices_fn=lambda shot: candidate_frame_indices(
            shot, HybridSelectorConfig().candidate_step_sec
        ),
        spec=_spec(),
        encoder=encoder,
        embedding_batch_size=32,
    )
    first = load_or_build_selector_cache(**kwargs)
    second = load_or_build_selector_cache(**kwargs)
    assert extraction_calls == [len(first[0])]
    assert encoder.calls == 1
    assert second[3]["cache_reused"] is True
    assert np.array_equal(first[1], second[1])

    manifest_path = tmp_path / "output/selector_cache/V/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metaclip"]["model_revision"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SelectorCacheError, match="metaclip mismatch"):
        load_or_build_selector_cache(**kwargs)


def test_edge_change_signal_reacts_to_text_like_edges(tmp_path: Path) -> None:
    first = np.zeros((180, 320), dtype=np.uint8)
    second = first.copy()
    cv2.putText(second, "SLIDE 2", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    assert cv2.imwrite(str(paths[0]), first)
    assert cv2.imwrite(str(paths[1]), second)
    scores = compute_edge_change_scores(paths, [1, 1])
    assert scores[0] == 0.0
    assert scores[1] > 0.01


def test_hybrid_integration_preserves_canonical_mapping_and_is_resumable(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "V.mp4"
    video_path.write_bytes(b"video")
    video = VideoInfo("V", video_path, 1.0, 21, 21.0)
    shots = [Shot("V", "test", 7, 0, 20, 1.0)]

    def fake_extract(_video, candidates, output_root):
        for candidate in candidates:
            path = output_root / candidate.image_path
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))

    monkeypatch.setattr("mvr.ingestion.selector_cache.extract_keyframes", fake_extract)
    kwargs = dict(
        video=video,
        shots=shots,
        output_root=tmp_path / "output",
        config=HybridSelectorConfig(),
        spec=_spec(),
        encoder=FakeEncoder(),
        embedding_batch_size=32,
    )
    keyframes, manifest = run_hybrid_selector(**kwargs)
    assert [item.keyframe_id for item in keyframes] == [
        f"V_S0007_K{ordinal:02d}" for ordinal in range(1, len(keyframes) + 1)
    ]
    assert all((tmp_path / "output" / item.image_path).is_file() for item in keyframes)
    assert manifest["max_observed_temporal_gap_sec"] <= 8.0
    assert (tmp_path / "output/selector_embeddings/metaclip2/manifest.json").is_file()

    repeated, repeated_manifest = run_hybrid_selector(**kwargs)
    assert repeated == keyframes
    assert repeated_manifest["candidate_cache_reused"] is True


def test_semantic_dedup_integration_preserves_canonical_mapping_and_is_resumable(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "V.mp4"
    video_path.write_bytes(b"video")
    video = VideoInfo("V", video_path, 1.0, 21, 21.0)
    shots = [Shot("V", "test", 7, 0, 20, 1.0)]

    def fake_extract(_video, candidates, output_root):
        for candidate in candidates:
            path = output_root / candidate.image_path
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))

    monkeypatch.setattr("mvr.ingestion.selector_cache.extract_keyframes", fake_extract)
    kwargs = dict(
        video=video,
        shots=shots,
        output_root=tmp_path / "output",
        config=SemanticDedupConfig(),
        spec=_spec(),
        encoder=FakeEncoder(),
        embedding_batch_size=32,
    )
    keyframes, manifest = run_semantic_dedup_selector(**kwargs)
    assert [item.keyframe_id for item in keyframes] == [
        f"V_S0007_K{ordinal:02d}" for ordinal in range(1, len(keyframes) + 1)
    ]
    assert all((tmp_path / "output" / item.image_path).is_file() for item in keyframes)
    assert (tmp_path / "output/selector_embeddings/metaclip2/manifest.json").is_file()

    # FakeEncoder returns identical vectors for every candidate, so every
    # candidate after the shot's first is a "duplicate" (similarity 1.0 > the
    # 0.85 threshold) and only survives via the forced 8-second gap keep.
    max_gap_frames = manifest["config"]["max_gap_seconds"] * shots[0].fps
    frame_indices = [item.frame_idx for item in keyframes]
    assert frame_indices[0] == 0
    for left, right in zip(frame_indices, frame_indices[1:]):
        assert right - left <= max_gap_frames
    assert manifest["dropped_count"] > 0
    assert manifest["forced_gap_keep_count"] == len(keyframes) - 1

    repeated, repeated_manifest = run_semantic_dedup_selector(**kwargs)
    assert repeated == keyframes
    assert repeated_manifest["candidate_cache_reused"] is True
