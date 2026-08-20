from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from mvr.enrichment.embeddings.beit3 import (
    Beit3DependencyError,
    Beit3ImageEncoder,
    _install_torch_six_inf_compat,
)
from mvr.enrichment.embeddings.artifacts import (
    EmbeddingArtifactError,
    sha256_file,
    verify_file_sha256,
)
from mvr.enrichment.embeddings.offline import (
    OfflineEmbeddingError,
    audit_embedding_artifact,
    extract_embedding_artifacts,
    run_selected_embedding_providers,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC = load_provider_specs(REPO_ROOT / "configs" / "embedding_providers.json").get(
    "beit3"
)


def test_legacy_torch_six_inf_compatibility_is_minimal(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch._six", raising=False)

    _install_torch_six_inf_compat()

    assert sys.modules["torch._six"].inf == math.inf
    assert set(vars(sys.modules["torch._six"])) >= {
        "__doc__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "inf",
    }


class FakeBeit3Encoder:
    def __init__(self, output_factory=None):
        self.output_factory = output_factory
        self.calls: list[tuple[str, ...]] = []

    def encode_images(self, image_paths):
        self.calls.append(tuple(Path(path).name for path in image_paths))
        if self.output_factory is not None:
            return self.output_factory(len(image_paths))
        values = np.zeros((len(image_paths), SPEC.embedding_dimension), dtype=np.float64)
        for index in range(len(image_paths)):
            values[index, index % SPEC.embedding_dimension] = index + 2.0
        return values


def test_beit3_runtime_is_lazy_and_missing_dependency_is_actionable(tmp_path):
    calls = []

    def unavailable(spec, checkpoint, device):
        calls.append((spec.model_id, checkpoint, device))
        raise Beit3DependencyError("use isolated BEiT-3 runtime")

    encoder = Beit3ImageEncoder(
        spec=SPEC,
        checkpoint_path=tmp_path / "checkpoint.pth",
        runtime_loader=unavailable,
    )
    assert calls == []
    with pytest.raises(Beit3DependencyError, match="isolated BEiT-3 runtime"):
        encoder.encode_images([tmp_path / "image.jpg"])
    assert calls == [("beit3", tmp_path / "checkpoint.pth", "auto")]


def test_locked_asset_checksum_fails_closed_before_model_loading(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"wrong checkpoint")
    expected = "0" * 64

    with pytest.raises(EmbeddingArtifactError, match="SHA256 mismatch"):
        verify_file_sha256(checkpoint, expected, "BEiT-3 checkpoint")
    assert sha256_file(checkpoint) != expected


def _rows(root: Path):
    rows = [
        {
            "keyframe_id": "kf_b",
            "video_id": "L21_V002",
            "image_path": "L21_V002/b.jpg",
            "frame_idx": "20",
            "shot_id": "2",
            "timestamp": "0.8",
        },
        {
            "keyframe_id": "kf_a",
            "video_id": "L21_V001",
            "image_path": "L21_V001/a.jpg",
            "frame_idx": "10",
            "shot_id": "1",
            "timestamp": "0.4",
        },
        {
            "keyframe_id": "kf_c",
            "video_id": "L21_V002",
            "image_path": "L21_V002/c.jpg",
            "frame_idx": "30",
            "shot_id": "3",
            "timestamp": "1.2",
        },
    ]
    for row in rows:
        path = root / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return rows


def _read_mapping(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_fake_beit3_writes_shape_dtype_normalization_and_identity(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _rows(keyframes)

    provider_root = extract_embedding_artifacts(
        rows,
        keyframe_root=keyframes,
        embeddings_root=tmp_path / "features" / "embeddings",
        spec=SPEC,
        encoder=FakeBeit3Encoder(),
        batch_size=1,
        checkpoint_sha256="a" * 64,
    )

    first = np.load(provider_root / "L21_V002.npy", allow_pickle=False)
    second = np.load(provider_root / "L21_V001.npy", allow_pickle=False)
    assert first.shape == (2, 768)
    assert second.shape == (1, 768)
    assert first.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    mapping = _read_mapping(provider_root / "feature_mapping.csv")
    assert [row["keyframe_id"] for row in mapping] == ["kf_b", "kf_a", "kf_c"]
    assert [row["feature_index"] for row in mapping] == ["0", "0", "1"]
    manifest = json.loads((provider_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == "a" * 64
    assert manifest["source_revision"] == SPEC.source_revision
    assert manifest["model_revision"] is None
    assert manifest["execution"]["adapter"].endswith("FakeBeit3Encoder")
    assert manifest["execution"]["batch_size"] == 1
    assert manifest["execution"]["latency_ms"] > 0
    assert set(manifest["execution"]["vector_files_sha256"]) == {
        "L21_V001.npy",
        "L21_V002.npy",
    }


def test_mapping_and_checksum_are_deterministic(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _rows(keyframes)
    roots = []
    for name in ("one", "two"):
        roots.append(
            extract_embedding_artifacts(
                rows,
                keyframe_root=keyframes,
                embeddings_root=tmp_path / name,
                spec=SPEC,
                encoder=FakeBeit3Encoder(),
            )
        )

    mappings = [(root / "feature_mapping.csv").read_bytes() for root in roots]
    manifests = [
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for root in roots
    ]
    assert mappings[0] == mappings[1]
    assert manifests[0]["mapping_sha256"] == manifests[1]["mapping_sha256"]


def test_audit_matches_fasttest_counter_contract(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _rows(keyframes)
    provider_root = extract_embedding_artifacts(
        rows,
        keyframe_root=keyframes,
        embeddings_root=tmp_path / "embeddings",
        spec=SPEC,
        encoder=FakeBeit3Encoder(),
    )
    assert audit_embedding_artifact(provider_root) == {
        "expected_keyframes": 3,
        "vectors": 3,
        "unique_ids": 3,
        "missing_ids": 0,
        "extra_ids": 0,
        "invalid_vectors": 0,
    }


def test_missing_image_fails_before_encoder_or_manifest(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _rows(keyframes)
    (keyframes / rows[0]["image_path"]).unlink()
    encoder = FakeBeit3Encoder()

    with pytest.raises(OfflineEmbeddingError, match="missing image for kf_b"):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "embeddings",
            spec=SPEC,
            encoder=encoder,
        )
    assert encoder.calls == []
    assert not (tmp_path / "embeddings" / "beit3").exists()


def test_duplicate_keyframe_id_is_rejected(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _rows(keyframes)
    # Same video AND same keyframe id. Setting only the id left the two
    # rows in different videos, which is the organizer's normal case --
    # every video numbers its keyframes from 001 -- so the assertion
    # demanded that a valid pair be rejected.
    rows[1]["keyframe_id"] = rows[0]["keyframe_id"]
    rows[1]["video_id"] = rows[0]["video_id"]
    with pytest.raises(OfflineEmbeddingError, match="duplicate keyframe"):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "embeddings",
            spec=SPEC,
            encoder=FakeBeit3Encoder(),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda count: np.ones((count, 12), dtype=np.float32), "returned shape"),
        (
            lambda count: np.full((count, 768), np.nan, dtype=np.float32),
            "returned NaN",
        ),
        (
            lambda count: np.full((count, 768), np.inf, dtype=np.float32),
            "returned Inf",
        ),
    ],
)
def test_invalid_vectors_do_not_publish_manifest(tmp_path, factory, message):
    keyframes = tmp_path / "keyframes"
    rows = _rows(keyframes)
    with pytest.raises(OfflineEmbeddingError, match=message):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "embeddings",
            spec=SPEC,
            encoder=FakeBeit3Encoder(factory),
        )
    assert not (tmp_path / "embeddings" / "beit3").exists()


def test_provider_selection_runs_only_requested_beit3():
    calls = []
    run_selected_embedding_providers(
        ("beit3",),
        {"clip": lambda: calls.append("clip"), "beit3": lambda: calls.append("beit3")},
    )
    assert calls == ["beit3"]


def test_unknown_provider_fails_without_fallback():
    with pytest.raises(ValueError, match="unknown or unavailable"):
        run_selected_embedding_providers(("unknown",), {"beit3": lambda: None})
