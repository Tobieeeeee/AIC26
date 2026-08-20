from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mvr.enrichment.embeddings.offline import (
    OfflineEmbeddingError,
    audit_embedding_artifact,
    extract_embedding_artifacts,
    run_selected_embedding_providers,
)
from mvr.enrichment.embeddings.perception_encoder import (
    PerceptionEncoderDependencyError,
    PerceptionEncoderImageEncoder,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC = load_provider_specs(REPO_ROOT / "configs" / "embedding_providers.json").get(
    "perception_encoder"
)


class FakePerceptionEncoder:
    def __init__(self, factory=None):
        self.factory = factory

    def encode_images(self, paths):
        if self.factory:
            return self.factory(len(paths))
        array = np.zeros((len(paths), SPEC.embedding_dimension), dtype=np.float64)
        array[:, -1] = 3.0
        return array


def _fixture(root: Path):
    rows = []
    for index, video in enumerate(("L21_V003", "L21_V001", "L21_V003")):
        relative = f"{video}/{index}.jpg"
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        rows.append(
            {
                "keyframe_id": f"pe_kf_{index}",
                "video_id": video,
                "image_path": relative,
                "frame_idx": str(index + 100),
                "shot_id": str(index),
                "timestamp": str(index / 3),
            }
        )
    return rows


def test_perception_encoder_contract_and_artifact(tmp_path):
    keyframes = tmp_path / "keyframes"
    provider = extract_embedding_artifacts(
        _fixture(keyframes),
        keyframe_root=keyframes,
        embeddings_root=tmp_path / "embeddings",
        spec=SPEC,
        encoder=FakePerceptionEncoder(),
        checkpoint_sha256=SPEC.checkpoint_sha256,
    )
    vectors = np.load(provider / "L21_V003.npy", allow_pickle=False)
    assert vectors.shape == (2, SPEC.embedding_dimension)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    manifest = json.loads((provider / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == SPEC.checkpoint_sha256
    assert manifest["model_revision"] == SPEC.model_revision
    assert audit_embedding_artifact(provider)["invalid_vectors"] == 0


def test_perception_mapping_checksum_is_deterministic(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _fixture(keyframes)
    outputs = []
    for name in ("first", "second"):
        outputs.append(
            extract_embedding_artifacts(
                rows,
                keyframe_root=keyframes,
                embeddings_root=tmp_path / name,
                spec=SPEC,
                encoder=FakePerceptionEncoder(),
            )
        )
    assert (outputs[0] / "feature_mapping.csv").read_bytes() == (
        outputs[1] / "feature_mapping.csv"
    ).read_bytes()
    hashes = [
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))[
            "mapping_sha256"
        ]
        for root in outputs
    ]
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda n: np.ones((n, 4), dtype=np.float32), "returned shape"),
        (
            lambda n: np.full((n, SPEC.embedding_dimension), np.nan, dtype=np.float32),
            "returned NaN",
        ),
        (
            lambda n: np.full((n, SPEC.embedding_dimension), np.inf, dtype=np.float32),
            "returned Inf",
        ),
    ],
)
def test_perception_rejects_invalid_vectors(tmp_path, factory, message):
    keyframes = tmp_path / "keyframes"
    with pytest.raises(OfflineEmbeddingError, match=message):
        extract_embedding_artifacts(
            _fixture(keyframes),
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "embeddings",
            spec=SPEC,
            encoder=FakePerceptionEncoder(factory),
        )
    assert not (tmp_path / "embeddings" / "perception_encoder").exists()


def test_perception_rejects_missing_and_duplicate_identity(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _fixture(keyframes)
    (keyframes / rows[-1]["image_path"]).unlink()
    with pytest.raises(OfflineEmbeddingError, match="missing image"):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "missing",
            spec=SPEC,
            encoder=FakePerceptionEncoder(),
        )
    (keyframes / rows[-1]["image_path"]).write_bytes(b"image")
    rows[-1]["keyframe_id"] = rows[0]["keyframe_id"]
    with pytest.raises(OfflineEmbeddingError, match="duplicate keyframe_id"):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "duplicate",
            spec=SPEC,
            encoder=FakePerceptionEncoder(),
        )


def test_perception_runtime_is_lazy_and_actionable(tmp_path):
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise PerceptionEncoderDependencyError("locked PE source required")

    encoder = PerceptionEncoderImageEncoder(
        spec=SPEC,
        checkpoint_path=tmp_path / "PE-Core-L14-336.pt",
        runtime_loader=unavailable,
    )
    assert calls == []
    with pytest.raises(PerceptionEncoderDependencyError, match="locked PE source"):
        encoder.encode_images([tmp_path / "image.jpg"])
    assert len(calls) == 1


def test_selection_runs_only_perception_encoder():
    calls = []
    run_selected_embedding_providers(
        ("perception_encoder",),
        {
            "beit3": lambda: calls.append("beit3"),
            "metaclip2": lambda: calls.append("metaclip2"),
            "perception_encoder": lambda: calls.append("perception_encoder"),
        },
    )
    assert calls == ["perception_encoder"]


def test_unified_powershell_runner_exposes_all_three_providers():
    script = (REPO_ROOT / "scripts" / "run_all_features.ps1").read_text(
        encoding="utf-8"
    )
    assert '"beit3,metaclip2,perception_encoder"' in script
    assert "--embedding-providers" in script
