from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mvr.enrichment.embeddings.metaclip2 import (
    MetaClip2DependencyError,
    MetaClip2ImageEncoder,
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
    "metaclip2"
)


class FakeMetaClip2:
    def __init__(self, factory=None):
        self.factory = factory

    def encode_images(self, paths):
        if self.factory:
            return self.factory(len(paths))
        array = np.zeros((len(paths), SPEC.embedding_dimension), dtype=np.float64)
        array[:, 0] = 7.0
        return array


def _fixture(root: Path):
    rows = []
    for index, video in enumerate(("L21_V002", "L21_V001", "L21_V002")):
        image_path = f"{video}/{index}.jpg"
        path = root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        rows.append(
            {
                "keyframe_id": f"kf_{index}",
                "video_id": video,
                "image_path": image_path,
                "frame_idx": str(index * 10),
                "shot_id": str(index),
                "timestamp": str(index / 2),
            }
        )
    return rows


def test_metaclip2_contract_and_atomic_artifact(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _fixture(keyframes)
    provider = extract_embedding_artifacts(
        rows,
        keyframe_root=keyframes,
        embeddings_root=tmp_path / "embeddings",
        spec=SPEC,
        encoder=FakeMetaClip2(),
        batch_size=2,
    )
    vectors = np.load(provider / "L21_V002.npy", allow_pickle=False)
    assert vectors.shape == (2, SPEC.embedding_dimension)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    manifest = json.loads((provider / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revision"] == SPEC.model_revision
    assert manifest["embedding_dimension"] == SPEC.embedding_dimension
    assert audit_embedding_artifact(provider)["invalid_vectors"] == 0


def test_metaclip2_mapping_and_checksum_are_deterministic(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _fixture(keyframes)
    manifests = []
    mappings = []
    for name in ("a", "b"):
        provider = extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / name,
            spec=SPEC,
            encoder=FakeMetaClip2(),
        )
        mappings.append((provider / "feature_mapping.csv").read_bytes())
        manifests.append(
            json.loads((provider / "manifest.json").read_text(encoding="utf-8"))
        )
    assert mappings[0] == mappings[1]
    assert manifests[0]["mapping_sha256"] == manifests[1]["mapping_sha256"]
    with (tmp_path / "a" / "metaclip2" / "feature_mapping.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert [row["keyframe_id"] for row in csv.DictReader(stream)] == [
            "kf_0",
            "kf_1",
            "kf_2",
        ]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda n: np.ones((n, 9), dtype=np.float32), "returned shape"),
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
def test_metaclip2_rejects_invalid_output(tmp_path, factory, message):
    keyframes = tmp_path / "keyframes"
    rows = _fixture(keyframes)
    with pytest.raises(OfflineEmbeddingError, match=message):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "embeddings",
            spec=SPEC,
            encoder=FakeMetaClip2(factory),
        )
    assert not (tmp_path / "embeddings" / "metaclip2").exists()


def test_metaclip2_rejects_missing_and_duplicate_identity(tmp_path):
    keyframes = tmp_path / "keyframes"
    rows = _fixture(keyframes)
    (keyframes / rows[0]["image_path"]).unlink()
    with pytest.raises(OfflineEmbeddingError, match="missing image"):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "missing",
            spec=SPEC,
            encoder=FakeMetaClip2(),
        )
    (keyframes / rows[0]["image_path"]).write_bytes(b"image")
    rows[1]["keyframe_id"] = rows[0]["keyframe_id"]
    with pytest.raises(OfflineEmbeddingError, match="duplicate keyframe_id"):
        extract_embedding_artifacts(
            rows,
            keyframe_root=keyframes,
            embeddings_root=tmp_path / "duplicate",
            spec=SPEC,
            encoder=FakeMetaClip2(),
        )


def test_metaclip2_runtime_is_lazy_and_fail_closed(tmp_path):
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise MetaClip2DependencyError("pinned local cache required")

    encoder = MetaClip2ImageEncoder(spec=SPEC, runtime_loader=unavailable)
    assert calls == []
    with pytest.raises(MetaClip2DependencyError, match="pinned local cache"):
        encoder.encode_images([tmp_path / "image.jpg"])
    assert len(calls) == 1


def test_selection_runs_only_metaclip2():
    calls = []
    run_selected_embedding_providers(
        ("metaclip2",),
        {
            "beit3": lambda: calls.append("beit3"),
            "metaclip2": lambda: calls.append("metaclip2"),
        },
    )
    assert calls == ["metaclip2"]
