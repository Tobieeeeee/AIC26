import argparse
import inspect
from pathlib import Path

import numpy as np
import pytest

from mvr.cli import _add_ingest_parser
from mvr.enrichment.embeddings.offline import (
    OfflineEmbeddingError,
    audit_embedding_artifact,
    reuse_embedding_artifact,
    reuse_embedding_artifacts,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.ingestion.models import Keyframe
from mvr.ingestion.pipeline import run_pipeline
from mvr.ingestion.selector_cache import _write_selected_embedding_artifact
from scripts.run_ingestion import build_parser


def _cli_defaults():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_parser(subparsers)
    return parser.parse_args(["ingest", "video.mp4", "--output", "out"])


def test_all_runnable_ingestion_defaults_match_canonical_profile() -> None:
    cli = _cli_defaults()
    script = build_parser().parse_args(["--input", "video.mp4", "--output", "out"])
    expected = {
        "primary": "transnetv2",
        "transnet_threshold": 0.5,
        "selector_mode": "hybrid_temporal_content",
        "long_shot_seconds": 12.0,
        "sample_every_seconds": 8.0,
    }
    assert cli.methods == ["transnetv2"]
    assert script.methods == "transnetv2"
    signature = inspect.signature(run_pipeline)
    assert signature.parameters["methods"].default is None
    for name, value in expected.items():
        assert getattr(cli, name) == value
        assert getattr(script, name) == value
        assert signature.parameters[name].default == value


def test_default_yaml_matches_canonical_profile() -> None:
    text = Path("configs/default.yaml").read_text(encoding="utf-8")
    ingestion = text.split("\nenrichment:", maxsplit=1)[0]
    assert "shot_detector: transnetv2" in ingestion
    assert "transnet_threshold: 0.5" in ingestion
    assert "mode: hybrid_temporal_content" in ingestion
    assert "long_threshold_sec: 12.0" in ingestion
    assert "interval_sec: 8.0" in ingestion


def test_selector_embedding_artifact_is_reused_without_encoder(tmp_path: Path) -> None:
    spec = load_provider_specs(Path("configs/embedding_providers.json")).get("metaclip2")
    keyframe = Keyframe(
        "V_S0001_K01", "V", 1, 1, 10, 1.0, "keyframes/V/V_S0001_K01.jpg"
    )
    source = _write_selected_embedding_artifact(
        output_root=tmp_path / "ingest",
        keyframes=[keyframe],
        vectors_by_identity={("V", 1, 10): np.r_[1.0, np.zeros(511, dtype=np.float32)]},
        spec=spec,
    )
    rows = [
        {
            "keyframe_id": keyframe.keyframe_id,
            "video_id": keyframe.video_id,
            "frame_idx": str(keyframe.frame_idx),
            "image_path": keyframe.image_path,
        }
    ]
    target = reuse_embedding_artifact(
        source,
        embeddings_root=tmp_path / "features/embeddings",
        rows=rows,
        spec=spec,
    )
    assert target == tmp_path / "features/embeddings/metaclip2"
    assert audit_embedding_artifact(target)["invalid_vectors"] == 0


def test_selector_embedding_reuse_rejects_canonical_identity_mismatch(tmp_path: Path) -> None:
    spec = load_provider_specs(Path("configs/embedding_providers.json")).get("metaclip2")
    keyframe = Keyframe("V_S0001_K01", "V", 1, 1, 10, 1.0, "x.jpg")
    source = _write_selected_embedding_artifact(
        output_root=tmp_path / "ingest",
        keyframes=[keyframe],
        vectors_by_identity={("V", 1, 10): np.r_[1.0, np.zeros(511, dtype=np.float32)]},
        spec=spec,
    )
    with pytest.raises(OfflineEmbeddingError, match="candidate identity"):
        reuse_embedding_artifact(
            source,
            embeddings_root=tmp_path / "features/embeddings",
            rows=[{"keyframe_id": "other", "video_id": "V"}],
            spec=spec,
        )


def test_collection_reuse_merges_per_video_selector_artifacts(tmp_path: Path) -> None:
    spec = load_provider_specs(Path("configs/embedding_providers.json")).get("metaclip2")
    keyframes = [
        Keyframe(
            "L21_V001_S0001_K01",
            "L21_V001",
            1,
            1,
            10,
            1.0,
            "data/corpus/keyframes/L21/L21_V001/L21_V001_S0001_K01.jpg",
        ),
        Keyframe(
            "L21_V002_S0001_K01",
            "L21_V002",
            1,
            1,
            20,
            2.0,
            "data/corpus/keyframes/L21/L21_V002/L21_V002_S0001_K01.jpg",
        ),
    ]
    sources = []
    rows = []
    for index, keyframe in enumerate(keyframes):
        vector = np.zeros(512, dtype=np.float32)
        vector[index] = 1.0
        sources.append(
            _write_selected_embedding_artifact(
                output_root=tmp_path / f"ingest-{index}",
                keyframes=[keyframe],
                vectors_by_identity={
                    (keyframe.video_id, keyframe.shot_id, keyframe.frame_idx): vector
                },
                spec=spec,
            )
        )
        rows.append(
            {
                "keyframe_id": keyframe.keyframe_id,
                "video_id": keyframe.video_id,
                "frame_idx": str(keyframe.frame_idx),
                "shot_id": str(keyframe.shot_id),
                "timestamp": str(keyframe.timestamp),
                "image_path": keyframe.image_path,
            }
        )

    target = reuse_embedding_artifacts(
        sources,
        embeddings_root=tmp_path / "features/embeddings",
        rows=rows,
        spec=spec,
    )

    assert sorted(path.name for path in target.glob("*.npy")) == [
        "L21_V001.npy",
        "L21_V002.npy",
    ]
    assert audit_embedding_artifact(target) == {
        "expected_keyframes": 2,
        "vectors": 2,
        "unique_ids": 2,
        "missing_ids": 0,
        "extra_ids": 0,
        "invalid_vectors": 0,
    }
