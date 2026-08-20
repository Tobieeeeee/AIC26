from __future__ import annotations

import os
from pathlib import Path

import pytest

from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder
from mvr.enrichment.embeddings.offline import (
    audit_embedding_artifact,
    extract_embedding_artifacts,
    load_canonical_keyframes,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs


_MAPPING = os.environ.get("MVR_BEIT3_FASTTEST_MAPPING")
_KEYFRAMES = os.environ.get("MVR_BEIT3_FASTTEST_KEYFRAMES")
_CHECKPOINT = os.environ.get("MVR_BEIT3_CHECKPOINT")


@pytest.mark.skipif(
    not all((_MAPPING, _KEYFRAMES, _CHECKPOINT)),
    reason="requires explicit local BEiT-3 checkpoint and five-keyframe fixture",
)
def test_real_beit3_five_keyframe_fasttest(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    spec = load_provider_specs(repo_root / "configs" / "embedding_providers.json").get(
        "beit3"
    )
    rows = load_canonical_keyframes(Path(_MAPPING))
    assert len(rows) == 5
    provider_root = extract_embedding_artifacts(
        rows,
        keyframe_root=Path(_KEYFRAMES),
        embeddings_root=tmp_path / "embeddings",
        spec=spec,
        encoder=Beit3ImageEncoder(spec=spec, checkpoint_path=Path(_CHECKPOINT)),
    )
    assert audit_embedding_artifact(provider_root) == {
        "expected_keyframes": 5,
        "vectors": 5,
        "unique_ids": 5,
        "missing_ids": 0,
        "extra_ids": 0,
        "invalid_vectors": 0,
    }
