from __future__ import annotations

import os
from pathlib import Path

import pytest

from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.offline import (
    audit_embedding_artifact,
    extract_embedding_artifacts,
    load_canonical_keyframes,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs


_MAPPING = os.environ.get("MVR_METACLIP2_FASTTEST_MAPPING")
_KEYFRAMES = os.environ.get("MVR_METACLIP2_FASTTEST_KEYFRAMES")
_MODEL = os.environ.get("MVR_METACLIP2_MODEL")


@pytest.mark.skipif(
    not all((_MAPPING, _KEYFRAMES, _MODEL)),
    reason="requires explicit pinned MetaCLIP-2 cache and five-keyframe fixture",
)
def test_real_metaclip2_five_keyframe_fasttest(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    spec = load_provider_specs(repo_root / "configs" / "embedding_providers.json").get(
        "metaclip2"
    )
    rows = load_canonical_keyframes(Path(_MAPPING))
    assert len(rows) == 5
    provider = extract_embedding_artifacts(
        rows,
        keyframe_root=Path(_KEYFRAMES),
        embeddings_root=tmp_path / "embeddings",
        spec=spec,
        encoder=MetaClip2ImageEncoder(spec=spec, model_location=_MODEL),
    )
    assert audit_embedding_artifact(provider) == {
        "expected_keyframes": 5,
        "vectors": 5,
        "unique_ids": 5,
        "missing_ids": 0,
        "extra_ids": 0,
        "invalid_vectors": 0,
    }
