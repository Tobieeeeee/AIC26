from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mvr.indexing.provider_indexes import (
    FaissProviderBackend,
    ProviderIndexContract,
)
from mvr.retrieval.provider_semantic import FaissProviderSearchBackend


def test_real_faiss_provider_backend_persists_isolated_collection(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    contract = ProviderIndexContract(
        model_id="beit3",
        collection_name="test__fasttest__beit3",
        dimension=2,
        metric="inner_product",
        normalized=True,
        backend="faiss_index_flat_ip",
    )
    backend = FaissProviderBackend(tmp_path)
    assert backend.prepare(contract, "create") == 0
    assert backend.insert_batch(
        contract,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [
            {"keyframe_id": "K1", "video_id": "V1", "frame_id": 1, "shot_id": 1, "timestamp": 0.1},
            {"keyframe_id": "K2", "video_id": "V1", "frame_id": 2, "shot_id": 2, "timestamp": 0.2},
        ],
    ) == []
    assert backend.finalize(contract) == 2
    target = tmp_path / contract.collection_name
    index = faiss.read_index(str(target / "keyframes.faiss"))
    assert index.d == 2
    assert index.ntotal == 2
    assert (target / "vectors.npy").is_file()
    assert (target / "metadata.json").is_file()
    hits = FaissProviderSearchBackend(tmp_path).search(
        contract, np.asarray([1.0, 0.0], dtype=np.float32), top_k=2
    )
    assert [hit.metadata["keyframe_id"] for hit in hits] == ["K1", "K2"]
    assert hits[0].score == pytest.approx(1.0)
