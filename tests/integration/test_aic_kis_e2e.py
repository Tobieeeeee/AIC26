"""Task 08 — Textual KIS end-to-end verification on real local AIC data.

Opt-in only. Proves the complete documented path for real, with no mocks:

    original AIC videos -> video/frame registry -> existing indexes/retrievers
    -> KIS orchestration -> optional exact-frame refinement
    -> ranked Top<=100 original-frame predictions -> official evaluator
    -> prediction/export adapter -> KIS API response
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MVR_RUN_AIC_KIS_E2E") != "1",
    reason="set MVR_RUN_AIC_KIS_E2E=1 (and MVR_AIC_DATASET_ROOT) to run the real AIC KIS E2E",
)


def _require_dataset_root() -> Path:
    raw = os.environ.get("MVR_AIC_DATASET_ROOT", "").strip()
    if not raw:
        pytest.skip("set MVR_AIC_DATASET_ROOT to run this real-data E2E test")
    root = Path(raw)
    if not root.is_dir():
        pytest.skip(f"MVR_AIC_DATASET_ROOT does not exist: {root}")
    return root


def _require_real_index() -> Path:
    index_root = Path("artifacts/indexes/aic_l21")
    required = ("keyframes.faiss", "metadata.json", "manifest.json")
    if not all((index_root / name).is_file() for name in required):
        pytest.skip(
            f"real FAISS index not found at {index_root} "
            "(run the Task 06 index build first)"
        )
    return index_root


def test_full_kis_path_end_to_end_on_real_data(tmp_path: Path) -> None:
    from mvr.api.app import LazyFaissSemanticService
    from mvr.evaluation.aic_kis import KisAnswer, KisGroundTruth, evaluate_kis
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.ingestion.aic_dataset import AicDatasetRegistry
    from mvr.retrieval.frame_refinement import RefinementWindowConfig, refine_top_candidates
    from mvr.retrieval.kis import search_kis
    from mvr.submission.kis_export import (
        export_local_verification_json,
        load_local_verification_json,
        validate_kis_predictions,
    )
    from mvr.enrichment.embeddings.openclip import OpenClipEncoder

    dataset_root = _require_dataset_root()
    index_root = _require_real_index()

    # 1. Frame registry over the real AIC dataset (Task 02).
    registry = AicDatasetRegistry(dataset_root)

    # 2. Existing index/retriever, loaded exactly as production would (Task 01 baseline).
    faiss_index = FaissIndex.load(index_root)
    semantic_service = LazyFaissSemanticService(index_root, device="cpu")
    assert semantic_service.available, "real FAISS index must be usable, not just present"

    # 3. KIS orchestration (Task 04): a real natural-language query, real search.
    # top_k=20 matches the strongest official cutoff verified for this query
    # below (recall_at_k[20]) -- the correct answer is at real rank 6.
    query_text = "A geyser of water shooting high into the sky from a dry well in a field"
    candidates = search_kis(
        query_text, top_k=20, semantic_service=semantic_service,
        multimodal_service=None, aic_registry=registry,
    )
    assert candidates, "KIS orchestration must return at least one resolved candidate"
    assert all(isinstance(c.frame_id, int) for c in candidates)
    assert len({(c.video_id, c.frame_id) for c in candidates}) == len(candidates)

    # 4. Optional exact-frame refinement (Task 05): real FFmpeg decode + real
    #    encoder rescoring on the top-1 candidate only, to keep this test fast.
    encoder = OpenClipEncoder(
        model_name=faiss_index.manifest["embedding_model"],
        pretrained=faiss_index.manifest["pretrained"],
        device="cpu",
    )
    query_vector = encoder.encode_text([query_text])[0]
    refined = refine_top_candidates(
        candidates, query_vector=query_vector, encoder=encoder,
        video_path_for=registry.video_path, video_info_for=registry.video_info,
        window=RefinementWindowConfig(frame_radius=15), top_n=1, enabled=True,
    )
    assert refined[0].video_id == candidates[0].video_id
    assert isinstance(refined[0].frame_id, int)

    # 5. Official evaluator (Task 03) against the verified real GT interval
    #    for this query (see artifacts/verification/aic_kis/gt_queries.json).
    #
    # This must prove actual correctness, not just that scoring executed:
    # a completely wrong retrieval also satisfies `0.0 <= final_score <= 1.0`.
    # The real correct answer for this query is verified to appear at rank 6
    # (see Task 06's records.json and the PR #2 hardening investigation), so
    # recall_at_k[20] == 1.0 is the strongest official cutoff (1/5/20/50/100)
    # supported by measured evidence for the current index/model -- do not
    # tighten this to recall_at_k[5] without re-verifying the real rank.
    gt = KisGroundTruth(video_id="L21_V001", start_frame=5670, end_frame=5850)
    answers = [KisAnswer(video_id=c.video_id, frame_id=c.frame_id) for c in refined]
    score = evaluate_kis(answers, gt)
    assert 0.0 <= score.final_score <= 1.0
    assert score.recall_at_k[20] == 1.0, (
        "the known real-data query must retrieve a correct answer "
        f"(video_id={gt.video_id}, frame in [{gt.start_frame},{gt.end_frame}]) "
        f"within the top 20; got candidates={[(c.video_id, c.frame_id) for c in refined]}"
    )

    # 6. Prediction/export adapter (Task 07): validate + local round-trip.
    predictions = validate_kis_predictions(refined)
    assert len(predictions) <= 100
    assert [p.rank for p in predictions] == list(range(1, len(predictions) + 1))
    export_path = tmp_path / "e2e_export.json"
    export_local_verification_json(predictions, export_path, query_id="e2e_smoke")
    round_tripped = load_local_verification_json(export_path)
    assert round_tripped == predictions

    # 7. KIS API response (Task 04's route), exercised through the real ASGI
    #    app wired to the real services -- not a fake.
    from fastapi.testclient import TestClient

    from mvr.api.app import create_app

    app = create_app(
        semantic_service=semantic_service, multimodal_service=None,
        kis_pipeline_resolver=None, kis_aic_registry=registry,
    )
    client = TestClient(app)
    response = client.post("/search/kis", json={"query": query_text, "top_k": 20})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == len(candidates)
    assert payload["results"][0]["video_id"] == candidates[0].video_id
    assert payload["results"][0]["frame_id"] == candidates[0].frame_id
