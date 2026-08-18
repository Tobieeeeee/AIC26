"""Unit tests for the TKIS Query-Variant Retrieval Ablation framework."""
import json
from pathlib import Path

from mvr.retrieval.kis import KisCandidate
from mvr.evaluation.aic_kis import KisGroundTruth
from scripts.verification.query_retrieval_ablation.inputs import AblationQuery, load_ablation_queries
from scripts.verification.query_retrieval_ablation.fusion import reciprocal_rank_fusion
from scripts.verification.query_retrieval_ablation.metrics import evaluate_ablation_variant, compute_pairwise_comparison
from scripts.verification.query_retrieval_ablation.models import AblationVariantRecord


def test_load_ablation_queries(tmp_path: Path) -> None:
    gt_file = tmp_path / "gt.json"
    gt_file.write_text(json.dumps({
        "queries": [
            {
                "query_id": "q1",
                "query_vi": "Test vi",
                "video_id": "L21",
                "start_frame": 10,
                "end_frame": 20
            }
        ]
    }))

    queries = load_ablation_queries(gt_file)
    assert len(queries) == 1
    assert queries[0].query_id == "q1"
    assert queries[0].query_vi == "Test vi"
    assert queries[0].video_id == "L21"


def test_reciprocal_rank_fusion() -> None:
    c1 = KisCandidate(rank=1, video_id="V1", frame_id=10, score=0.9, keyframe_id="kf1")
    c2 = KisCandidate(rank=2, video_id="V2", frame_id=20, score=0.8, keyframe_id="kf2")

    c3 = KisCandidate(rank=1, video_id="V1", frame_id=10, score=0.8, keyframe_id="kf1")
    c4 = KisCandidate(rank=2, video_id="V3", frame_id=30, score=0.7, keyframe_id="kf3")

    fused = reciprocal_rank_fusion([[c1, c2], [c3, c4]], rrf_k=60, top_k=2)

    # V1,10 should be rank 1 since it's rank 1 in both
    assert len(fused) == 2
    assert fused[0].video_id == "V1"
    assert fused[0].frame_id == 10

    # V2,20 or V3,30 will be next
    # Their ranks are 2 in their respective lists, so their RRF scores are equal
    # but V2 comes first in the deterministic sort since "V2" < "V3"
    assert fused[1].video_id == "V2"


def test_evaluate_ablation_variant() -> None:
    gt = KisGroundTruth(video_id="V1", start_frame=10, end_frame=20)
    c1 = KisCandidate(rank=1, video_id="V2", frame_id=10, score=0.9, keyframe_id="kf1")
    c2 = KisCandidate(rank=2, video_id="V1", frame_id=15, score=0.8, keyframe_id="kf2")

    record = evaluate_ablation_variant("q1", "vi_raw", [c1, c2], gt, 10.5)

    assert record.first_correct_rank == 2
    assert record.r_at_1 == 0.0
    assert record.r_at_5 == 1.0


def test_pairwise_comparison() -> None:
    r1 = AblationVariantRecord("q1", "vi_raw", first_correct_rank=5, r_at_1=0, r_at_5=1, r_at_20=1, r_at_50=1, r_at_100=1, final_score=0.8, retrieval_latency_ms=10)
    r2 = AblationVariantRecord("q1", "en_full", first_correct_rank=1, r_at_1=1, r_at_5=1, r_at_20=1, r_at_50=1, r_at_100=1, final_score=1.0, retrieval_latency_ms=10)

    comp = compute_pairwise_comparison([r2], [r1], "en_full", "vi_raw")

    assert comp.wins == 1
    assert comp.moves_earlier == 1
    assert comp.rescues == 0

from scripts.verification.query_retrieval_ablation.models import QueryVariant
from scripts.verification.query_retrieval_ablation.variants import freeze_query_variants
from scripts.verification.query_retrieval_ablation.runner import AblationRunner

def test_pairwise_both_miss() -> None:
    r1 = AblationVariantRecord("q1", "vi_raw", first_correct_rank=None, r_at_1=0, r_at_5=0, r_at_20=0, r_at_50=0, r_at_100=0, final_score=0.0, retrieval_latency_ms=10)
    r2 = AblationVariantRecord("q1", "en_full", first_correct_rank=None, r_at_1=0, r_at_5=0, r_at_20=0, r_at_50=0, r_at_100=0, final_score=0.0, retrieval_latency_ms=10)
    comp = compute_pairwise_comparison([r2], [r1], "en_full", "vi_raw")
    assert comp.ties == 1
    assert comp.wins == 0
    assert comp.losses == 0
    assert comp.rescues == 0
    assert comp.regressions == 0
    assert comp.moves_earlier == 0
    assert comp.moves_later == 0

def test_evaluate_empty_candidates() -> None:
    gt = KisGroundTruth(video_id="V1", start_frame=10, end_frame=20)
    record = evaluate_ablation_variant("q1", "vi_raw", [], gt, 10.0, "some_error")
    assert record.first_correct_rank is None
    assert record.r_at_1 == 0.0
    assert record.final_score == 0.0
    assert record.error == "some_error"

def test_rrf_top_k_enforcement() -> None:
    c1 = KisCandidate(rank=1, video_id="V1", frame_id=10, score=0.9, keyframe_id="kf1")
    c2 = KisCandidate(rank=2, video_id="V2", frame_id=20, score=0.8, keyframe_id="kf2")
    c3 = KisCandidate(rank=3, video_id="V3", frame_id=30, score=0.7, keyframe_id="kf3")
    fused = reciprocal_rank_fusion([[c1, c2, c3], []], rrf_k=60, top_k=2)
    assert len(fused) == 2
    assert fused[0].video_id == "V1"
    assert fused[1].video_id == "V2"

def test_rrf_empty_branch() -> None:
    c1 = KisCandidate(rank=1, video_id="V1", frame_id=10, score=0.9, keyframe_id="kf1")
    fused = reciprocal_rank_fusion([[], [c1]], rrf_k=60, top_k=2)
    assert len(fused) == 1
    assert fused[0].video_id == "V1"

class DummyTranslator:
    name = "dummy_trans"
    def translate(self, text: str) -> str: return text + "_en"

class DummyDecomposer:
    name = "dummy_qwen"
    def decompose(self, text: str) -> tuple[list[str], str, str]: return (["concept"], text + "_kw", "raw")

def test_frozen_variant_cache_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "variants.json"
    queries = [AblationQuery("q1", "Test vi", "V1", 10, 20)]
    variants1 = freeze_query_variants(queries, DummyTranslator(), DummyDecomposer(), manifest_path)
    assert len(variants1) == 1
    assert variants1[0].translation_success is True
    assert variants1[0].decomposition_success is True
    assert variants1[0].query_en_full == "Test vi_en"

    class BrokenTranslator:
        name = "broken"
        def translate(self, text: str) -> str: raise ValueError()

    variants2 = freeze_query_variants(queries, BrokenTranslator(), DummyDecomposer(), manifest_path)
    assert len(variants2) == 1
    assert variants2[0].query_en_full == "Test vi_en"
    assert variants2[0].translation_model == "dummy_trans"

class DummySemanticService: pass
class DummyAicRegistry: pass

def test_runner_translation_and_qwen_failures() -> None:
    runner = AblationRunner(DummySemanticService(), DummyAicRegistry())
    from unittest.mock import patch
    with patch("scripts.verification.query_retrieval_ablation.runner.search_kis") as mock_search:
        mock_search.return_value = [KisCandidate(1, "V1", 10, 1.0, "kf1")]
        q = AblationQuery("q1", "vi", "V1", 10, 20)

        # Scenario A: translation fails
        v_a = QueryVariant("q1", "vi", "", "kw", "t", "q", "v1", False, True)
        recs_a = runner.run_query(q, v_a)
        assert next(r for r in recs_a if r.variant == "en_full").error == "translation_failed"
        assert next(r for r in recs_a if r.variant == "en_full_qwen_rrf").error == "rrf_missing_en_branch"

        # Scenario B: Qwen fails
        v_b = QueryVariant("q1", "vi", "en", "", "t", "q", "v1", True, False)
        recs_b = runner.run_query(q, v_b)
        assert next(r for r in recs_b if r.variant == "qwen_keyword").error == "decomposition_failed"
        assert next(r for r in recs_b if r.variant == "en_full_qwen_rrf").error == "rrf_missing_keyword_branch"

        # Scenario C: empty mapped candidates
        v_c = QueryVariant("q1", "vi", "en", "kw", "t", "q", "v1", True, True)
        mock_search.return_value = []
        recs_c = runner.run_query(q, v_c)
        for r in recs_c:
            assert r.error == "empty_mapped_candidates"
