"""Ablation benchmark runner."""
from __future__ import annotations

import time
from typing import Any

from mvr.evaluation.aic_kis import KisGroundTruth
from mvr.retrieval.kis import search_kis
from .fusion import reciprocal_rank_fusion
from .inputs import AblationQuery
from .metrics import evaluate_ablation_variant
from .models import AblationVariantRecord, QueryVariant


class AblationRunner:
    def __init__(
        self,
        semantic_service: Any,
        aic_registry: Any,
        top_k: int = 100,
        rrf_k: int = 60,
    ) -> None:
        self.semantic_service = semantic_service
        self.aic_registry = aic_registry
        self.top_k = top_k
        self.rrf_k = rrf_k

    def run_query(self, query: AblationQuery, variant: QueryVariant) -> list[AblationVariantRecord]:
        records: list[AblationVariantRecord] = []
        gt = KisGroundTruth(
            video_id=query.video_id,
            start_frame=query.start_frame,
            end_frame=query.end_frame,
        )

        def _search(text: str) -> tuple[list[Any], float, str | None]:
            if not text:
                return [], 0.0, None
            start_time = time.perf_counter()
            candidates = search_kis(
                text,
                top_k=self.top_k,
                semantic_service=self.semantic_service,
                multimodal_service=None,
                aic_registry=self.aic_registry,
            )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            err = "empty_mapped_candidates" if not candidates else None
            return candidates, elapsed, err

        # 1. vi_raw
        if not variant.query_vi:
            c_vi, lat_vi, err_vi = [], 0.0, "empty_vi_query"
        else:
            c_vi, lat_vi, err_vi = _search(variant.query_vi)
        records.append(evaluate_ablation_variant(query.query_id, "vi_raw", c_vi, gt, lat_vi, err_vi))

        # 2. en_full
        if not variant.translation_success or not variant.query_en_full:
            c_en, lat_en, err_en = [], 0.0, "translation_failed"
        else:
            c_en, lat_en, err_en = _search(variant.query_en_full)
        records.append(evaluate_ablation_variant(query.query_id, "en_full", c_en, gt, lat_en, err_en))

        # 3. qwen_keyword
        if not variant.decomposition_success or not variant.keyword_query:
            c_qwen, lat_qwen, err_qwen = [], 0.0, "decomposition_failed"
        else:
            c_qwen, lat_qwen, err_qwen = _search(variant.keyword_query)
        records.append(evaluate_ablation_variant(query.query_id, "qwen_keyword", c_qwen, gt, lat_qwen, err_qwen))

        # 4. en_full_qwen_rrf
        err_rrf = None
        if err_en == "translation_failed":
            err_rrf = "rrf_missing_en_branch"
        elif err_qwen == "decomposition_failed":
            err_rrf = "rrf_missing_keyword_branch"

        if err_rrf:
            c_rrf, lat_rrf = [], 0.0
        else:
            start_rrf = time.perf_counter()
            c_rrf = reciprocal_rank_fusion([c_en, c_qwen], rrf_k=self.rrf_k, top_k=self.top_k)
            fusion_time = (time.perf_counter() - start_rrf) * 1000.0
            # Approximate parallel execution latency
            lat_rrf = max(lat_en, lat_qwen) + fusion_time
            if not c_rrf:
                err_rrf = "empty_mapped_candidates"

        records.append(evaluate_ablation_variant(query.query_id, "en_full_qwen_rrf", c_rrf, gt, lat_rrf, err_rrf))

        return records
