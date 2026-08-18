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

        def _search(text: str) -> tuple[list[Any], float]:
            if not text:
                return [], 0.0
            start_time = time.perf_counter()
            candidates = search_kis(
                text,
                top_k=self.top_k,
                semantic_service=self.semantic_service,
                multimodal_service=None,
                aic_registry=self.aic_registry,
            )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return candidates, elapsed

        # 1. vi_raw
        c_vi, lat_vi = _search(variant.query_vi)
        records.append(evaluate_ablation_variant(query.query_id, "vi_raw", c_vi, gt, lat_vi))

        # 2. en_full
        c_en, lat_en = _search(variant.query_en_full)
        records.append(evaluate_ablation_variant(query.query_id, "en_full", c_en, gt, lat_en))

        # 3. qwen_keyword
        c_qwen, lat_qwen = _search(variant.keyword_query)
        records.append(evaluate_ablation_variant(query.query_id, "qwen_keyword", c_qwen, gt, lat_qwen))

        # 4. en_full_qwen_rrf
        # Timing for RRF includes the fusion time, but we reuse the search results.
        # Wait, the latency for RRF in a real system might be running both concurrently or sequentially.
        # We will record the max latency of the two branches + fusion time.
        start_rrf = time.perf_counter()
        c_rrf = reciprocal_rank_fusion([c_en, c_qwen], rrf_k=self.rrf_k, top_k=self.top_k)
        fusion_time = (time.perf_counter() - start_rrf) * 1000.0
        
        # Approximate parallel execution latency
        lat_rrf = max(lat_en, lat_qwen) + fusion_time
        records.append(evaluate_ablation_variant(query.query_id, "en_full_qwen_rrf", c_rrf, gt, lat_rrf))

        return records
