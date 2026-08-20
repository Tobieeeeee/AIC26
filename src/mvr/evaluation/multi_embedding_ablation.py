"""Seven fixed structural ablations for the multi-embedding fasttest."""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mvr.retrieval.provider_semantic import IndependentSemanticResponse
from mvr.retrieval.semantic_rrf import fuse_independent_response


PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A_beit3", ("beit3",)),
    ("B_metaclip2", ("metaclip2",)),
    ("C_perception_encoder", ("perception_encoder",)),
    ("D_beit3_metaclip2_rrf", ("beit3", "metaclip2")),
    ("E_beit3_perception_encoder_rrf", ("beit3", "perception_encoder")),
    ("F_metaclip2_perception_encoder_rrf", ("metaclip2", "perception_encoder")),
    ("G_all_three_rrf", PROVIDERS),
)


class MultiEmbeddingFasttestError(ValueError):
    """Raised when structural fasttest invariants fail."""


@dataclass(frozen=True)
class AblationConfig:
    run_id: str
    queries: tuple[dict[str, str], ...]
    # (video_id, keyframe_id). A keyframe_id alone does not identify a
    # keyframe: 842 of the organizer's videos contain one called "034",
    # so a result from the wrong video would validate against this set.
    canonical_ids: frozenset[tuple[str, str]]
    expected_keyframes: int = 5
    rrf_k: int = 60
    top_k: int = 5
    warmup_runs: int = 1
    measured_runs: int = 3
    query_set_version: str = "fasttest-smoke-v1"
    ground_truth_version: str | None = None
    representative_ground_truth: bool = False
    provider_provenance: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if len(self.canonical_ids) != self.expected_keyframes:
            raise MultiEmbeddingFasttestError("canonical identity count must equal five")
        if self.rrf_k <= 0 or self.top_k <= 0:
            raise MultiEmbeddingFasttestError("rrf_k and top_k must be positive")
        if self.warmup_runs < 0 or self.measured_runs <= 0:
            raise MultiEmbeddingFasttestError("invalid warmup/measured run counts")
        if not self.queries:
            raise MultiEmbeddingFasttestError("at least one smoke query is required")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "median_ms": None, "p95_ms": None}
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(np.asarray(values), 95)),
    }


def _quality_not_evaluated(config: AblationConfig) -> dict[str, str]:
    reason = (
        "not evaluated: representative ground truth unavailable"
        if not config.representative_ground_truth
        else "not evaluated: evaluator not configured"
    )
    return {
        "R@1": reason,
        "R@5": reason,
        "R@20": reason,
        "R@50": reason,
        "R@100": reason,
        "final_score": reason,
    }


def run_ablation_suite(
    config: AblationConfig,
    retrieve: Callable[[str, Sequence[str], int], IndependentSemanticResponse],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run all seven fixed configurations without changing ranking logic."""

    rows = []
    for ablation_id, provider_ids in ABLATIONS:
        query_latencies: list[float] = []
        fusion_latencies: list[float] = []
        provider_encoding: dict[str, list[float]] = {provider: [] for provider in provider_ids}
        provider_search: dict[str, list[float]] = {provider: [] for provider in provider_ids}
        final_results: dict[str, list[dict[str, Any]]] = {}
        failures: dict[str, str] = {}
        for query in config.queries:
            text = query["text"]
            for _ in range(config.warmup_runs):
                retrieve(text, provider_ids, config.top_k)
            for measured_index in range(config.measured_runs):
                started = clock()
                response = retrieve(text, provider_ids, config.top_k)
                retrieval_finished = clock()
                if set(response.ranked_lists) != set(provider_ids):
                    raise MultiEmbeddingFasttestError(
                        f"{ablation_id} provider coverage mismatch"
                    )
                for provider_id, results in response.ranked_lists.items():
                    if any(
                        (result.video_id, result.keyframe_id) not in config.canonical_ids
                        for result in results
                    ):
                        raise MultiEmbeddingFasttestError(
                            f"{provider_id} returned a non-canonical identity"
                        )
                fusion_started = clock()
                if len(provider_ids) == 1:
                    only = response.ranked_lists[provider_ids[0]][: config.top_k]
                    output = [item.to_dict() for item in only]
                else:
                    output = [
                        item.to_dict()
                        for item in fuse_independent_response(
                            response, rrf_k=config.rrf_k, top_k=config.top_k
                        ).results
                    ]
                fusion_finished = clock()
                query_latencies.append((retrieval_finished - started) * 1000.0)
                fusion_latencies.append((fusion_finished - fusion_started) * 1000.0)
                for provider_id, timing in response.timings_ms.items():
                    provider_encoding[provider_id].append(float(timing["encoding_ms"]))
                    provider_search[provider_id].append(float(timing["search_ms"]))
                failures.update(response.errors)
                if measured_index == config.measured_runs - 1:
                    final_results[query["id"]] = output
        rows.append(
            {
                "ablation_id": ablation_id,
                "providers": list(provider_ids),
                "rrf_k": config.rrf_k if len(provider_ids) > 1 else None,
                "structural_status": "pass",
                "quality": _quality_not_evaluated(config),
                "latency": {
                    "query": _latency_summary(query_latencies),
                    "fusion": _latency_summary(fusion_latencies),
                    "encoding_by_provider": {
                        provider: _latency_summary(values)
                        for provider, values in provider_encoding.items()
                    },
                    "search_by_provider": {
                        provider: _latency_summary(values)
                        for provider, values in provider_search.items()
                    },
                    "warmup_runs_excluded": config.warmup_runs,
                    "model_loading_latency": "not included in measured query latency after warm-up",
                },
                "provider_failures": dict(sorted(failures.items())),
                "results": final_results,
            }
        )
    return {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "provider_sets": [list(providers) for _, providers in ABLATIONS],
        "expected_keyframes": config.expected_keyframes,
        "canonical_ids": [
            f"{video_id}/{keyframe_id}"
            for video_id, keyframe_id in sorted(config.canonical_ids)
        ],
        "query_set_version": config.query_set_version,
        "ground_truth_version": config.ground_truth_version,
        "representative_ground_truth": config.representative_ground_truth,
        "provider_provenance": dict(config.provider_provenance or {}),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "ablations": rows,
        "conclusion": (
            "Structural and latency smoke only; quality was not evaluated and no "
            "provider combination is claimed to be better."
        ),
    }


def render_ablation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Multi-embedding fasttest and ablation report",
        "",
        f"Run ID: `{report['run_id']}`",
        "",
        "| Ablation | Providers | Structural | Quality | Query median ms | Fusion median ms |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in report["ablations"]:
        quality = row["quality"]["R@1"]
        lines.append(
            "| {id} | {providers} | {status} | {quality} | {query} | {fusion} |".format(
                id=row["ablation_id"],
                providers=", ".join(row["providers"]),
                status=row["structural_status"],
                quality=quality,
                query=row["latency"]["query"]["median_ms"],
                fusion=row["latency"]["fusion"]["median_ms"],
            )
        )
    lines.extend(["", "## Bounded conclusion", "", str(report["conclusion"]), ""])
    return "\n".join(lines)


def deterministic_report_payload(report: Mapping[str, Any]) -> str:
    """Serialize after removing nondeterministic timing/runtime fields."""

    stable = json.loads(json.dumps(report))
    stable.pop("runtime", None)
    for row in stable["ablations"]:
        row.pop("latency", None)
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
