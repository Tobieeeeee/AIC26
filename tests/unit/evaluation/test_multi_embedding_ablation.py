from __future__ import annotations

import json

import pytest

from mvr.evaluation.multi_embedding_ablation import (
    ABLATIONS,
    AblationConfig,
    MultiEmbeddingFasttestError,
    deterministic_report_payload,
    render_ablation_markdown,
    run_ablation_suite,
)
from mvr.retrieval.provider_semantic import (
    IndependentSemanticResponse,
    ProviderSemanticResult,
)


def _result(provider: str, keyframe_id: str, rank: int):
    index = int(keyframe_id[1:])
    return ProviderSemanticResult(
        rank=rank,
        score=1.0 / rank,
        model_id=provider,
        keyframe_id=keyframe_id,
        video_id="V1",
        frame_id=index * 10,
        shot_id=index,
        timestamp=index / 2,
    )


def _config(**kwargs):
    values = {
        "run_id": "test-run",
        "queries": ({"id": "Q1", "text": "query"},),
        # (video_id, keyframe_id): the bare id would validate a result
        # from any video, and 842 organizer videos share each id.
        "canonical_ids": frozenset(
            ("V1", f"K{i}") for i in range(1, 6)
        ),
        "warmup_runs": 1,
        "measured_runs": 2,
    }
    values.update(kwargs)
    return AblationConfig(**values)


def _retrieve(query, providers, top_k):
    lists = {
        provider: tuple(_result(provider, f"K{i}", i) for i in range(1, 6))
        for provider in providers
    }
    return IndependentSemanticResponse(
        ranked_lists=lists,
        requested_providers=tuple(providers),
        successful_providers=tuple(providers),
        failed_providers=(),
        errors={},
        failure_policy="fail_closed",
        timings_ms={
            provider: {"encoding_ms": 1.0, "search_ms": 2.0}
            for provider in providers
        },
    )


def test_all_seven_ablation_rows_and_bounded_quality_report() -> None:
    ticks = iter(float(index) / 1000 for index in range(10000))
    report = run_ablation_suite(_config(), _retrieve, clock=lambda: next(ticks))
    assert len(report["ablations"]) == 7
    assert [tuple(row["providers"]) for row in report["ablations"]] == [
        providers for _, providers in ABLATIONS
    ]
    assert all(row["structural_status"] == "pass" for row in report["ablations"])
    assert all("not evaluated" in row["quality"]["R@1"] for row in report["ablations"])
    assert "no provider combination is claimed to be better" in report["conclusion"]
    assert report["ablations"][0]["latency"]["encoding_by_provider"]["beit3"]["median_ms"] == 1.0
    markdown = render_ablation_markdown(report)
    assert markdown.count("| A_beit3 |") == 1
    assert markdown.count("| G_all_three_rrf |") == 1


def test_report_is_deterministic_except_timing_and_runtime() -> None:
    first = run_ablation_suite(_config(), _retrieve)
    second = run_ablation_suite(_config(), _retrieve)
    assert deterministic_report_payload(first) == deterministic_report_payload(second)


def test_noncanonical_result_fails_structural_gate() -> None:
    def invalid(query, providers, top_k):
        response = _retrieve(query, providers, top_k)
        provider = providers[0]
        lists = dict(response.ranked_lists)
        lists[provider] = (_result(provider, "K99", 1),)
        return IndependentSemanticResponse(
            ranked_lists=lists,
            requested_providers=response.requested_providers,
            successful_providers=response.successful_providers,
            failed_providers=(),
            errors={},
            failure_policy="fail_closed",
            timings_ms=response.timings_ms,
        )

    with pytest.raises(MultiEmbeddingFasttestError, match="non-canonical"):
        run_ablation_suite(_config(warmup_runs=0, measured_runs=1), invalid)


def test_count_and_provider_coverage_mismatches_fail() -> None:
    with pytest.raises(MultiEmbeddingFasttestError, match="count"):
        _config(canonical_ids=frozenset({("V1", "K1")}))

    def missing_provider(query, providers, top_k):
        response = _retrieve(query, providers, top_k)
        lists = dict(response.ranked_lists)
        lists.pop(providers[-1])
        return IndependentSemanticResponse(
            ranked_lists=lists,
            requested_providers=tuple(providers),
            successful_providers=tuple(lists),
            failed_providers=(providers[-1],),
            errors={providers[-1]: "failed"},
            failure_policy="degraded",
        )

    with pytest.raises(MultiEmbeddingFasttestError, match="coverage mismatch"):
        run_ablation_suite(
            _config(warmup_runs=0, measured_runs=1), missing_provider
        )


def test_machine_report_is_json_serializable() -> None:
    report = run_ablation_suite(
        _config(warmup_runs=0, measured_runs=1), _retrieve
    )
    assert json.loads(json.dumps(report))["run_id"] == "test-run"
