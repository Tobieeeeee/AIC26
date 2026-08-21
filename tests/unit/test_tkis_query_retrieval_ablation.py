"""The TKIS ablation harness: scoring, aggregation, and pairwise comparison.

This file was committed empty, together with `metrics.py` and the CLI, by
three commits whose messages say they added a benchmark, hardened its
failure handling, and separated quality metrics from failures (`8f4e32a`,
`6456027`, `0040a9b`). Nothing was there: `runner.py:9` imports
`evaluate_ablation_variant` from an empty module, so the harness has never
been importable, let alone run.
"""
from __future__ import annotations

import pytest

from mvr.evaluation.aic_kis import KisGroundTruth
from scripts.verification.query_retrieval_ablation.metrics import (
    compare_variants,
    evaluate_ablation_variant,
    summarize_variant,
)


class _Candidate:
    """Only what scoring reads off a `KisCandidate`."""

    def __init__(self, video_id: str, frame_id: int) -> None:
        self.video_id = video_id
        self.frame_id = frame_id


GT = KisGroundTruth(video_id="L21_V001", start_frame=100, end_frame=200)


def _hit() -> _Candidate:
    return _Candidate("L21_V001", 150)


def _miss(frame_id: int = 5000) -> _Candidate:
    return _Candidate("L21_V002", frame_id)


# -- scoring one query -------------------------------------------------


def test_a_correct_answer_at_rank_one_scores_every_cutoff() -> None:
    record = evaluate_ablation_variant("q1", "en_full", [_hit()], GT, 12.5)

    assert record.first_correct_rank == 1
    assert (record.r_at_1, record.r_at_5, record.r_at_100) == (1.0, 1.0, 1.0)
    assert record.final_score == 1.0
    assert record.error is None


def test_a_correct_answer_at_rank_three_misses_only_r_at_1() -> None:
    """The official rule is interval containment at a cutoff, not set recall:
    R@5 is 1.0 the moment any of the first five answers lands inside the
    ground-truth interval."""
    candidates = [_miss(1), _miss(2), _hit()]

    record = evaluate_ablation_variant("q1", "en_full", candidates, GT, 1.0)

    assert record.first_correct_rank == 3
    assert record.r_at_1 == 0.0
    assert record.r_at_5 == 1.0
    assert record.final_score == pytest.approx(4 / 5)


def test_the_right_video_at_the_wrong_frame_scores_nothing() -> None:
    """The failure this whole branch of work is about: a well-formed answer
    naming the correct video and a frame outside the interval is worth
    exactly as much as naming the wrong video."""
    record = evaluate_ablation_variant(
        "q1", "en_full", [_Candidate("L21_V001", 5000)], GT, 1.0
    )

    assert record.first_correct_rank is None
    assert record.final_score == 0.0


def test_answers_beyond_rank_100_cannot_score() -> None:
    candidates = [_miss(i) for i in range(100)] + [_hit()]

    record = evaluate_ablation_variant("q1", "en_full", candidates, GT, 1.0)

    assert record.first_correct_rank is None
    assert record.final_score == 0.0


def test_a_failed_query_is_recorded_as_an_error_not_as_a_zero() -> None:
    """A variant that could not run is not a variant that scored zero. Mixing
    them makes a translation outage look like a retrieval result."""
    record = evaluate_ablation_variant(
        "q1", "en_full", [], GT, 0.0, error="translation_failed"
    )

    assert record.error == "translation_failed"
    assert record.first_correct_rank is None
    assert record.final_score == 0.0


# -- aggregating a variant ---------------------------------------------


def test_summary_averages_only_the_queries_that_ran() -> None:
    """The distinction the empty `metrics.py` was committed three times to
    make: failures move `failure_rate`, never the quality mean."""
    records = [
        evaluate_ablation_variant("q1", "en_full", [_hit()], GT, 10.0),
        evaluate_ablation_variant("q2", "en_full", [_miss()], GT, 20.0),
        evaluate_ablation_variant(
            "q3", "en_full", [], GT, 0.0, error="translation_failed"
        ),
    ]

    summary = summarize_variant("en_full", records)

    assert (summary.attempted_queries, summary.successful_queries) == (3, 2)
    assert summary.failed_queries == 1
    assert summary.error_counts == {"translation_failed": 1}
    # q1 scored 1.0, q2 scored 0.0; the failed q3 is not in the mean.
    assert summary.r_at_1 == 0.5
    assert summary.final_score == 0.5
    assert summary.mean_latency_ms == 15.0


def test_a_variant_that_never_ran_reports_none_rather_than_zero() -> None:
    """Zero is a measurement. None is the absence of one, and a report that
    prints 0.0000 for a variant nothing ran invites the wrong conclusion."""
    records = [
        evaluate_ablation_variant(
            "q1", "qwen_keyword", [], GT, 0.0, error="decomposition_failed"
        )
    ]

    summary = summarize_variant("qwen_keyword", records)

    assert summary.successful_queries == 0
    assert summary.failure_rate == 1.0
    assert summary.r_at_1 is None
    assert summary.final_score is None
    assert summary.mean_latency_ms is None


def test_two_different_errors_are_counted_separately() -> None:
    """One aggregate failure count cannot distinguish an Elasticsearch outage
    from a model that will not load, and those need different responses."""
    records = [
        evaluate_ablation_variant("q1", "rrf", [], GT, 0.0, error="translation_failed"),
        evaluate_ablation_variant("q2", "rrf", [], GT, 0.0, error="translation_failed"),
        evaluate_ablation_variant(
            "q3", "rrf", [], GT, 0.0, error="empty_mapped_candidates"
        ),
    ]

    summary = summarize_variant("rrf", records)

    assert summary.error_counts == {
        "translation_failed": 2,
        "empty_mapped_candidates": 1,
    }
    assert summary.failure_rate == 1.0


# -- comparing two variants --------------------------------------------


def test_comparison_counts_from_the_candidate_side() -> None:
    """`variant_a` is the BASELINE, `variant_b` the CANDIDATE, and every
    counter reads from B's side: `wins` means B beat A.

    Stated loudly because getting it backwards does not fail anything -- it
    silently inverts the conclusion of the whole benchmark.
    """
    baseline = [
        evaluate_ablation_variant("q1", "en_full", [_miss(1), _hit()], GT, 1.0),
        evaluate_ablation_variant("q2", "en_full", [_hit()], GT, 1.0),
    ]
    candidate = [
        evaluate_ablation_variant("q1", "vi_lex_en_dense", [_hit()], GT, 1.0),
        evaluate_ablation_variant("q2", "vi_lex_en_dense", [_hit()], GT, 1.0),
    ]

    comparison = compare_variants("en_full", "vi_lex_en_dense", baseline, candidate)

    assert comparison.valid_pairs == 2
    assert comparison.wins == 1          # q1: rank 2 -> rank 1
    assert comparison.ties == 1          # q2: unchanged
    assert comparison.losses == 0
    assert comparison.moves_earlier == 1
    assert comparison.moves_later == 0


def test_a_pair_is_excluded_when_either_side_failed() -> None:
    """Scoring a query the baseline could not attempt would credit the
    candidate for the baseline's outage."""
    baseline = [
        evaluate_ablation_variant(
            "q1", "en_full", [], GT, 0.0, error="translation_failed"
        )
    ]
    candidate = [evaluate_ablation_variant("q1", "vi_lex_en_dense", [_hit()], GT, 1.0)]

    comparison = compare_variants("en_full", "vi_lex_en_dense", baseline, candidate)

    assert comparison.attempted_pairs == 1
    assert comparison.valid_pairs == 0
    assert comparison.excluded_pairs == 1
    assert comparison.wins == 0


def test_rescues_and_regressions_are_counted_separately() -> None:
    """A rescue -- nothing to something -- is the outcome worth the most and
    is invisible in a mean over two queries."""
    baseline = [
        evaluate_ablation_variant("q1", "en_full", [_miss()], GT, 1.0),
        evaluate_ablation_variant("q2", "en_full", [_hit()], GT, 1.0),
    ]
    candidate = [
        evaluate_ablation_variant("q1", "vi_lex_en_dense", [_hit()], GT, 1.0),
        evaluate_ablation_variant("q2", "vi_lex_en_dense", [_miss()], GT, 1.0),
    ]

    comparison = compare_variants("en_full", "vi_lex_en_dense", baseline, candidate)

    assert comparison.rescues == 1
    assert comparison.regressions == 1
    assert comparison.wins == 1
    assert comparison.losses == 1


def test_queries_present_in_only_one_variant_are_not_compared() -> None:
    baseline = [evaluate_ablation_variant("q1", "en_full", [_hit()], GT, 1.0)]
    candidate = [
        evaluate_ablation_variant("q1", "vi_lex_en_dense", [_hit()], GT, 1.0),
        evaluate_ablation_variant("q9", "vi_lex_en_dense", [_hit()], GT, 1.0),
    ]

    comparison = compare_variants("en_full", "vi_lex_en_dense", baseline, candidate)

    assert comparison.attempted_pairs == 1
    assert comparison.valid_pairs == 1


def test_a_duplicate_query_id_within_one_variant_is_refused() -> None:
    """Two records for one query means the caller ran something twice; taking
    either silently would make the comparison depend on dict ordering."""
    records = [
        evaluate_ablation_variant("q1", "en_full", [_hit()], GT, 1.0),
        evaluate_ablation_variant("q1", "en_full", [_miss()], GT, 1.0),
    ]

    with pytest.raises(ValueError, match="duplicate query_id"):
        compare_variants("en_full", "vi_lex_en_dense", records, [])


# -- the runner ---------------------------------------------------------


def _variant(**overrides):
    from scripts.verification.query_retrieval_ablation.models import QueryVariant

    defaults = dict(
        query_id="q1",
        query_vi="xe máy màu đỏ",
        query_en_full="a red motorbike",
        keyword_query="motorbike red",
        translation_model="fake",
        qwen_model="fake",
        qwen_schema_version="v1",
        translation_success=True,
        decomposition_success=True,
    )
    return QueryVariant(**{**defaults, **overrides})


def _query():
    from scripts.verification.query_retrieval_ablation.inputs import AblationQuery

    return AblationQuery(
        query_id="q1", query_vi="xe máy màu đỏ",
        video_id="L21_V001", start_frame=100, end_frame=200,
    )


class _RecordingService:
    """Captures what each branch was asked, and answers with one correct hit."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def search_hybrid(self, query, *, top_k, rrf_k, lexical_query=None):
        self.calls.append((query, lexical_query))
        return [
            {
                "video_id": "L21_V001",
                "frame_id": "034",
                "frame_idx": 150,
                "timestamp": 6.0,
                "score": 1.0,
            }
        ]


def test_the_runner_produces_one_record_per_variant() -> None:
    from scripts.verification.query_retrieval_ablation.runner import AblationRunner

    service = _RecordingService()
    records = AblationRunner(
        semantic_service=None, aic_registry=None, multimodal_service=service
    ).run_query(_query(), _variant())

    assert [r.variant for r in records] == [
        "vi_raw",
        "en_full",
        "qwen_keyword",
        "en_full_qwen_rrf",
        "vi_lex_en_dense",
    ]


def test_only_the_split_variant_sends_two_different_strings() -> None:
    """Every other variant's name promises one string to both branches.
    Splitting them all would change what those rows measure while leaving
    their labels alone -- which is how a benchmark starts lying."""
    from scripts.verification.query_retrieval_ablation.runner import AblationRunner

    service = _RecordingService()
    AblationRunner(
        semantic_service=None, aic_registry=None, multimodal_service=service
    ).run_query(_query(), _variant())

    # `lexical_query=None` means "send this one string to both branches",
    # so it is not a split -- only an explicit second string is.
    split = [call for call in service.calls if call[1] is not None]
    assert split == [("a red motorbike", "xe máy màu đỏ")]


def test_the_split_variant_refuses_to_run_without_a_lexical_branch() -> None:
    """With no lexical branch this variant IS en_full. Recording it as a
    result would put a duplicate row beside its own baseline and make the
    split look like it changed nothing."""
    from scripts.verification.query_retrieval_ablation.runner import AblationRunner

    records = AblationRunner(
        semantic_service=None, aic_registry=None, multimodal_service=_RecordingService()
    ).run_query(_query(), _variant())
    with_branch = next(r for r in records if r.variant == "vi_lex_en_dense")
    assert with_branch.error is None

    class _Semantic:
        def search(self, query, top_k):
            return []

    records = AblationRunner(
        semantic_service=_Semantic(), aic_registry=None
    ).run_query(_query(), _variant())
    without = next(r for r in records if r.variant == "vi_lex_en_dense")
    assert without.error == "lexical_branch_unavailable"


def test_a_failed_translation_marks_both_english_variants() -> None:
    from scripts.verification.query_retrieval_ablation.runner import AblationRunner

    records = AblationRunner(
        semantic_service=None, aic_registry=None, multimodal_service=_RecordingService()
    ).run_query(_query(), _variant(translation_success=False, query_en_full=""))

    errors = {r.variant: r.error for r in records}
    assert errors["en_full"] == "translation_failed"
    assert errors["vi_lex_en_dense"] == "translation_failed"
    assert errors["vi_raw"] is None


# -- the CLI's offline halves ------------------------------------------


def _cli():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "verification" / "run_tkis_query_retrieval_ablation.py"
    )
    spec = importlib.util.spec_from_file_location("_tkis_ablation_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_cli_module_imports_without_a_model_or_elasticsearch() -> None:
    """The whole harness has never been importable: `runner.py` pulled
    `evaluate_ablation_variant` from an empty `metrics.py`. Nothing here may
    need a GPU or a running Elasticsearch just to load."""
    cli = _cli()

    assert cli.VARIANTS[-1] == "vi_lex_en_dense"
    assert cli.HEADLINE_COMPARISON == ("en_full", "vi_lex_en_dense")


def test_no_elasticsearch_means_no_lexical_branch_rather_than_a_crash() -> None:
    cli = _cli()

    assert cli.build_multimodal_service(object(), "", "") is None
    assert cli.build_multimodal_service(object(), "http://localhost:9200", "") is None


def test_a_duplicate_query_id_in_the_manifest_is_refused(tmp_path) -> None:
    """Two frozen variants for one query means the manifest was appended to
    rather than rebuilt; taking either silently would make the benchmark
    depend on file order."""
    import json

    cli = _cli()
    row = _variant().to_dict()
    path = tmp_path / "variants.json"
    path.write_text(json.dumps([row, row]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate query_id"):
        cli.load_variants(path)


def test_a_variant_that_never_ran_prints_dashes_not_zeroes() -> None:
    """`0.0000` reads as "measured, scored nothing". This variant was never
    executed, and the report must not let those look alike."""
    cli = _cli()
    records = [
        evaluate_ablation_variant(
            "q1", "vi_lex_en_dense", [], GT, 0.0, error="lexical_branch_unavailable"
        )
    ]
    summaries = [
        summarize_variant("en_full", [evaluate_ablation_variant("q1", "en_full", [_hit()], GT, 1.0)]),
        summarize_variant("vi_lex_en_dense", records),
    ]
    comparison = compare_variants("en_full", "vi_lex_en_dense", [], records)

    report = cli.render_report(summaries, comparison)

    assert "--" in report
    assert "0.0000" not in report.split("vi_lex_en_dense")[2].split("\n")[0]
    assert "lexical_branch_unavailable" in report
    assert "NOTHING WAS COMPARED" in report
