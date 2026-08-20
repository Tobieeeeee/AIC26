import pytest

from mvr.evaluation.trake import (
    TRAKE_CUTOFFS,
    TrakeChainPrediction,
    TrakeGroundTruth,
    evaluate_trake,
    trake_final_round_score,
    trake_r_score,
)

# -- trake_r_score: cross-checked against trake_align.py's own self-check --
# assert r_score(7, (1, 3, 5), 7, (1, 4, 20)) == 2 / 3
# assert r_score(7, (1, 3, 5), 8, (1, 3, 5)) == 0.0


def test_r_score_wrong_video_is_zero_even_with_matching_frames() -> None:
    gt = TrakeGroundTruth(video_id=8, frames=(1, 3, 5))
    prediction = TrakeChainPrediction(video_id=7, frames=(1, 3, 5))

    assert trake_r_score(prediction, gt) == 0.0


def test_r_score_partial_credit_matches_lab_reference_example() -> None:
    gt = TrakeGroundTruth(video_id=7, frames=(1, 4, 20))
    prediction = TrakeChainPrediction(video_id=7, frames=(1, 3, 5))

    # Marker 1: |1-1|=0 <= 1 -> correct. Marker 2: |3-4|=1 <= 1 -> correct.
    # Marker 3: |5-20|=15 > 1 -> wrong. 2/3 correct, matching trake_align.py.
    assert trake_r_score(prediction, gt) == 2 / 3


def test_r_score_inclusive_tolerance_boundary() -> None:
    gt = TrakeGroundTruth(video_id="v", frames=(100,), tolerance=2)

    assert trake_r_score(TrakeChainPrediction("v", (102,)), gt) == 1.0  # exactly tol
    assert trake_r_score(TrakeChainPrediction("v", (103,)), gt) == 0.0  # tol + 1


def test_r_score_none_marker_never_counts_as_correct() -> None:
    gt = TrakeGroundTruth(video_id="v", frames=(10, 20, 30))
    prediction = TrakeChainPrediction("v", (10, None, 30))

    assert trake_r_score(prediction, gt) == 2 / 3


def test_r_score_none_marker_does_not_match_unknown_gt_placeholder() -> None:
    # Even if a GT frame were somehow the same sentinel-like value, an
    # unresolved (None) prediction must never be treated as correct.
    gt = TrakeGroundTruth(video_id="v", frames=(10, 0, 30))
    prediction = TrakeChainPrediction("v", (10, None, 30))

    assert trake_r_score(prediction, gt) == 2 / 3


def test_r_score_rejects_marker_count_mismatch() -> None:
    gt = TrakeGroundTruth(video_id="v", frames=(1, 2, 3))
    prediction = TrakeChainPrediction("v", (1, 2))

    with pytest.raises(ValueError):
        trake_r_score(prediction, gt)


# -- evaluate_trake ----------------------------------------------------------

GT = TrakeGroundTruth(video_id="L21_V001", frames=(100, 200, 300), tolerance=1)
CORRECT = TrakeChainPrediction("L21_V001", (100, 200, 300))
WRONG = TrakeChainPrediction("L21_V001", (999, 999, 999))


def test_evaluate_trake_empty_prediction_list_scores_zero() -> None:
    score = evaluate_trake([], GT)

    assert score.recall_at_k == {k: 0.0 for k in TRAKE_CUTOFFS}
    assert score.final_score == 0.0


def test_evaluate_trake_correct_chain_rank_1() -> None:
    predictions = [CORRECT] + [WRONG for _ in range(10)]

    score = evaluate_trake(predictions, GT)

    assert all(score.recall_at_k[k] == 1.0 for k in TRAKE_CUTOFFS)
    assert score.final_score == 1.0


def test_evaluate_trake_running_max_across_cutoffs_at_rank_21() -> None:
    predictions = [WRONG for _ in range(20)] + [CORRECT]  # correct chain at rank 21

    score = evaluate_trake(predictions, GT)

    assert score.recall_at_k[1] == 0.0
    assert score.recall_at_k[5] == 0.0
    assert score.recall_at_k[20] == 0.0
    assert score.recall_at_k[50] == 1.0
    assert score.recall_at_k[100] == 1.0
    assert score.final_score == 2 / 5


def test_evaluate_trake_uses_best_partial_score_within_cutoff() -> None:
    # A partially-correct chain (2/3) ranked above a wrong chain (0/3) must
    # win the running max for every cutoff it's within.
    partial = TrakeChainPrediction("L21_V001", (100, 200, 999))
    predictions = [partial, WRONG]

    score = evaluate_trake(predictions, GT)

    assert score.recall_at_k[1] == 2 / 3
    for k in (5, 20, 50, 100):
        assert score.recall_at_k[k] == 2 / 3


def test_evaluate_trake_prediction_after_rank_100_is_ignored() -> None:
    predictions = [WRONG for _ in range(100)] + [CORRECT]  # rank 101

    score = evaluate_trake(predictions, GT)

    assert score.recall_at_k == {k: 0.0 for k in TRAKE_CUTOFFS}
    assert score.final_score == 0.0


# -- trake_final_round_score: cross-checked against trake_align.py's own
# self-check --
# assert final_score(60, 0) == 90.0 and final_score(120, 1) == 70.0
# assert final_score(290, 6) == 0.0


def test_final_round_score_matches_lab_reference_values() -> None:
    assert trake_final_round_score(60, 0) == 90.0
    assert trake_final_round_score(120, 1) == 70.0


def test_final_round_score_never_negative() -> None:
    assert trake_final_round_score(290, 6) == 0.0
