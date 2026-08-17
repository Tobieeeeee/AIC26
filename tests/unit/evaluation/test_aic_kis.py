from mvr.evaluation.aic_kis import (
    KIS_CUTOFFS,
    KisAnswer,
    KisGroundTruth,
    evaluate_kis,
    kis_r_score,
)

GT = KisGroundTruth(video_id="L21_V001", start_frame=250, end_frame=270)


def test_r_score_inside_interval_is_one() -> None:
    assert kis_r_score(KisAnswer("L21_V001", 261), GT) == 1.0


def test_r_score_outside_interval_is_zero() -> None:
    assert kis_r_score(KisAnswer("L21_V001", 300), GT) == 0.0


def test_r_score_wrong_video_is_zero_even_inside_frame_range() -> None:
    assert kis_r_score(KisAnswer("L21_V002", 261), GT) == 0.0


def test_r_score_inclusive_start_boundary() -> None:
    assert kis_r_score(KisAnswer("L21_V001", 250), GT) == 1.0


def test_r_score_inclusive_end_boundary() -> None:
    assert kis_r_score(KisAnswer("L21_V001", 270), GT) == 1.0


def test_r_score_just_outside_boundaries_is_zero() -> None:
    assert kis_r_score(KisAnswer("L21_V001", 249), GT) == 0.0
    assert kis_r_score(KisAnswer("L21_V001", 271), GT) == 0.0


def test_evaluate_kis_empty_answer_list_scores_zero() -> None:
    score = evaluate_kis([], GT)

    assert score.recall_at_k == {k: 0.0 for k in KIS_CUTOFFS}
    assert score.final_score == 0.0


def test_evaluate_kis_fewer_than_five_answers_still_scores_correctly() -> None:
    answers = [
        KisAnswer("L21_V001", 100),
        KisAnswer("L21_V001", 261),
        KisAnswer("L21_V001", 400),
    ]

    score = evaluate_kis(answers, GT)

    # Correct answer is rank 2, within every cutoff even though there are
    # only 3 answers total (fewer than the k=5 cutoff itself).
    assert score.recall_at_k[1] == 0.0
    for k in (5, 20, 50, 100):
        assert score.recall_at_k[k] == 1.0
    assert score.final_score == 4 / 5


def test_evaluate_kis_correct_answer_rank_1() -> None:
    answers = [KisAnswer("L21_V001", 261)] + [
        KisAnswer("L21_V001", 999) for _ in range(10)
    ]

    score = evaluate_kis(answers, GT)

    assert all(score.recall_at_k[k] == 1.0 for k in KIS_CUTOFFS)
    assert score.final_score == 1.0


def test_evaluate_kis_correct_answer_rank_3() -> None:
    answers = (
        [KisAnswer("L21_V001", 999), KisAnswer("L21_V001", 998)]
        + [KisAnswer("L21_V001", 261)]
        + [KisAnswer("L21_V001", 999) for _ in range(10)]
    )

    score = evaluate_kis(answers, GT)

    assert score.recall_at_k[1] == 0.0
    for k in (5, 20, 50, 100):
        assert score.recall_at_k[k] == 1.0


def test_evaluate_kis_correct_answer_rank_21() -> None:
    wrong = [KisAnswer("L21_V001", 999) for _ in range(20)]
    answers = wrong + [KisAnswer("L21_V001", 261)]

    score = evaluate_kis(answers, GT)

    assert score.recall_at_k[1] == 0.0
    assert score.recall_at_k[5] == 0.0
    assert score.recall_at_k[20] == 0.0
    assert score.recall_at_k[50] == 1.0
    assert score.recall_at_k[100] == 1.0
    assert score.final_score == 2 / 5


def test_evaluate_kis_answer_after_rank_100_is_ignored() -> None:
    wrong = [KisAnswer("L21_V001", 999) for _ in range(100)]
    answers = wrong + [KisAnswer("L21_V001", 261)]  # rank 101

    score = evaluate_kis(answers, GT)

    assert score.recall_at_k == {k: 0.0 for k in KIS_CUTOFFS}
    assert score.final_score == 0.0


def test_evaluate_kis_real_data_example_from_handoff_doc() -> None:
    """Cross-check against the verified real AIC mapping example: keyframe
    L21_V001/003.jpg resolves to frame_idx=261 (see AIC_DATA_HANDOFF.md)."""
    gt = KisGroundTruth(video_id="L21_V001", start_frame=255, end_frame=265)
    correct_prediction = KisAnswer(video_id="L21_V001", frame_id=261)

    score = evaluate_kis([correct_prediction], gt)

    assert score.final_score == 1.0
