from mvr.evaluation.aic_qna import (
    QNA_CUTOFFS,
    QnaAnswer,
    QnaGroundTruth,
    evaluate_qna,
    qna_answer_matches,
    qna_r_score,
)

GT = QnaGroundTruth(video_id="L05_V005", start_frame=800, end_frame=900, answer="màu xanh")


def test_r_score_perfect_match_is_one() -> None:
    assert qna_r_score(QnaAnswer("L05_V005", 888, "màu xanh"), GT) == 1.0


def test_r_score_wrong_answer_is_zero() -> None:
    assert qna_r_score(QnaAnswer("L05_V005", 888, "màu trắng"), GT) == 0.0


def test_r_score_wrong_video_is_zero_even_with_correct_answer() -> None:
    assert qna_r_score(QnaAnswer("L06_V007", 888, "màu xanh"), GT) == 0.0


def test_r_score_frame_outside_interval_is_zero_even_with_correct_answer() -> None:
    assert qna_r_score(QnaAnswer("L05_V005", 950, "màu xanh"), GT) == 0.0


def test_r_score_inclusive_frame_boundaries() -> None:
    assert qna_r_score(QnaAnswer("L05_V005", 800, "màu xanh"), GT) == 1.0
    assert qna_r_score(QnaAnswer("L05_V005", 900, "màu xanh"), GT) == 1.0


def test_r_score_answer_match_is_case_and_whitespace_insensitive() -> None:
    assert qna_r_score(QnaAnswer("L05_V005", 850, "  MÀU XANH  "), GT) == 1.0


def test_organizer_worked_example_number_word_equivalence() -> None:
    gt = QnaGroundTruth(video_id="video_xyz", start_frame=3400, end_frame=3500, answer="5")
    assert qna_r_score(QnaAnswer("video_xyz", 3450, "5"), gt) == 1.0
    assert qna_r_score(QnaAnswer("video_xyz", 3450, "Năm"), gt) == 1.0
    assert qna_r_score(QnaAnswer("video_xyz", 3450, "năm"), gt) == 1.0


def test_organizer_worked_example_reverse_gt_direction() -> None:
    gt = QnaGroundTruth(video_id="video_xyz", start_frame=3400, end_frame=3500, answer="Năm")
    assert qna_r_score(QnaAnswer("video_xyz", 3450, "5"), gt) == 1.0


def test_english_number_word_equivalence() -> None:
    gt = QnaGroundTruth(video_id="v1", start_frame=1, end_frame=10, answer="three")
    assert qna_r_score(QnaAnswer("v1", 5, "3"), gt) == 1.0


def test_answer_matches_rejects_unrelated_number_words() -> None:
    assert qna_answer_matches("5", "6") is False
    assert qna_answer_matches("five", "6") is False


def test_evaluate_qna_empty_answer_list_scores_zero() -> None:
    score = evaluate_qna([], GT)

    assert score.recall_at_k == {k: 0.0 for k in QNA_CUTOFFS}
    assert score.final_score == 0.0


def test_evaluate_qna_correct_answer_rank_3() -> None:
    answers = [
        QnaAnswer("L05_V005", 888, "màu đỏ"),
        QnaAnswer("L05_V005", 888, "màu vàng"),
        QnaAnswer("L05_V005", 888, "màu xanh"),
    ] + [QnaAnswer("L05_V005", 999, "wrong") for _ in range(10)]

    score = evaluate_qna(answers, GT)

    assert score.recall_at_k[1] == 0.0
    for k in (5, 20, 50, 100):
        assert score.recall_at_k[k] == 1.0


def test_evaluate_qna_correct_video_frame_but_wrong_answer_scores_zero() -> None:
    answers = [QnaAnswer("L05_V005", 888, "màu trắng")]

    score = evaluate_qna(answers, GT)

    assert score.final_score == 0.0


def test_evaluate_qna_answer_after_rank_100_is_ignored() -> None:
    wrong = [QnaAnswer("L05_V005", 999, "wrong") for _ in range(100)]
    answers = wrong + [QnaAnswer("L05_V005", 888, "màu xanh")]  # rank 101

    score = evaluate_qna(answers, GT)

    assert score.final_score == 0.0


def test_evaluate_qna_organizer_worked_example() -> None:
    """Cross-check against the organizer doc's own Q&A worked example."""
    gt = QnaGroundTruth(video_id="video_xyz", start_frame=3450, end_frame=3450, answer="5")
    correct = QnaAnswer(video_id="video_xyz", frame_id=3450, answer="Năm")

    score = evaluate_qna([correct], gt)

    assert score.final_score == 1.0
