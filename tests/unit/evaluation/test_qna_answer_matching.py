"""Answer matching after the first real end-to-end run exposed two gaps.

The module docstring already said the approximation should be widened
"explicitly in a later task rather than silently" if it proved too
conservative on an observed real query. It did, on the only query the
system has ever answered correctly:

    question : "Hỏi xã này có tên là gì?"   (what is this commune called?)
    OCR      : "... Xã Giang Ly huyên Khónh Vinh, tỉnh Khánh Hoà ..."
    model    : "Xã Giang Ly"
    truth    : "Giang Ly"

Scored 0. A human judge -- which is what the organizer actually uses --
would not call that wrong: "xã" is the administrative unit word, not part
of the name. Stripping a leading unit word is a linguistic normalization
with a defensible rule, not a loosening of the bar.

The second gap is that one number cannot express this honestly. So the
strict matcher stays strict and a separate lenient matcher gives the upper
bound; the benchmark reports both, and the truth sits between them.
"""
from __future__ import annotations

import pytest

from mvr.evaluation.aic_qna import (
    QnaAnswer,
    QnaGroundTruth,
    evaluate_qna,
    qna_answer_matches,
    qna_answer_matches_lenient,
)


# -- administrative-unit prefixes ---------------------------------------


def test_the_real_query_that_motivated_this() -> None:
    assert qna_answer_matches("Xã Giang Ly", "Giang Ly")


@pytest.mark.parametrize(
    "unit",
    ["Xã", "Phường", "Thị trấn", "Thị xã", "Huyện", "Quận", "Tỉnh", "Thành phố"],
)
def test_a_leading_administrative_unit_is_not_part_of_the_name(unit: str) -> None:
    assert qna_answer_matches(f"{unit} Giang Ly", "Giang Ly")


def test_the_prefix_may_sit_on_either_side() -> None:
    """Ground truth is human-written and may carry the unit word itself."""
    assert qna_answer_matches("Giang Ly", "Xã Giang Ly")
    assert qna_answer_matches("Xã Giang Ly", "xã giang ly")


def test_a_bare_unit_word_matches_nothing() -> None:
    """Stripping must never leave an empty string that matches everything."""
    assert not qna_answer_matches("Xã", "Giang Ly")
    assert not qna_answer_matches("Giang Ly", "Tỉnh")


def test_different_names_still_do_not_match() -> None:
    assert not qna_answer_matches("Xã Giang Ly", "Xã Khánh Vĩnh")


def test_only_a_leading_unit_is_stripped_so_strict_stays_strict() -> None:
    """The full address is more than the answer, and still scores 0."""
    assert not qna_answer_matches("Xã Giang Ly huyện Khánh Vĩnh", "Giang Ly")


def test_a_number_answer_keeps_working() -> None:
    """Quận 1 -> 1 is the right reading of 'which district is this?'."""
    assert qna_answer_matches("1", "Quận 1")


# -- lenient upper bound ------------------------------------------------


def test_lenient_accepts_the_answer_inside_a_longer_reply() -> None:
    assert qna_answer_matches_lenient("Xã Giang Ly huyện Khánh Vĩnh", "Giang Ly")


def test_lenient_accepts_everything_strict_accepts() -> None:
    assert qna_answer_matches_lenient("Xã Giang Ly", "Giang Ly")
    assert qna_answer_matches_lenient("1", "Quận 1")


def test_lenient_still_rejects_a_different_answer() -> None:
    assert not qna_answer_matches_lenient("Khánh Vĩnh", "Giang Ly")


def test_lenient_rejects_blanks_rather_than_matching_everything() -> None:
    assert not qna_answer_matches_lenient("", "Giang Ly")
    assert not qna_answer_matches_lenient("Giang Ly", "   ")


# -- threading the matcher through the score ----------------------------


def _answers_with(answer: str) -> list[QnaAnswer]:
    return [QnaAnswer(video_id="L30_V072", frame_id=1745, answer=answer)]


def _gt() -> QnaGroundTruth:
    return QnaGroundTruth(
        video_id="L30_V072", start_frame=1745, end_frame=1776, answer="Giang Ly"
    )


def test_evaluate_qna_defaults_to_the_strict_matcher() -> None:
    lenient_only = _answers_with("Xã Giang Ly huyện Khánh Vĩnh")

    assert evaluate_qna(lenient_only, _gt()).final_score == 0.0


def test_evaluate_qna_accepts_an_explicit_matcher_for_the_upper_bound() -> None:
    lenient_only = _answers_with("Xã Giang Ly huyện Khánh Vĩnh")

    score = evaluate_qna(lenient_only, _gt(), matcher=qna_answer_matches_lenient)

    assert score.final_score == 1.0


def test_the_matcher_cannot_rescue_a_wrong_frame() -> None:
    """Widening answer matching must not widen the interval rule."""
    outside = [QnaAnswer(video_id="L30_V072", frame_id=9999, answer="Giang Ly")]

    assert evaluate_qna(outside, _gt(), matcher=qna_answer_matches_lenient).final_score == 0.0
