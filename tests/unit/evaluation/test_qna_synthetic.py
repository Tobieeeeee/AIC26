"""Ground truth built backwards from a known frame.

Real ground truth is expensive and, at three queries, permanently capped:
the p2/p3 queries are reference queries whose answers lie outside the data
we hold. Of those three, only one is currently scorable, so every decision
about answering is being made on n=1.

The usual direction is query -> search -> frame -> a human confirms. The
expensive, error-prone step is the human confirmation, and it already went
wrong once during verification.

This inverts it: **choose the frame first**, read its OCR, and derive a
question whose answer is a span of that text. `video_id` and `frame_id`
cannot be wrong because they are the input, not a search result.

What that does and does not buy, stated plainly because the distinction
decides how the numbers may be read:

    correct by construction      video_id, frame_id
                                 the answer appears on that frame

    NOT guaranteed               the answer matches what a human sees
                                 (OCR mangles text: "Khánh Vĩnh" was read
                                 as "Khónh Vinh" on the real p1-15 frame)

                                 the answer is unique to that frame
                                 (handled by a uniqueness filter, not by
                                 construction)

So this measures **answer extraction** -- given context X and question Q,
does the model return span S -- where OCR errors cancel out because both
sides use the same X. It does NOT measure whether the system produces what
a human would call the answer, and it does NOT measure retrieval at all,
since any query derived from the frame's own text leaks the answer.
"""
from __future__ import annotations

import pytest

from mvr.evaluation.qna_synthetic import (
    SyntheticQnaCase,
    drop_ambiguous_cases,
    generate_cases,
)

#: Verbatim EasyOCR output from L30_V072 keyframe 034 -- the one frame the
#: whole system has ever scored on. If the generator cannot reproduce this
#: case, it is not modelling the thing being measured.
P1_15_OCR = (
    "EM YcÙNG ĐẾN Xã Giang Ly huyên Khónh Vinh, tỉnh Khánh Hoà 04.08/2024 TRUÒNG"
)


def _cases(ocr_text: str) -> list[SyntheticQnaCase]:
    return generate_cases(
        ocr_text, video_id="L30_V072", keyframe_id="034", frame_id=1745
    )


# -- the anchor: reproduce a human-verified case ------------------------


def test_the_generator_reproduces_the_one_verified_case() -> None:
    """The frame names three administrative levels, so three questions is
    the correct yield -- one per level, each with its own answer:

        "Xã Giang Ly huyên Khónh Vinh, tỉnh Khánh Hoà"
         ^^^ ^^^^^^^ ^^^^^ ^^^^^^^^^^  ^^^^ ^^^^^^^^^

    The commune one is the human-verified case; the other two are honest
    spans of the same text, OCR damage included.
    """
    admin = [c for c in _cases(P1_15_OCR) if c.template == "administrative-unit"]

    by_unit = {c.question: c.answer for c in admin}
    commune = next(q for q in by_unit if "xã" in q.lower())
    assert by_unit[commune] == "Giang Ly"

    case = next(c for c in admin if c.answer == "Giang Ly")
    assert (case.video_id, case.frame_id) == ("L30_V072", 1745)


def test_a_date_is_found_even_with_ocr_mangled_separators() -> None:
    """Real OCR read "04.08.2024" as "04.08/2024" -- mixed separators."""
    dates = [c for c in _cases(P1_15_OCR) if c.template == "date"]

    assert len(dates) == 1
    assert dates[0].answer == "04.08/2024"


# -- the invariant that makes the set trustworthy -----------------------


@pytest.mark.parametrize(
    "ocr_text",
    [
        P1_15_OCR,
        "Phường Bến Nghé Quận 1 ngày 12-05-2023",
        "Tỉnh Khánh Hoà",
        "không có gì đáng chú ý ở đây",
        "",
    ],
)
def test_every_answer_is_a_span_of_the_context(ocr_text: str) -> None:
    """The guarantee the whole approach rests on. Enforced, not hoped for."""
    for case in generate_cases(ocr_text, video_id="V", keyframe_id="k", frame_id=1):
        assert case.answer in case.ocr_text


def test_a_frame_with_no_recognizable_pattern_yields_nothing() -> None:
    """Silence is the right answer -- inventing a question from text that
    supports none would put a wrong expectation into the benchmark."""
    assert _cases("mờ nhoè không đọc được") == []


def test_blank_ocr_yields_nothing() -> None:
    assert _cases("") == []
    assert _cases("   ") == []


# -- administrative units -----------------------------------------------


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("Xã", "xã"),
        ("Phường", "phường"),
        ("Thị trấn", "thị trấn"),
        ("Huyện", "huyện"),
        ("Quận", "quận"),
        ("Tỉnh", "tỉnh"),
        ("Thành phố", "thành phố"),
    ],
)
def test_each_administrative_unit_is_recognised(unit: str, expected: str) -> None:
    cases = [
        c for c in _cases(f"{unit} Giang Ly") if c.template == "administrative-unit"
    ]

    assert len(cases) == 1
    assert cases[0].answer == "Giang Ly"
    assert expected in cases[0].question.lower()


def test_the_unit_word_is_never_part_of_the_answer() -> None:
    """The mistake that made a correct answer score zero on the real run."""
    case = next(c for c in _cases("Xã Giang Ly") if c.template == "administrative-unit")

    assert not case.answer.lower().startswith("xã")


# -- ambiguity filtering ------------------------------------------------


def test_an_answer_appearing_on_many_frames_is_dropped() -> None:
    """The scoring rule requires frame_id inside the ground-truth interval,
    so a case whose answer is visible on hundreds of frames would mark a
    right answer wrong. That understates performance, which is the more
    dangerous direction for a benchmark."""
    cases = [
        SyntheticQnaCase(
            video_id="V", keyframe_id=f"{n:03d}", frame_id=n * 10,
            question="q?", answer="THỜI SỰ", ocr_text="THỜI SỰ", template="banner",
        )
        for n in range(1, 30)
    ] + [
        SyntheticQnaCase(
            video_id="V", keyframe_id="099", frame_id=990,
            question="q?", answer="Giang Ly", ocr_text="Xã Giang Ly",
            template="administrative-unit",
        )
    ]

    kept = drop_ambiguous_cases(cases, max_occurrences=3)

    assert [c.answer for c in kept] == ["Giang Ly"]


def test_an_answer_repeated_a_few_times_survives() -> None:
    cases = [
        SyntheticQnaCase(
            video_id="V", keyframe_id=f"{n:03d}", frame_id=n * 10,
            question="q?", answer="Giang Ly", ocr_text="Xã Giang Ly",
            template="administrative-unit",
        )
        for n in (1, 2)
    ]

    assert len(drop_ambiguous_cases(cases, max_occurrences=3)) == 2


def test_ambiguity_is_judged_on_the_normalized_answer() -> None:
    """"GIANG LY" and "Giang Ly" are the same answer for scoring."""
    cases = [
        SyntheticQnaCase(
            video_id="V", keyframe_id=f"{n:03d}", frame_id=n * 10, question="q?",
            answer=answer, ocr_text=answer, template="banner",
        )
        for n, answer in enumerate(["Giang Ly", "GIANG LY", "giang ly", "Giang  Ly"], 1)
    ]

    assert drop_ambiguous_cases(cases, max_occurrences=3) == []


# -- feeding the evaluator ----------------------------------------------


def test_a_case_converts_to_the_evaluator_ground_truth() -> None:
    case = next(c for c in _cases(P1_15_OCR) if c.answer == "Giang Ly")

    gt = case.to_ground_truth()

    assert (gt.video_id, gt.start_frame, gt.end_frame, gt.answer) == (
        "L30_V072", 1745, 1745, "Giang Ly"
    )


def test_the_interval_is_the_single_frame_it_was_built_from() -> None:
    """Widening it would be inventing tolerance the organizer did not grant."""
    case = next(c for c in _cases(P1_15_OCR) if c.template == "date")

    assert case.to_ground_truth().start_frame == case.to_ground_truth().end_frame


# -- false positives found by running on 60 real frames -----------------


def test_diacritics_are_not_folded_away_when_matching_a_unit_word() -> None:
    """Vietnamese diacritics carry meaning. Folding them made "quân" (army)
    fire the "quận" (district) rule, and "tính"/"tình" fire "tỉnh"
    (province). On real frames every single generated case was a false
    positive from exactly this.

    For a ground-truth generator a false positive is far worse than a miss:
    it writes a wrong expectation into the benchmark, where a miss only
    makes the benchmark smaller.
    """
    assert _cases("quân đội nhân dân Việt Nam") == []
    assert _cases("tính chất Cách Mạng") == []
    assert _cases("tình hình Biển Đông") == []


def test_an_ocr_damaged_unit_word_is_skipped_rather_than_guessed() -> None:
    """The cost of the rule above, accepted deliberately: the real p1-15
    frame read "huyện" as "huyên", and that case is now lost. Losing it
    keeps the other three false positives out."""
    cases = _cases("huyên Khónh Vinh")

    assert cases == []


def test_a_whole_sentence_is_not_accepted_as_a_place_name() -> None:
    """One real false positive answered "SỰ CỦA TA TRONG KHÁNG CHlẾN CHỐNG
    PHÁP" -- seven words. No Vietnamese place name runs that long."""
    cases = _cases("Quận SỰ CỦA TA TRONG KHÁNG CHIẾN CHỐNG PHÁP")

    assert cases == []


def test_a_three_word_name_is_truncated_and_must_be_caught_by_review() -> None:
    """The known cost of the two-word cap, recorded rather than hidden.

    "Thành phố Hồ Chí Minh" yields "Hồ Chí" -- wrong. It was wrong before
    the cap too ("HỔ CHÍ MINH Công", swallowing the next phrase), so the cap
    did not cause this; it only changed the shape of the failure.

    Boundary detection in noisy all-caps OCR has no reliable signal, and
    hand-tuning heuristics against a handful of examples is the same
    overfitting this whole module exists to escape. Generated cases are
    therefore CANDIDATES: a human accepts them before they become a
    benchmark. At n~17 that review costs about two minutes.
    """
    assert [c.answer for c in _cases("Thành phố Hồ Chí Minh")] == ["Hồ Chí"]


# -- precision fixes measured on 571 cached real frames ------------------


def test_a_name_is_at_most_two_words() -> None:
    """Every correct name across 571 real frames was one or two words:
    Tân Bình, Hóc Môn, Gia Lai, Chư Pơng, Giang Ly, Kursk. A longer run only
    ever meant the match had walked into the next phrase -- "THỦ ĐỨC NGAy
    SAcH" for "Thủ Đức"."""
    cases = _cases("THÀNH PHỐ THỦ ĐỨC NGAy SAcH Vun")

    assert [c.answer for c in cases] == ["THỦ ĐỨC"]


def test_a_token_with_no_vowel_is_ocr_damage_not_a_name() -> None:
    """Real output included "TỈNH Qws" -- three consonants. Every Vietnamese
    and transliterated place name has a vowel in each syllable."""
    assert _cases("TỈNH Qws & n 20") == []


def test_the_vowel_rule_does_not_touch_dates() -> None:
    """Dates are digits and separators; the rule is about names only."""
    assert [c.answer for c in _cases("ngày 31/12/2025")] == ["31/12/2025"]
