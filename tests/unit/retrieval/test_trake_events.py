import pytest

from mvr.retrieval.trake_events import (
    DEFAULT_GAP_SECONDS,
    GAP_PHRASE_TABLE,
    TrakeEventQuery,
    TrakeGapConstraint,
    TrakeQuery,
    parse_organizer_trake_query,
    translate_gap_phrase,
)

# -- translate_gap_phrase -----------------------------------------------------


@pytest.mark.parametrize("phrase,expected", list(GAP_PHRASE_TABLE.items()))
def test_translate_gap_phrase_matches_documented_range(phrase, expected) -> None:
    constraint = translate_gap_phrase(phrase)

    assert (constraint.min_gap_seconds, constraint.max_gap_seconds) == expected


def test_translate_gap_phrase_falls_back_to_default_when_unmatched() -> None:
    constraint = translate_gap_phrase("không có gợi ý thời gian nào cả")

    assert (constraint.min_gap_seconds, constraint.max_gap_seconds) == DEFAULT_GAP_SECONDS


def test_translate_gap_phrase_falls_back_to_default_on_empty_text() -> None:
    constraint = translate_gap_phrase("")

    assert (constraint.min_gap_seconds, constraint.max_gap_seconds) == DEFAULT_GAP_SECONDS


def test_translate_gap_phrase_prefers_more_specific_phrase_over_substring() -> None:
    # "ngay sau đó" contains "sau đó" as a substring; the more specific
    # phrase must win, not be shadowed by the shorter one.
    constraint = translate_gap_phrase("vận động viên giậm nhảy ngay sau đó qua xà")

    assert (constraint.min_gap_seconds, constraint.max_gap_seconds) == GAP_PHRASE_TABLE["ngay sau đó"]
    assert (constraint.min_gap_seconds, constraint.max_gap_seconds) != GAP_PHRASE_TABLE["sau đó"]


def test_translate_gap_phrase_matches_embedded_in_longer_text() -> None:
    constraint = translate_gap_phrase("chạy đà rồi giậm nhảy")

    assert (constraint.min_gap_seconds, constraint.max_gap_seconds) == GAP_PHRASE_TABLE["rồi"]


# -- TrakeGapConstraint --------------------------------------------------------


def test_gap_constraint_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        TrakeGapConstraint(min_gap_seconds=10.0, max_gap_seconds=5.0)


def test_gap_constraint_rejects_negative_min() -> None:
    with pytest.raises(ValueError):
        TrakeGapConstraint(min_gap_seconds=-1.0, max_gap_seconds=5.0)


def test_gap_constraint_rejects_negative_max() -> None:
    with pytest.raises(ValueError):
        TrakeGapConstraint(min_gap_seconds=0.0, max_gap_seconds=-5.0)


def test_gap_constraint_accepts_equal_bounds() -> None:
    constraint = TrakeGapConstraint(min_gap_seconds=5.0, max_gap_seconds=5.0)

    assert constraint.min_gap_seconds == constraint.max_gap_seconds == 5.0


# -- TrakeEventQuery ------------------------------------------------------------


def test_event_query_rejects_index_below_one() -> None:
    with pytest.raises(ValueError):
        TrakeEventQuery(index=0, text="chạy đà")


def test_event_query_rejects_blank_text() -> None:
    with pytest.raises(ValueError):
        TrakeEventQuery(index=1, text="   ")


# -- TrakeQuery -----------------------------------------------------------------


def test_query_rejects_single_event() -> None:
    with pytest.raises(ValueError):
        TrakeQuery(events=(TrakeEventQuery(1, "chạy đà"),), gaps=())


def test_query_rejects_mismatched_gap_count() -> None:
    events = (TrakeEventQuery(1, "chạy đà"), TrakeEventQuery(2, "giậm nhảy"))
    gaps = (
        TrakeGapConstraint(0.0, 10.0),
        TrakeGapConstraint(0.0, 10.0),  # one too many for a 2-event chain
    )

    with pytest.raises(ValueError):
        TrakeQuery(events=events, gaps=gaps)


def test_query_rejects_non_sequential_event_indices() -> None:
    events = (TrakeEventQuery(1, "chạy đà"), TrakeEventQuery(3, "giậm nhảy"))
    gaps = (TrakeGapConstraint(0.0, 10.0),)

    with pytest.raises(ValueError):
        TrakeQuery(events=events, gaps=gaps)


def test_query_rejects_duplicate_event_indices() -> None:
    events = (TrakeEventQuery(1, "chạy đà"), TrakeEventQuery(1, "giậm nhảy"))
    gaps = (TrakeGapConstraint(0.0, 10.0),)

    with pytest.raises(ValueError):
        TrakeQuery(events=events, gaps=gaps)


def test_realistic_four_event_high_jump_query_constructs_successfully() -> None:
    # "Vận động viên chạy đà, rồi giậm nhảy, ngay sau đó qua xà, cuối cùng
    # tiếp đất" -- the lecture's own high-jump example, k=4.
    events = (
        TrakeEventQuery(1, "vận động viên chạy đà"),
        TrakeEventQuery(2, "giậm nhảy"),
        TrakeEventQuery(3, "qua xà"),
        TrakeEventQuery(4, "tiếp đất"),
    )
    gaps = (
        translate_gap_phrase("rồi"),
        translate_gap_phrase("ngay sau đó"),
        translate_gap_phrase(""),  # no clue between E3 and E4
    )

    query = TrakeQuery(events=events, gaps=gaps)

    assert len(query.events) == 4
    assert len(query.gaps) == 3
    assert (query.gaps[0].min_gap_seconds, query.gaps[0].max_gap_seconds) == (0.0, 60.0)
    assert (query.gaps[1].min_gap_seconds, query.gaps[1].max_gap_seconds) == (0.0, 10.0)
    assert (query.gaps[2].min_gap_seconds, query.gaps[2].max_gap_seconds) == DEFAULT_GAP_SECONDS


# -- parse_organizer_trake_query ----------------------------------------------
#
# The organizer hands out TRAKE as a `.txt` whose events are already numbered
# E1..Ek; `run_trake_pipeline.py` reads JSON. Nothing in the repo bridged the
# two, so every TRAKE query had to be retyped by hand mid-competition.


REAL_P1_16 = (
    "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, "
    "bên cạnh lá cờ trắng viền đỏ.\n"
    "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\n"
    "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.\n"
    "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.\n"
)


def test_parses_each_numbered_event_without_its_marker() -> None:
    query = parse_organizer_trake_query(REAL_P1_16)

    assert query.events == (
        "Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.",
        "Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.",
        "Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.",
    )


def test_the_scene_setting_before_e1_is_not_an_event() -> None:
    """It describes the video, not a moment to locate. Scoring an event
    that has no timestamp would put a wrong frame in the chain."""
    query = parse_organizer_trake_query(REAL_P1_16)

    assert query.preamble.startswith("Đoạn video bắt đầu bằng")
    assert all("Đoạn video bắt đầu" not in event for event in query.events)


def test_produces_exactly_one_fewer_gap_than_events() -> None:
    """TrakeQuery rejects any other count, and run_trake_pipeline rejects it
    again when reading the file."""
    query = parse_organizer_trake_query(REAL_P1_16)

    assert len(query.gap_phrases) == len(query.events) - 1


def test_a_wrapped_event_continues_the_event_it_follows() -> None:
    """A line with no marker is far more likely to be the previous event
    running on than a gap clue, and losing half an event's description
    costs retrieval much more than a default gap window does."""
    query = parse_organizer_trake_query(
        "E1 Người đàn ông mở cửa\nvà bước vào phòng.\nE2 Đèn bật sáng.\n"
    )

    assert query.events[0] == "Người đàn ông mở cửa và bước vào phòng."


def test_a_leading_time_phrase_becomes_the_gap_and_leaves_the_event_text() -> None:
    """`Ngay sau đó` between two events is a timing constraint (0-10s), not
    part of what the frame looks like."""
    query = parse_organizer_trake_query(
        "E1 Con lân xoay trên trụ.\nE2 Ngay sau đó, con lân ngoạm quả bí.\n"
    )

    assert query.gap_phrases == ("ngay sau đó",)
    assert query.events[1] == "con lân ngoạm quả bí."


def test_a_time_word_inside_a_sentence_is_not_read_as_a_gap() -> None:
    """`rồi` is an ordinary Vietnamese word. Matching it mid-sentence would
    silently narrow the gap window to 60s and could drop the true chain."""
    query = parse_organizer_trake_query(
        "E1 Người đàn ông mở cửa.\nE2 Con lân nhảy lên rồi tiếp đất trên trụ.\n"
    )

    assert query.gap_phrases == ("",)
    assert query.events[1] == "Con lân nhảy lên rồi tiếp đất trên trụ."


def test_accepts_the_punctuation_variants_a_marker_may_carry() -> None:
    query = parse_organizer_trake_query("E1. Cửa mở.\nE2: Đèn sáng.\nE 3) Người bước ra.\n")

    assert query.events == ("Cửa mở.", "Đèn sáng.", "Người bước ra.")


def test_refuses_a_query_with_a_single_event() -> None:
    """One event is plain KIS; TrakeQuery would reject it later, after the
    operator had already spent the run."""
    with pytest.raises(ValueError, match="at least 2"):
        parse_organizer_trake_query("E1 Chỉ có một sự kiện.\n")


def test_refuses_events_numbered_out_of_sequence() -> None:
    """E1, E3 means E2 was lost -- to a typo or to a parse this code got
    wrong. Chaining the two survivors would answer a different query."""
    with pytest.raises(ValueError, match="sequential"):
        parse_organizer_trake_query("E1 Cửa mở.\nE3 Đèn sáng.\n")


def test_refuses_a_text_with_no_markers_at_all() -> None:
    with pytest.raises(ValueError, match="no numbered events"):
        parse_organizer_trake_query("Một đoạn mô tả không có sự kiện nào được đánh số.")
