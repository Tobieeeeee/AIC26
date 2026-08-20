import pytest

from mvr.retrieval.trake_events import (
    DEFAULT_GAP_SECONDS,
    GAP_PHRASE_TABLE,
    TrakeEventQuery,
    TrakeGapConstraint,
    TrakeQuery,
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
