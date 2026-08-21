import json

import pytest

from scripts.run_trake_pipeline import load_query_file


def test_load_query_file_reads_events_and_gap_phrases(tmp_path) -> None:
    path = tmp_path / "query.json"
    path.write_text(
        json.dumps({"events": ["event one", "event two"], "gap_phrases": ["rồi"]}),
        encoding="utf-8",
    )

    events, gap_phrases = load_query_file(path)

    assert events == ["event one", "event two"]
    assert gap_phrases == ["rồi"]


def test_load_query_file_rejects_fewer_than_two_events(tmp_path) -> None:
    path = tmp_path / "query.json"
    path.write_text(json.dumps({"events": ["only one"], "gap_phrases": []}), encoding="utf-8")

    with pytest.raises(SystemExit):
        load_query_file(path)


def test_load_query_file_rejects_mismatched_gap_phrase_count(tmp_path) -> None:
    path = tmp_path / "query.json"
    path.write_text(
        json.dumps({"events": ["a", "b", "c"], "gap_phrases": ["rồi"]}),  # needs exactly 2
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        load_query_file(path)


def test_load_query_file_rejects_blank_event_text(tmp_path) -> None:
    path = tmp_path / "query.json"
    path.write_text(
        json.dumps({"events": ["real event", "   "], "gap_phrases": [""]}), encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        load_query_file(path)


def test_load_query_file_defaults_missing_gap_phrases_to_empty_list(tmp_path) -> None:
    """Matches TrakeSearchRequest's default_factory=list -- omitting
    gap_phrases entirely is only valid for a 1-gap (2-event) query."""
    path = tmp_path / "query.json"
    path.write_text(json.dumps({"events": ["event one", "event two"]}), encoding="utf-8")

    with pytest.raises(SystemExit):
        # 2 events need exactly 1 gap phrase; omitting the key defaults to
        # [] (0 phrases), which is a mismatch -- must raise, not guess "".
        load_query_file(path)
