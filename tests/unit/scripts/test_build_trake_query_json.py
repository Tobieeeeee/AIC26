"""The bridge between the organizer's TRAKE `.txt` and the pipeline's JSON.

`run_trake_pipeline.py` reads `{"events": [...], "gap_phrases": [...]}`.
The organizer hands out prose with the events numbered E1..Ek. Nothing
connected the two, so every TRAKE query had to be retyped by hand in the
middle of a run -- the one track this system could not answer from the
files it was given.
"""
import json

import pytest

from scripts.build_trake_query_json import convert_file, convert_directory
from scripts.run_trake_pipeline import load_query_file


REAL_P1_16 = (
    "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, "
    "bên cạnh lá cờ trắng viền đỏ.\n"
    "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\n"
    "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.\n"
    "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.\n"
)


def test_the_written_file_is_one_the_pipeline_accepts(tmp_path) -> None:
    """The whole point. `load_query_file` is the real consumer, and it
    raises SystemExit on anything it cannot use."""
    source = tmp_path / "query-p1-16-trake.txt"
    source.write_text(REAL_P1_16, encoding="utf-8")

    written = convert_file(source, tmp_path)
    events, gap_phrases = load_query_file(written)

    assert len(events) == 3
    assert len(gap_phrases) == 2


def test_the_json_is_named_after_the_query_it_came_from(tmp_path) -> None:
    """Submissions are named from the query file's stem, so a renamed
    intermediate would write the wrong CSV name at the end of the run."""
    source = tmp_path / "query-p1-16-trake.txt"
    source.write_text(REAL_P1_16, encoding="utf-8")

    written = convert_file(source, tmp_path)

    assert written.name == "query-p1-16-trake.json"


def test_the_scene_setting_sentence_is_kept_but_not_as_an_event(tmp_path) -> None:
    source = tmp_path / "q-trake.txt"
    source.write_text(REAL_P1_16, encoding="utf-8")

    payload = json.loads(convert_file(source, tmp_path).read_text(encoding="utf-8"))

    assert payload["preamble"].startswith("Đoạn video bắt đầu bằng")
    assert len(payload["events"]) == 3


def test_a_directory_converts_only_the_trake_queries(tmp_path) -> None:
    """A query pack holds all three tracks. Converting a KIS file would
    write a JSON nothing reads and report a success that did not happen."""
    (tmp_path / "query-1-kis.txt").write_text("một cảnh nào đó", encoding="utf-8")
    (tmp_path / "query-2-qa.txt").write_text("một câu hỏi nào đó", encoding="utf-8")
    (tmp_path / "query-3-trake.txt").write_text(REAL_P1_16, encoding="utf-8")
    out = tmp_path / "out"

    written = convert_directory(tmp_path, out)

    assert [path.name for path in written] == ["query-3-trake.json"]


def test_a_malformed_query_names_itself_in_the_error(tmp_path) -> None:
    """Mid-competition the operator sees only the error text; a ValueError
    with no filename in it does not say which query to go fix."""
    source = tmp_path / "query-9-trake.txt"
    source.write_text("Không có sự kiện nào được đánh số.", encoding="utf-8")

    with pytest.raises(ValueError, match="query-9-trake.txt"):
        convert_file(source, tmp_path)


def test_one_broken_query_does_not_stop_the_rest_of_the_pack(tmp_path) -> None:
    """The other queries are still worth points."""
    (tmp_path / "query-1-trake.txt").write_text("hỏng", encoding="utf-8")
    (tmp_path / "query-2-trake.txt").write_text(REAL_P1_16, encoding="utf-8")

    written = convert_directory(tmp_path, tmp_path / "out")

    assert [path.name for path in written] == ["query-2-trake.json"]
