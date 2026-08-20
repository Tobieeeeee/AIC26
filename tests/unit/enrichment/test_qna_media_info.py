import json

import pytest


def _write(root, video_id, **fields):
    payload = {
        "title": "",
        "description": "",
        "keywords": [],
        "author": "",
        "publish_date": "",
        "watch_url": "",
        **fields,
    }
    (root / f"{video_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def media_root(tmp_path):
    root = tmp_path / "media_info"
    root.mkdir()
    _write(
        root, "L24_V011",
        title="Đình thần Nguyễn Trung Trực - Kiên Giang",
        description="Lễ hội truyền thống tại Rạch Giá",
        keywords=["di tich", "kien giang"],
    )
    _write(
        root, "L30_V072",
        title="CLB FANA trao quà từ thiện",
        description="Chương trình thiện nguyện tại Khánh Hòa",
        keywords=["tu thien"],
    )
    _write(root, "L21_V001", title="Bản tin 60 giây", description="Tin tức tổng hợp")
    return root


# -- loading ---------------------------------------------------------------


def test_index_loads_every_media_info_file(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert len(index) == 3
    assert index.get("L24_V011").title.startswith("Đình thần")


def test_index_returns_none_for_unknown_video(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert index.get("L99_V999") is None


def test_index_tolerates_a_malformed_file(media_root) -> None:
    """Some organizer videos ship no/!broken metadata; that must not break
    the whole corpus load."""
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    (media_root / "L21_V002.json").write_text("{ not json", encoding="utf-8")

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert len(index) == 3  # the broken one is skipped, the rest still load


# -- search ----------------------------------------------------------------


def test_search_matches_title_ignoring_vietnamese_diacritics(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    hits = index.search(["nguyen trung truc"])

    assert [h.video_id for h in hits] == ["L24_V011"]


def test_search_matches_when_query_has_diacritics_but_data_does_not(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    hits = index.search(["Kiên Giang"])

    assert [h.video_id for h in hits] == ["L24_V011"]


def test_search_is_case_insensitive(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert [h.video_id for h in index.search(["fana"])] == ["L30_V072"]
    assert [h.video_id for h in index.search(["FANA"])] == ["L30_V072"]


def test_search_ranks_more_matched_terms_first(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    hits = index.search(["tu thien", "khanh hoa", "kien giang"])

    assert hits[0].video_id == "L30_V072"  # matches two terms
    assert hits[0].matched_terms == 2


def test_search_returns_empty_when_nothing_matches(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert index.search(["khong ton tai o dau ca"]) == []


def test_search_ignores_blank_terms(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert index.search(["   ", ""]) == []


def test_search_respects_limit(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    assert len(index.search(["tin"], limit=1)) <= 1


# -- prompt formatting -----------------------------------------------------


def test_media_info_formats_a_compact_prompt_block(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(media_root)

    block = index.get("L30_V072").as_prompt_text()

    assert "CLB FANA" in block
    assert "Khánh Hòa" in block


def test_prompt_text_truncates_a_long_description(media_root) -> None:
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    _write(media_root, "L21_V003", title="T", description="x" * 5000)
    index = MediaInfoSearchIndex.from_directory(media_root)

    block = index.get("L21_V003").as_prompt_text(max_description_chars=100)

    assert len(block) < 400
