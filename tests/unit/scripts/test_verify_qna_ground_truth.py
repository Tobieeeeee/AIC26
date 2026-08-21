from scripts.verify_qna_ground_truth import youtube_url_at


def test_youtube_url_appends_timestamp() -> None:
    url = youtube_url_at("https://youtube.com/watch?v=abc123", 87.6)

    assert url == "https://youtube.com/watch?v=abc123&t=87s"


def test_youtube_url_uses_question_mark_when_no_query_string() -> None:
    url = youtube_url_at("https://youtu.be/abc123", 12.9)

    assert url == "https://youtu.be/abc123?t=12s"


def test_youtube_url_truncates_rather_than_rounds_seconds() -> None:
    """Seeking slightly before the moment is safer than overshooting it."""
    assert youtube_url_at("https://youtu.be/x", 59.99).endswith("t=59s")


def test_youtube_url_is_blank_when_no_watch_url_known() -> None:
    assert youtube_url_at("", 10.0) == ""
