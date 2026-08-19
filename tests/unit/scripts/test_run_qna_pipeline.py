from scripts.run_qna_pipeline import split_event_and_question


def test_split_finds_question_starting_with_hoi() -> None:
    text = (
        "Đoạn video về một chương trình từ thiện. "
        "Hỏi xã này có tên là gì? (tại thời điểm đó)"
    )
    event, question = split_event_and_question(text)

    assert event == "Đoạn video về một chương trình từ thiện."
    assert question.startswith("Hỏi xã này")


def test_split_falls_back_when_question_has_no_hoi_or_question_mark() -> None:
    """Real organizer edge case (query-p3-32-qa.txt): the final sentence is
    an imperative ("Đếm số lượng...") with neither a "Hỏi" marker nor a "?"
    -- the heuristic must not guess which sentence is the question, so it
    falls back to using the full text for both halves."""
    text = (
        "Cảnh quay một món ăn được bày trên vỉ nướng. "
        "Sau đó là cảnh làm xốt chấm. "
        "Đếm số lượng các miếng trên vỉ nướng."
    )
    event, question = split_event_and_question(text)

    assert event == text
    assert question == text


def test_split_handles_multi_sentence_event_description() -> None:
    text = (
        "Cảnh quay một món ăn được bày trên vỉ nướng. "
        "Sau đó là cảnh làm xốt chấm. "
        "Hỏi có bao nhiêu miếng trên vỉ nướng?"
    )
    event, question = split_event_and_question(text)

    assert "vỉ nướng" in event
    assert "xốt chấm" in event
    assert question == "Hỏi có bao nhiêu miếng trên vỉ nướng?"


def test_split_falls_back_to_full_text_when_no_question_marker() -> None:
    text = "Một đoạn mô tả không có câu hỏi rõ ràng."

    event, question = split_event_and_question(text)

    assert event == text
    assert question == text


def test_split_real_organizer_sample() -> None:
    text = (
        "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. "
        "Trong đoạn video có thể thấy một người đang cầm công thức món ăn "
        "với nguyên liệu chính là 200g thịt nạc xay. "
        "Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"
    )

    event, question = split_event_and_question(text)

    assert event.startswith("Đoạn video về một người phụ nữ")
    assert "200g thịt nạc xay" in event
    assert question == "Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"
