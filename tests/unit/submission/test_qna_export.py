import json

import pytest

from mvr.submission.qna_export import (
    InvalidQnaPredictionError,
    QnaPrediction,
    export_local_verification_csv,
    export_local_verification_json,
    load_local_verification_json,
    validate_qna_predictions,
)


class _Candidate:
    def __init__(self, video_id="L21_V001", frame_id=261, answer="Ninh Hòa"):
        self.video_id = video_id
        self.frame_id = frame_id
        self.answer = answer


def test_validate_assigns_contiguous_ranks() -> None:
    validated = validate_qna_predictions(
        [_Candidate(frame_id=10), _Candidate(frame_id=20), _Candidate(frame_id=30)]
    )

    assert [p.rank for p in validated] == [1, 2, 3]
    assert [p.frame_id for p in validated] == [10, 20, 30]


def test_validate_keeps_same_answer_across_different_frames() -> None:
    """Hedging one confident answer across several candidate frames is a
    legitimate use of the 100-answer budget, not a duplicate."""
    validated = validate_qna_predictions(
        [_Candidate(frame_id=10), _Candidate(frame_id=11), _Candidate(frame_id=12)]
    )

    assert len(validated) == 3


def test_validate_keeps_different_answers_on_the_same_frame() -> None:
    """Hedging several answers on one confident frame is likewise valid."""
    validated = validate_qna_predictions(
        [
            _Candidate(frame_id=10, answer="Ninh Hòa"),
            _Candidate(frame_id=10, answer="Vạn Ninh"),
        ]
    )

    assert len(validated) == 2


def test_validate_drops_exact_triple_duplicates_keeping_first() -> None:
    validated = validate_qna_predictions(
        [
            _Candidate(frame_id=10, answer="A"),
            _Candidate(frame_id=10, answer="A"),
            _Candidate(frame_id=11, answer="A"),
        ]
    )

    assert len(validated) == 2
    assert [(p.frame_id, p.answer) for p in validated] == [(10, "A"), (11, "A")]


def test_validate_rejects_blank_answer() -> None:
    with pytest.raises(InvalidQnaPredictionError):
        validate_qna_predictions([_Candidate(answer="   ")])


def test_validate_rejects_blank_video_id() -> None:
    with pytest.raises(InvalidQnaPredictionError):
        validate_qna_predictions([_Candidate(video_id="")])


def test_validate_rejects_negative_frame_id() -> None:
    with pytest.raises(InvalidQnaPredictionError):
        validate_qna_predictions([_Candidate(frame_id=-1)])


def test_validate_rejects_non_integer_frame_id() -> None:
    with pytest.raises(InvalidQnaPredictionError):
        validate_qna_predictions([_Candidate(frame_id=True)])


def test_validate_rejects_more_than_100_unique_predictions() -> None:
    candidates = [_Candidate(frame_id=i) for i in range(101)]

    with pytest.raises(InvalidQnaPredictionError):
        validate_qna_predictions(candidates)


def test_validate_allows_exactly_100() -> None:
    candidates = [_Candidate(frame_id=i) for i in range(100)]

    assert len(validate_qna_predictions(candidates)) == 100


def test_csv_export_round_trip(tmp_path) -> None:
    predictions = validate_qna_predictions([_Candidate(frame_id=261, answer="Ninh Hòa")])
    path = export_local_verification_csv(
        predictions, tmp_path / "out.csv", query_id="query-p1-15-qa"
    )

    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "format,query_id,rank,video_id,frame_id,answer"
    assert "Ninh Hòa" in rows[1]


def test_json_export_round_trip(tmp_path) -> None:
    predictions = validate_qna_predictions(
        [_Candidate(frame_id=261, answer="Ninh Hòa"), _Candidate(frame_id=262, answer="Vạn Ninh")]
    )
    path = export_local_verification_json(
        predictions, tmp_path / "out.json", query_id="query-p1-15-qa"
    )

    reloaded = load_local_verification_json(path)
    assert reloaded == predictions


def test_json_export_is_labelled_as_unofficial(tmp_path) -> None:
    predictions = validate_qna_predictions([_Candidate()])
    path = export_local_verification_json(predictions, tmp_path / "o.json", query_id="q")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "local_verification_export"
    assert "NOT the official" in payload["note"]


def test_load_rejects_foreign_format(tmp_path) -> None:
    path = tmp_path / "o.json"
    path.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")

    with pytest.raises(InvalidQnaPredictionError):
        load_local_verification_json(path)


def test_prediction_converts_to_evaluator_identity() -> None:
    from mvr.evaluation.aic_qna import QnaAnswer

    prediction = QnaPrediction(rank=1, video_id="L21_V001", frame_id=261, answer="Ninh Hòa")

    assert prediction.to_answer() == QnaAnswer(
        video_id="L21_V001", frame_id=261, answer="Ninh Hòa"
    )
