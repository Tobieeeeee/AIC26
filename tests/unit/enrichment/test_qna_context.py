import json


def _write_objects(tmp_path, payload) -> str:
    path = tmp_path / "objects.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class _FakeOcrEngine:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[object] = []

    def read_text(self, image: object) -> str:
        self.calls.append(image)
        return self._text


def test_build_frame_context_combines_objects_and_ocr(tmp_path) -> None:
    from mvr.enrichment.qna_context import build_frame_context

    objects_path = _write_objects(
        tmp_path,
        {
            "detection_class_entities": ["Person", "Person"],
            "detection_scores": [0.9, 0.8],
            "detection_boxes": [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]],
        },
    )
    engine = _FakeOcrEngine("EXIT 5B")
    image = object()

    context = build_frame_context(
        video_id="L01_V001",
        frame_id=1500,
        image=image,
        objects_path=objects_path,
        ocr_engine=engine,
    )

    assert context.video_id == "L01_V001"
    assert context.frame_id == 1500
    assert context.ocr_text == "EXIT 5B"
    assert context.object_counts == {"Person": 2}
    assert context.objects is not None
    assert len(context.objects.detections) == 2
    assert engine.calls == [image]


def test_build_frame_context_without_objects_file(tmp_path) -> None:
    from mvr.enrichment.qna_context import build_frame_context

    engine = _FakeOcrEngine("hello")

    context = build_frame_context(
        video_id="L01_V001",
        frame_id=1500,
        image=object(),
        objects_path=None,
        ocr_engine=engine,
    )

    assert context.objects is None
    assert context.object_counts == {}
    assert context.ocr_text == "hello"


def test_build_frame_context_missing_objects_file_on_disk(tmp_path) -> None:
    from mvr.enrichment.qna_context import build_frame_context

    missing_path = tmp_path / "does_not_exist.json"
    engine = _FakeOcrEngine("hello")

    context = build_frame_context(
        video_id="L01_V001",
        frame_id=1500,
        image=object(),
        objects_path=missing_path,
        ocr_engine=engine,
    )

    assert context.objects is None
    assert context.object_counts == {}


def test_build_frame_context_without_ocr_engine(tmp_path) -> None:
    from mvr.enrichment.qna_context import build_frame_context

    context = build_frame_context(
        video_id="L01_V001",
        frame_id=1500,
        image=object(),
        objects_path=None,
        ocr_engine=None,
    )

    assert context.ocr_text == ""


def test_build_frame_context_rejects_invalid_objects_file(tmp_path) -> None:
    from mvr.enrichment.qna_context import build_frame_context
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError

    objects_path = _write_objects(
        tmp_path,
        {
            "detection_class_entities": ["Person"],
            "detection_scores": [1.5],
            "detection_boxes": [[0.0, 0.0, 1.0, 1.0]],
        },
    )

    try:
        build_frame_context(
            video_id="L01_V001",
            frame_id=1500,
            image=object(),
            objects_path=objects_path,
            ocr_engine=None,
        )
        raise AssertionError("expected InvalidAicObjectsError")
    except InvalidAicObjectsError:
        pass
