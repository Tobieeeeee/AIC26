import json

import pytest


VALID_PAYLOAD = {
    "detection_class_entities": ["Person", "Car", "Person"],
    "detection_scores": [0.91, 0.42, 0.35],
    "detection_boxes": [
        [0.1, 0.2, 0.5, 0.6],
        [0.0, 0.0, 1.0, 1.0],
        [0.3, 0.3, 0.4, 0.4],
    ],
}


def _write(tmp_path, payload) -> str:
    path = tmp_path / "0000.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_frame_objects_parses_detections_and_counts(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import load_frame_objects

    path = _write(tmp_path, VALID_PAYLOAD)

    frame_objects = load_frame_objects(path, video_id="L01_V001", frame_ref="0000")

    assert frame_objects.video_id == "L01_V001"
    assert frame_objects.frame_ref == "0000"
    assert len(frame_objects.detections) == 3
    assert frame_objects.detections[0].label == "Person"
    assert frame_objects.detections[0].confidence == pytest.approx(0.91)
    assert frame_objects.detections[0].bbox == (0.1, 0.2, 0.5, 0.6)
    assert frame_objects.counts == {"Person": 2, "Car": 1}


def test_load_frame_objects_rejects_mismatched_array_lengths(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError, load_frame_objects

    payload = dict(VALID_PAYLOAD)
    payload["detection_scores"] = [0.91, 0.42]
    path = _write(tmp_path, payload)

    with pytest.raises(InvalidAicObjectsError):
        load_frame_objects(path, video_id="L01_V001", frame_ref="0000")


@pytest.mark.parametrize(
    "bad_scores",
    [
        [0.91, 1.5, 0.35],
        [0.91, -0.1, 0.35],
    ],
)
def test_load_frame_objects_rejects_out_of_range_confidence(tmp_path, bad_scores) -> None:
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError, load_frame_objects

    payload = dict(VALID_PAYLOAD)
    payload["detection_scores"] = bad_scores
    path = _write(tmp_path, payload)

    with pytest.raises(InvalidAicObjectsError):
        load_frame_objects(path, video_id="L01_V001", frame_ref="0000")


def test_load_frame_objects_rejects_malformed_bbox(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError, load_frame_objects

    payload = dict(VALID_PAYLOAD)
    payload["detection_boxes"] = [[0.1, 0.2, 0.5], [0.0, 0.0, 1.0, 1.0], [0.3, 0.3, 0.4, 0.4]]
    path = _write(tmp_path, payload)

    with pytest.raises(InvalidAicObjectsError):
        load_frame_objects(path, video_id="L01_V001", frame_ref="0000")


def test_load_frame_objects_rejects_blank_label(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError, load_frame_objects

    payload = dict(VALID_PAYLOAD)
    payload["detection_class_entities"] = ["Person", "  ", "Person"]
    path = _write(tmp_path, payload)

    with pytest.raises(InvalidAicObjectsError):
        load_frame_objects(path, video_id="L01_V001", frame_ref="0000")


def test_load_frame_objects_rejects_missing_required_key(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError, load_frame_objects

    payload = {"detection_scores": [0.9], "detection_boxes": [[0.0, 0.0, 1.0, 1.0]]}
    path = _write(tmp_path, payload)

    with pytest.raises(InvalidAicObjectsError):
        load_frame_objects(path, video_id="L01_V001", frame_ref="0000")


def test_load_frame_objects_keeps_every_detection_by_default(tmp_path) -> None:
    """The loader is a faithful artifact reader: filtering is a consumer
    policy decision, not something the parser applies on its own."""
    from mvr.enrichment.objects.aic_objects import load_frame_objects

    path = _write(tmp_path, VALID_PAYLOAD)

    frame_objects = load_frame_objects(path, video_id="L01_V001", frame_ref="0000")

    assert len(frame_objects.detections) == 3


def test_load_frame_objects_applies_min_confidence(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import load_frame_objects

    path = _write(tmp_path, VALID_PAYLOAD)  # scores: 0.91, 0.42, 0.35

    frame_objects = load_frame_objects(
        path, video_id="L01_V001", frame_ref="0000", min_confidence=0.4
    )

    assert [d.confidence for d in frame_objects.detections] == pytest.approx([0.91, 0.42])
    assert frame_objects.counts == {"Person": 1, "Car": 1}


def test_load_frame_objects_min_confidence_is_inclusive(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import load_frame_objects

    payload = dict(VALID_PAYLOAD)
    payload["detection_scores"] = [0.4, 0.39999, 0.5]
    path = _write(tmp_path, payload)

    frame_objects = load_frame_objects(
        path, video_id="L01_V001", frame_ref="0000", min_confidence=0.4
    )

    assert [d.confidence for d in frame_objects.detections] == pytest.approx([0.4, 0.5])


def test_load_frame_objects_min_confidence_can_filter_everything(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import load_frame_objects

    path = _write(tmp_path, VALID_PAYLOAD)

    frame_objects = load_frame_objects(
        path, video_id="L01_V001", frame_ref="0000", min_confidence=0.99
    )

    assert frame_objects.detections == ()
    assert frame_objects.counts == {}


def test_load_frame_objects_still_validates_filtered_out_detections(tmp_path) -> None:
    """A malformed low-confidence detection must still raise -- filtering is
    not an excuse to silently accept a corrupt file."""
    from mvr.enrichment.objects.aic_objects import InvalidAicObjectsError, load_frame_objects

    payload = dict(VALID_PAYLOAD)
    payload["detection_scores"] = [0.91, 0.42, 1.5]  # last one invalid AND below no threshold
    path = _write(tmp_path, payload)

    with pytest.raises(InvalidAicObjectsError):
        load_frame_objects(
            path, video_id="L01_V001", frame_ref="0000", min_confidence=0.4
        )


def test_load_frame_objects_handles_empty_detections(tmp_path) -> None:
    from mvr.enrichment.objects.aic_objects import load_frame_objects

    payload = {
        "detection_class_entities": [],
        "detection_scores": [],
        "detection_boxes": [],
    }
    path = _write(tmp_path, payload)

    frame_objects = load_frame_objects(path, video_id="L01_V001", frame_ref="0000")

    assert frame_objects.detections == ()
    assert frame_objects.counts == {}
