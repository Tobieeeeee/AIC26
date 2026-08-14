import numpy as np
import pytest


def test_document_builder_connects_phase02_metadata_to_lab6_schema() -> None:
    from mvr.indexing.documents import build_search_document

    document = build_search_document(
        {
            "keyframe_id": "L01_V001_S0001_K01",
            "video_id": "L01_V001",
            "timestamp": "0.48",
        },
        caption="",
        ocr_text="GIÁ VÀNG",
        objects={"person": 2},
        clip_vector=np.array([0.6, 0.8], dtype=np.float32),
    )

    assert document.to_source() == {
        "frame_id": "L01_V001_S0001_K01",
        "video_id": "L01_V001",
        "ts": 0.48,
        "caption": "",
        "ocr_text": "GIÁ VÀNG",
        "objects": {"person": 2},
        "clip_vector": [0.6000000238418579, 0.800000011920929],
    }


@pytest.mark.parametrize("keyframe", [{}, {"keyframe_id": "   "}])
def test_document_builder_rejects_missing_or_blank_frame_id(keyframe) -> None:
    from mvr.indexing.documents import build_search_document

    values = {"video_id": "video", "timestamp": 0.0, **keyframe}
    with pytest.raises(ValueError, match="frame_id must be non-empty"):
        build_search_document(
            values,
            caption="",
            ocr_text="",
            objects={},
            clip_vector=[1.0, 0.0],
        )
