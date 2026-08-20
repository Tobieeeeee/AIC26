import pytest
from mvr.enrichment.ocr.scene_text import (
    OcrCandidate,
    bbox_iou,
    group_candidates,
    normalize_text,
    text_similarity,
)


def test_normalize_text():
    assert normalize_text("  Starbucks  ") == "starbucks"
    assert normalize_text("STARBUCK5") == "starbuck5"
    assert normalize_text("a  b\tc") == "a b c"


def test_text_similarity():
    assert text_similarity("SALE", "SALE") == 1.0
    assert text_similarity("", "") == 1.0
    assert text_similarity("SALE", "") == 0.0
    assert text_similarity("", "SALE") == 0.0
    # STARBUCKS vs STARBUCK5: len 9, dist 1 -> sim 8/9 = 0.888...
    assert text_similarity("STARBUCKS", "STARBUCK5") == pytest.approx(8 / 9)


def test_bbox_iou():
    b1 = [0, 0, 10, 10]
    b2 = [5, 5, 15, 15]
    # inter = 5x5 = 25
    # union = 100 + 100 - 25 = 175
    # iou = 25/175 = 1/7
    assert bbox_iou(b1, b2) == pytest.approx(1 / 7)

    b3 = [20, 20, 30, 30]
    assert bbox_iou(b1, b3) == 0.0


def test_temporal_consolidation_A():
    # A — Temporal consolidation
    cands = [
        OcrCandidate("V1", 1, "K1", 10, 9, 0.9, "STARBUCK5", 0.72, [100, 50, 430, 130]),
        OcrCandidate("V1", 1, "K1", 10, 10, 1.0, "STARBUCKS", 0.96, [100, 50, 430, 130]),
        OcrCandidate("V1", 1, "K1", 10, 11, 1.1, "STARBUCKS", 0.89, [100, 50, 430, 130]),
    ]
    groups = group_candidates(cands)
    assert len(groups) == 1
    res = groups[0]
    assert res.text == "STARBUCKS"
    assert res.conf == 0.96
    assert res.support_count == 3
    assert res.source_frame_idx == 10


def test_spatial_consistency_B():
    # B — Same text, separate regions
    cands = [
        OcrCandidate("V1", 1, "K1", 10, 10, 1.0, "SALE", 0.9, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 10, 1.0, "SALE", 0.9, [100, 100, 110, 110]),
    ]
    groups = group_candidates(cands)
    assert len(groups) == 2


def test_unrelated_text_C():
    # C — Same region, unrelated text
    cands = [
        OcrCandidate("V1", 1, "K1", 10, 9, 0.9, "OPEN", 0.9, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 10, 1.0, "CAFE", 0.9, [0, 0, 10, 10]),
    ]
    groups = group_candidates(cands)
    assert len(groups) == 2


def test_representative_confidence_D():
    # D — Representative confidence
    cands = [
        OcrCandidate("V1", 1, "K1", 10, 9, 0.9, "CAFE", 0.62, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 10, 1.0, "CAFE", 0.94, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 11, 1.1, "CAFE", 0.81, [0, 0, 10, 10]),
    ]
    groups = group_candidates(cands)
    assert len(groups) == 1
    assert groups[0].conf == 0.94
    assert groups[0].source_frame_idx == 10


def test_tie_breaker_E():
    # E — Tie breaker: equal confidence -> closest to anchor wins -> lower frame_idx wins -> lexical text
    cands = [
        OcrCandidate("V1", 1, "K1", 10, 8, 0.8, "CAFE", 0.9, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 9, 0.9, "CAFE", 0.9, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 11, 1.1, "CAFE", 0.9, [0, 0, 10, 10]),
    ]
    groups = group_candidates(cands)
    assert len(groups) == 1
    # distance to 10: 8 (dist 2), 9 (dist 1), 11 (dist 1)
    # 9 and 11 tie for distance (1). 9 is lower.
    assert groups[0].source_frame_idx == 9


def test_identity_preservation_I():
    # I — Identity preservation
    cands = [
        OcrCandidate("V99", 5, "K99", 10, 10, 1.5, "TEXT", 0.99, [1, 2, 3, 4]),
    ]
    groups = group_candidates(cands)
    assert len(groups) == 1
    res = groups[0]
    assert res.video_id == "V99"
    assert res.shot_id == 5
    assert res.keyframe_id == "K99"
    assert res.anchor_frame_idx == 10
    assert res.source_frame_idx == 10
    assert res.source_timestamp == 1.5
    assert res.text == "TEXT"


def test_determinism_J():
    # J — Determinism (shuffle candidates repeatedly)
    import random
    cands = [
        OcrCandidate("V1", 1, "K1", 10, 9, 0.9, "A", 0.9, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 10, 1.0, "B", 0.9, [0, 0, 10, 10]),
        OcrCandidate("V1", 1, "K1", 10, 11, 1.1, "C", 0.9, [0, 0, 10, 10]),
    ]
    canonical_output = None
    for i in range(10):
        shuffled = cands[:]
        random.seed(i)
        random.shuffle(shuffled)
        groups = group_candidates(shuffled, similarity_threshold=0.0) # all group together
        res = groups[0]
        output = (res.text, res.conf, res.source_frame_idx, res.support_count)
        if canonical_output is None:
            canonical_output = output
        else:
            assert output == canonical_output
