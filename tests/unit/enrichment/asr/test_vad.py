from mvr.enrichment.asr.contracts import SpeechRegion
from mvr.enrichment.asr.vad import merge_speech_regions


def test_merge_speech_regions_respects_threshold() -> None:
    regions = [
        SpeechRegion(1.0, 2.0),
        SpeechRegion(2.25, 3.0),
        SpeechRegion(3.6, 4.0),
    ]

    assert merge_speech_regions(regions, merge_gap_ms=250) == [
        SpeechRegion(1.0, 3.0),
        SpeechRegion(3.6, 4.0),
    ]
