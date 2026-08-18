import pytest
from mvr.retrieval.kis import KeyframeFrameIndexResolver, resolve_original_frame_id

class MockAicRegistry:
    def resolve_frame(self, video_id: str, keyframe_ref: str):
        # AIC Registry explicitly throws an error if not numeric
        try:
            int(keyframe_ref)
        except ValueError:
            from mvr.ingestion.aic_dataset import MissingFrameMappingError
            raise ValueError(f"keyframe reference must be numeric, got '{keyframe_ref}'")
        return type('Obj', (object,), {'frame_idx': 100})()

def test_pipeline_resolver_wins_before_aic_registry_fallback():
    # Setup mock resolver
    mapping = {"video7688_S0002_K01": 95}
    resolver = KeyframeFrameIndexResolver(mapping)
    
    registry = MockAicRegistry()

    # When the string keyframe_id is passed, the pipeline resolver should successfully return 95
    # and NOT fall back to the AIC registry (which would crash due to non-numeric).
    frame_idx = resolve_original_frame_id(
        video_id="video7688",
        keyframe_id="video7688_S0002_K01",
        pipeline_resolver=resolver,
        aic_registry=registry,
    )

    assert frame_idx == 95, f"Expected 95, got {frame_idx}"

def test_pipeline_resolver_missing_mapping():
    resolver = KeyframeFrameIndexResolver({"other_video": 10})
    registry = MockAicRegistry()
    
    # If the pipeline resolver does not contain the mapping, it falls back.
    # Since the key is non-numeric, the AIC registry should raise the ValueError.
    with pytest.raises(ValueError, match="keyframe reference must be numeric"):
        resolve_original_frame_id(
            video_id="video7688",
            keyframe_id="video7688_S0002_K01",
            pipeline_resolver=resolver,
            aic_registry=registry,
        )
