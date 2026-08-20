"""Per-keyframe voice activity detection and Vietnamese ASR features."""

from .contracts import ASRArtifact, ASRSegment, KeyframeAudioSource, SpeechRegion
from .pipeline import ASRPipelineConfig, build_asr_features

__all__ = [
    "ASRArtifact",
    "ASRPipelineConfig",
    "ASRSegment",
    "KeyframeAudioSource",
    "SpeechRegion",
    "build_asr_features",
]
