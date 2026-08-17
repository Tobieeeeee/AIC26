import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class QueryConcepts:
    subjects: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    scene: List[str] = field(default_factory=list)
    spatial_relations: List[str] = field(default_factory=list)
    temporal_cues: List[str] = field(default_factory=list)
    text_cues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "subjects": self.subjects,
            "objects": self.objects,
            "attributes": self.attributes,
            "actions": self.actions,
            "scene": self.scene,
            "spatial_relations": self.spatial_relations,
            "temporal_cues": self.temporal_cues,
            "text_cues": self.text_cues,
        }


@dataclass
class DecompositionResult:
    query_id: str
    query_vi: str
    query_en_full: str
    qwen_model: str
    concepts: QueryConcepts
    keyword_query: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    raw_model_output: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_vi": self.query_vi,
            "query_en_full": self.query_en_full,
            "qwen_model": self.qwen_model,
            "concepts": self.concepts.to_dict(),
            "keyword_query": self.keyword_query,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "raw_model_output": self.raw_model_output,
            "timestamp": self.timestamp,
        }
