import json
import re
from typing import Dict, Any, Tuple
from .models import QueryConcepts


def clean_json_output(text: str) -> str:
    """Strip markdown code fences or trailing text to extract JSON."""
    text = text.strip()
    # If the model wrapped the output in ```json ... ```
    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find the outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
        
    return text


def parse_and_validate_concepts(raw_json: str) -> QueryConcepts:
    """Parse JSON and normalize to the expected schema."""
    cleaned = clean_json_output(raw_json)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError("Root element must be a JSON object.")

    # Normalization: ensure everything is a list of strings
    def normalize_list(val: Any) -> list:
        if val is None:
            return []
        if isinstance(val, list):
            return [str(v).strip() for v in val if v is not None and str(v).strip()]
        if isinstance(val, str):
            # Fallback if model returned a comma separated string instead of list
            return [s.strip() for s in val.split(',') if s.strip()]
        return [str(val).strip()]

    expected_keys = {
        "subjects", "objects", "attributes", "actions",
        "scene", "spatial_relations", "temporal_cues", "text_cues"
    }
    if not any(key in data for key in expected_keys):
        raise ValueError("JSON does not contain any of the required schema keys.")

    return QueryConcepts(
        subjects=normalize_list(data.get("subjects")),
        objects=normalize_list(data.get("objects")),
        attributes=normalize_list(data.get("attributes")),
        actions=normalize_list(data.get("actions")),
        scene=normalize_list(data.get("scene")),
        spatial_relations=normalize_list(data.get("spatial_relations")),
        temporal_cues=normalize_list(data.get("temporal_cues")),
        text_cues=normalize_list(data.get("text_cues")),
    )


def build_keyword_query(concepts: QueryConcepts) -> str:
    """Build a deterministic deduplicated keyword query from concepts."""
    # Order required by task
    order = [
        concepts.subjects,
        concepts.objects,
        concepts.attributes,
        concepts.actions,
        concepts.scene,
        concepts.spatial_relations,
        concepts.temporal_cues,
        concepts.text_cues,
    ]
    
    seen = set()
    keywords = []
    
    for concept_list in order:
        for concept in concept_list:
            lower_concept = concept.lower()
            if lower_concept not in seen:
                seen.add(lower_concept)
                keywords.append(concept)
                
    return ", ".join(keywords)
