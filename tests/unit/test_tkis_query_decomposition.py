import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.verification.query_decomposition.models import QueryConcepts, DecompositionResult
from scripts.verification.query_decomposition.utils import (
    clean_json_output,
    parse_and_validate_concepts,
    build_keyword_query
)
from scripts.verification.query_decomposition.qwen import QwenDecomposer
from scripts.verification.run_tkis_query_decomposition_benchmark import load_input_queries, get_cache_key

# -----------------------------------------
# Test utils.py (JSON parsing, validation)
# -----------------------------------------

def test_clean_json_output():
    assert clean_json_output('{"a": 1}') == '{"a": 1}'
    assert clean_json_output('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_json_output('Some text\n```{"a": 1}```\nMore text') == '{"a": 1}'
    assert clean_json_output('Here is the json:\n{"a": 1}\nHope it helps!') == '{"a": 1}'

def test_parse_and_validate_concepts_valid():
    raw_json = '''{
        "subjects": ["man"],
        "objects": ["car"],
        "attributes": ["red"],
        "actions": ["driving"],
        "scene": [],
        "spatial_relations": ["in front of"],
        "temporal_cues": [],
        "text_cues": ["STOP"]
    }'''
    concepts = parse_and_validate_concepts(raw_json)
    assert concepts.subjects == ["man"]
    assert concepts.objects == ["car"]
    assert concepts.attributes == ["red"]
    assert concepts.actions == ["driving"]
    assert concepts.scene == []
    assert concepts.spatial_relations == ["in front of"]
    assert concepts.temporal_cues == []
    assert concepts.text_cues == ["STOP"]

def test_parse_and_validate_concepts_empty():
    raw_json = '{}'
    with pytest.raises(ValueError, match="JSON does not contain any of the required schema keys"):
        parse_and_validate_concepts(raw_json)

def test_parse_and_validate_concepts_coercion():
    # If the model returns strings instead of lists, or extra spaces
    raw_json = '''{
        "subjects": "man, woman",
        "objects": [" car ", null, "bike"]
    }'''
    concepts = parse_and_validate_concepts(raw_json)
    assert concepts.subjects == ["man", "woman"]
    assert concepts.objects == ["car", "bike"]

def test_parse_and_validate_concepts_invalid_json():
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_and_validate_concepts('{ bad json }')

def test_parse_and_validate_concepts_not_dict():
    with pytest.raises(ValueError, match="Root element must be a JSON object"):
        parse_and_validate_concepts('["a list instead of dict"]')

def test_build_keyword_query():
    concepts = QueryConcepts(
        subjects=["man", "Woman"],
        objects=["car"],
        attributes=["red", "Red", "fast"], # Duplicate case-insensitive
        actions=["driving"],
        spatial_relations=["in front of"]
    )
    # Expected order: subjects, objects, attributes, actions, scene, spatial, temporal, text
    keyword_query = build_keyword_query(concepts)
    assert keyword_query == "man, Woman, car, red, fast, driving, in front of"

# -----------------------------------------
# Test run_tkis_query_decomposition_benchmark.py
# -----------------------------------------

def test_load_input_queries_json(tmp_path):
    # Simulate Task 01 output
    input_data = [
        {"query_id": "q1", "method": "vi_raw", "success": True, "translated_text": "ignore"},
        {"query_id": "q1", "method": "local_vi2en", "success": True, "translated_text": "Good query"},
        {"query_id": "q2", "method": "local_vi2en", "success": False, "translated_text": ""},
        {"query_id": "q3", "method": "local_vi2en", "success": True, "translated_text": "Another good query"}
    ]
    file_path = tmp_path / "input.json"
    file_path.write_text(json.dumps(input_data))
    
    queries = load_input_queries(str(file_path), source_method="local_vi2en")
    assert len(queries) == 2
    assert queries[0]["query_en"] == "Good query"
    assert queries[1]["query_en"] == "Another good query"

def test_load_input_queries_csv(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text("query_id,query_en\nq1,First\nq2,\nq3,Third\n")
    queries = load_input_queries(str(file_path), source_method="local_vi2en") # source_method ignored for CSV
    assert len(queries) == 2
    assert queries[0]["query_id"] == "q1"
    assert queries[1]["query_en"] == "Third"

def test_get_cache_key():
    key1 = get_cache_key("query A", "model1")
    key2 = get_cache_key("query A", "model1")
    key3 = get_cache_key("query B", "model1")
    key4 = get_cache_key("query A", "model2")
    assert key1 == key2
    assert key1 != key3
    assert key1 != key4

# -----------------------------------------
# Test qwen.py (Model Decomposer)
# -----------------------------------------

class MockTokenizer:
    def __init__(self):
        self.chat_template = None
    
    def __call__(self, text, return_tensors):
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.input_ids = [[0, 1, 2]]
        return mock_inputs
        
    def batch_decode(self, ids, skip_special_tokens):
        # We simulate the tokenizer returning a valid JSON string
        return ['{"subjects": ["test subject"]}']

class MockModel:
    def __init__(self):
        self.eval = MagicMock()
    
    def to(self, device):
        return self
        
    def generate(self, **kwargs):
        # Simulate generating some tokens
        return [[0, 1, 2, 3, 4]]

@patch.dict('sys.modules', {'torch': MagicMock()})
@patch('scripts.verification.query_decomposition.qwen.QwenDecomposer._load')
def test_qwen_decomposer_success(mock_load):
    # We patch _load so we don't try to import transformers/torch globally
    decomposer = QwenDecomposer("dummy_model", "cpu")
    decomposer._tokenizer = MockTokenizer()
    decomposer._model = MockModel()
    
    # We also need to mock _resolve_device to return cpu
    with patch.object(decomposer, '_resolve_device', return_value='cpu'):
        concepts, keywords, raw = decomposer.decompose("Test query")
        assert concepts.subjects == ["test subject"]
        assert keywords == "test subject"
        assert raw == '{"subjects": ["test subject"]}'

@patch.dict('sys.modules', {'torch': MagicMock()})
@patch('scripts.verification.query_decomposition.qwen.QwenDecomposer._load')
def test_qwen_decomposer_retry_on_bad_json(mock_load):
    decomposer = QwenDecomposer("dummy_model", "cpu")
    decomposer._tokenizer = MockTokenizer()
    decomposer._model = MockModel()
    
    # Simulate a bad JSON on the first generate, and good JSON on the second
    responses = [
        'bad non json output',
        '{"subjects": ["retry success"]}'
    ]
    def mock_generate_side_effect(*args, **kwargs):
        return responses.pop(0)
        
    with patch.object(decomposer, '_generate', side_effect=mock_generate_side_effect):
        concepts, keywords, raw = decomposer.decompose("Test query")
        assert concepts.subjects == ["retry success"]
        assert len(responses) == 0

@patch.dict('sys.modules', {'torch': MagicMock()})
@patch('scripts.verification.query_decomposition.qwen.QwenDecomposer._load')
def test_qwen_decomposer_failure_after_retry(mock_load):
    decomposer = QwenDecomposer("dummy_model", "cpu")
    
    # Simulate bad JSON constantly
    with patch.object(decomposer, '_generate', return_value='bad json'):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            decomposer.decompose("Test query")

# -----------------------------------------
# Test Result Models
# -----------------------------------------

def test_decomposition_result_to_dict():
    concepts = QueryConcepts(subjects=["man"])
    res = DecompositionResult(
        query_id="q1",
        query_vi="vi text",
        query_en_full="en text",
        qwen_model="model",
        concepts=concepts,
        keyword_query="man",
        latency_ms=100.5,
        success=True,
    )
    d = res.to_dict()
    assert d["query_id"] == "q1"
    assert d["concepts"]["subjects"] == ["man"]
    assert d["keyword_query"] == "man"
    assert d["success"] is True
    assert "timestamp" in d
