import sys
from typing import Tuple, Optional
from .base import QueryDecomposer
from .models import QueryConcepts
from .utils import parse_and_validate_concepts, build_keyword_query

SYSTEM_PROMPT = """You are an expert AI visual search assistant. Your task is to extract key visual concepts from the given English query into a strict JSON format.
Only extract information explicitly supported by the query. Do NOT invent entities, colors, counts, actions, or relationships.
Do not hallucinate OCR text.
Output JSON ONLY. Do not explain. Do not add markdown fences.

Expected JSON Schema:
{
  "subjects": ["main people, animals, vehicles"],
  "objects": ["relevant visible objects"],
  "attributes": ["colors, clothing, counts, distinguishing properties"],
  "actions": ["visible actions"],
  "scene": ["scene/background/camera-view concepts"],
  "spatial_relations": ["important spatial relationships"],
  "temporal_cues": ["sequence/order explicitly present"],
  "text_cues": ["literal text/logos/signs to be visible"]
}

Example Input:
Three cyclists from the same team are riding in a straight line. They wear white uniforms. The first rider has a white helmet, the second a red helmet, and the last a black helmet.

Example JSON Output:
{
  "subjects": ["three cyclists"],
  "objects": ["bicycles", "helmets"],
  "attributes": ["same team", "white uniforms", "white helmet", "red helmet", "black helmet"],
  "actions": ["riding"],
  "scene": [],
  "spatial_relations": ["riding in a straight line"],
  "temporal_cues": ["first rider", "second rider", "last rider"],
  "text_cues": []
}"""


class QwenDecomposer(QueryDecomposer):
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self._device_str = device
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return self.model_name

    def _resolve_device(self) -> str:
        # Check if torch is available for robust error handling without crashing on import
        if "torch" not in sys.modules:
            try:
                import torch
            except ImportError:
                return "cpu"
        else:
            import torch

        if self._device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device_str

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Load model
        model_kwargs = {"trust_remote_code": True}
        if device == "cuda":
            # Typical optimization for inference on GPU
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
            
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        
        if device == "cpu" or "device_map" not in model_kwargs:
            self._model.to(device)
            
        self._model.eval()

    def _generate(self, prompt: str) -> str:
        import torch
        device = self._resolve_device()
        
        # Format conversation for instruction-tuned models if supported, else raw prompt
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            text = self._tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            text = f"{SYSTEM_PROMPT}\n\nInput:\n{prompt}\n\nJSON Output:\n"
            
        inputs = self._tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Deterministic decoding
            )
            
        # Extract only the newly generated tokens
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))
        ]
        
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def decompose(self, text: str) -> Tuple[QueryConcepts, str, Optional[str]]:
        self._load()
        
        raw_output = self._generate(text)
        
        try:
            concepts = parse_and_validate_concepts(raw_output)
        except ValueError as e:
            # One deterministic retry attempt asking it to fix the JSON
            retry_prompt = f"{text}\n\nYour previous output failed to parse as JSON with error: {e}. Please output ONLY valid JSON matching the schema."
            raw_output = self._generate(retry_prompt)
            concepts = parse_and_validate_concepts(raw_output) # Will bubble up error if it fails again
            
        keyword_query = build_keyword_query(concepts)
        return concepts, keyword_query, raw_output
