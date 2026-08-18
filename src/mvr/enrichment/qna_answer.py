"""Q&A answer reasoning: turn a resolved frame's structured context plus a
question into a short VI/EN answer.

Two layers, deliberately separated so the reasoning logic is testable
without a cached model:

- ``LlmBackend`` (Protocol) -- raw text generation. The real implementation
  is ``HuggingFaceCausalLmBackend`` below, which fails closed unless model
  weights are already cached locally or download is explicitly opted into
  -- the same policy the ``HCMAIC-SkillPixel-KIS`` reference repo's
  ``CrossEncoderReranker`` uses for its own real-provider guard.
- ``AnswerReasoner`` -- prompt construction, JSON parsing/validation, one
  deterministic retry on malformed output (mirrors
  ``scripts/verification/query_decomposition/qwen.py``'s
  ``QwenDecomposer._generate``/retry pattern), and provenance tagging. Pure
  logic, backend-agnostic.

Per the organizer's own AIC2026 examples and the SOICT 2025 winner paper's
(OpenCubee_1 / ConvAgent) finding that competition Q&A questions can require
general knowledge rather than only visual context, the prompt allows the
model to answer from ``context-grounded``, ``knowledge-grounded``, or
``mixed`` evidence and requires it to say which. This is a model self-report,
not independently verified -- record it as-authored, never treat it as
ground truth about how the answer was actually produced.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from mvr.enrichment.qna_context import FrameContext

AnswerSource = Literal["context-grounded", "knowledge-grounded", "mixed"]
_VALID_SOURCES: tuple[AnswerSource, ...] = ("context-grounded", "knowledge-grounded", "mixed")

SYSTEM_PROMPT = """You are an expert visual question-answering assistant for a video \
retrieval competition. You are given OCR text and object-detection counts \
observed on one specific video frame, plus a question about that frame's \
event.

Answer using the given context when it is sufficient (e.g. counting \
detected objects, reading OCR text). If the question requires general \
knowledge not present in the context (e.g. identifying a country, person, \
or fact referenced by the scene), use your own knowledge instead. If you \
used both, say so.

The answer may be in Vietnamese or English, matching the question's \
language. Keep the answer short (a word, number, or short phrase) -- do \
not write a full sentence.

Output JSON ONLY, no markdown fences, no explanation, exactly this schema:
{"answer": "<short answer>", "source": "context-grounded" | "knowledge-grounded" | "mixed"}
"""


class AnswerReasonerUnavailable(RuntimeError):
    """Raised when no real LLM backend can safely answer a question."""


class MalformedAnswerError(ValueError):
    """The backend's output could not be parsed into a valid answer."""


class LlmBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class QnaAnswer:
    """One generated answer plus its self-reported evidence source."""

    text: str
    source: AnswerSource
    raw_output: str


def _build_context_block(question: str, context: FrameContext) -> str:
    ocr_line = context.ocr_text.strip() or "(no OCR text detected)"
    counts = context.object_counts
    objects_line = (
        ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        if counts
        else "(no object detections available)"
    )
    return (
        f"Detected on-screen text (OCR): {ocr_line}\n"
        f"Detected objects (label: count): {objects_line}\n"
        f"Question: {question}\n"
    )


def _parse_answer(raw_output: str) -> QnaAnswer:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise MalformedAnswerError(f"output is not valid JSON: {raw_output!r}") from exc
    if not isinstance(payload, dict):
        raise MalformedAnswerError(f"output JSON must be an object, got {type(payload).__name__}")

    text = payload.get("answer")
    if not isinstance(text, str) or not text.strip():
        raise MalformedAnswerError(f"'answer' must be a non-blank string, got {text!r}")

    source = payload.get("source")
    if source not in _VALID_SOURCES:
        raise MalformedAnswerError(f"'source' must be one of {_VALID_SOURCES}, got {source!r}")

    return QnaAnswer(text=text.strip(), source=source, raw_output=raw_output)


class AnswerReasoner:
    """Backend-agnostic Q&A answer reasoning with one deterministic retry."""

    def __init__(self, backend: LlmBackend) -> None:
        self._backend = backend

    def answer(self, question: str, context: FrameContext) -> QnaAnswer:
        if not question.strip():
            raise ValueError("question must not be blank")

        context_block = _build_context_block(question, context)
        prompt = f"{SYSTEM_PROMPT}\n\n{context_block}"
        raw_output = self._backend.generate(prompt)
        try:
            return _parse_answer(raw_output)
        except MalformedAnswerError as first_error:
            retry_prompt = (
                f"{prompt}\nYour previous output failed to parse: {first_error}. "
                "Output ONLY valid JSON matching the schema."
            )
            raw_output = self._backend.generate(retry_prompt)
            return _parse_answer(raw_output)  # propagates on a second failure


class HuggingFaceCausalLmBackend:
    """Real local-HuggingFace-model backend. Fails closed by default.

    Mirrors ``scripts/verification/query_decomposition/qwen.py``'s loading
    shape, but refuses to silently download weights: pass
    ``allow_model_download=True`` explicitly, otherwise construction raises
    ``AnswerReasonerUnavailable`` unless the model is already cached
    locally.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        allow_model_download: bool = False,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.allow_model_download = allow_model_download
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any = None
        self._model: Any = None
        self._load()

    def _resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise AnswerReasonerUnavailable(
                "transformers is not installed; install the project's qna extra"
            ) from exc

        local_files_only = not self.allow_model_download
        if local_files_only:
            # Belt-and-suspenders: some transformers/huggingface_hub versions
            # still attempt a network HEAD request under local_files_only=True
            # before falling back to cache, which turns "fail closed" into a
            # slow network-timeout-bound failure instead of an instant one.
            # Force true offline mode whenever download was not explicitly
            # opted into.
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, local_files_only=local_files_only
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, local_files_only=local_files_only
            )
        except Exception as exc:
            mode = "local cache" if local_files_only else "configured model source"
            raise AnswerReasonerUnavailable(
                f"model {self.model_name!r} is unavailable from {mode}: {exc}"
            ) from exc

        device = self._resolve_device()
        self._model.to(device)
        self._model.eval()

    def manifest_metadata(self) -> dict[str, Any]:
        import importlib.metadata

        return {
            "model_id": self.model_name,
            "provider_package": "transformers",
            "provider_package_version": importlib.metadata.version("transformers"),
            "device": self._resolve_device(),
            "local_files_only": not self.allow_model_download,
            "max_new_tokens": self.max_new_tokens,
            "provider_evidence": "REAL_PROVIDER_ARTIFACT",
        }

    def generate(self, prompt: str) -> str:
        import torch

        device = self._resolve_device()
        if getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self._tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))
        ]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
