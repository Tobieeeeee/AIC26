"""Vietnamese to English translation of the Q&A retrieval query.

**Why this exists**, measured on the real 166,550-keyframe organizer index:
the same target video ranked *nowhere* for a Vietnamese query but **#112 of
166,550** for an English one describing the same scene. CLIP's text tower is
English-trained, so a Vietnamese query lands in a poorly-populated region of
the embedding space. Translating the retrieval query is the single largest
retrieval lever available without changing the embedding model. The SOICT
2025 winning system does the same (Qwen3-8B rewrites/translates queries into
well-formed English before retrieval).

**Only the retrieval query is translated.** The *question* stays in its
original language and goes to the answer reasoner untouched: the organizer
accepts Vietnamese or English answers, the answering LLM handles Vietnamese
well, and round-tripping the question through translation would risk
corrupting exactly the detail being asked about.

Translation is **opt-in and degrades safely**: with no translator the query
passes through unchanged, and a backend failure or blank result falls back
to the original text rather than aborting the query -- an untranslated
search still returns something, whereas a raised exception returns nothing.

This supersedes nothing: ``scripts/verification/query_translation/`` remains
the experimental benchmark harness (it compares DeepL/Google/local
backends). This module is the production path, and mirrors that harness's
``Translator`` interface so a benchmarked backend can be dropped in.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Small (~300MB) Vietnamese->English model, the same one the experimental
#: harness uses as its local backend.
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"


class TranslatorUnavailable(RuntimeError):
    """Raised when a real translation backend cannot run safely."""


class QueryTranslator(Protocol):
    def translate(self, text: str) -> str: ...


class PassthroughTranslator:
    """No-op translator: keeps the query exactly as written.

    Useful as an explicit "translation disabled" choice and for tests.
    """

    def translate(self, text: str) -> str:
        return text


def translate_for_retrieval(text: str, translator: QueryTranslator | None) -> str:
    """Translate a retrieval query, falling back to the original on trouble.

    Returns ``text`` unchanged when no translator is configured, when the
    backend raises, or when it returns something blank.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if translator is None:
        return text
    try:
        translated = translator.translate(text)
    except Exception:
        logger.warning("query translation failed; retrieving with the original text", exc_info=True)
        return text
    if not isinstance(translated, str) or not translated.strip():
        logger.warning("query translation returned nothing; retrieving with the original text")
        return text
    return translated.strip()


class HelsinkiViEnTranslator:
    """Local HuggingFace Vietnamese->English translation. Fails closed.

    Same policy as ``mvr.enrichment.qna_answer.HuggingFaceCausalLmBackend``:
    weights must already be cached locally unless download is explicitly
    opted into, so a missing model fails fast and loudly instead of
    silently pulling hundreds of MB mid-query.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_TRANSLATION_MODEL,
        device: str = "cpu",
        allow_model_download: bool = False,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device_spec = device
        self.allow_model_download = allow_model_download
        self.max_length = max_length
        self.num_beams = num_beams
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslatorUnavailable(
                "transformers is not installed; install the project's qna extra"
            ) from exc

        local_files_only = not self.allow_model_download
        if local_files_only:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=local_files_only
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, local_files_only=local_files_only
            )
        except Exception as exc:
            mode = "local cache" if local_files_only else "configured model source"
            raise TranslatorUnavailable(
                f"translation model {self.model_name!r} is unavailable from {mode}: {exc}"
            ) from exc

        if self.device_spec == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device_spec)
        self._model.eval()
        self._model.to(self._device)

    def manifest_metadata(self) -> dict[str, Any]:
        import importlib.metadata

        return {
            "model_id": self.model_name,
            "provider_package": "transformers",
            "provider_package_version": importlib.metadata.version("transformers"),
            "device": str(self._device),
            "local_files_only": not self.allow_model_download,
            "provider_evidence": "REAL_PROVIDER_ARTIFACT",
        }

    def translate(self, text: str) -> str:
        import torch

        if not text.strip():
            return ""
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                do_sample=False,
            )
        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if not decoded:
            raise TranslatorUnavailable("translation model returned no output")
        return decoded[0].strip()
