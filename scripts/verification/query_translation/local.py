"""Local HuggingFace Vi→En translation backend."""
from __future__ import annotations

from .base import Translator

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-vi-en"


class LocalTranslator(Translator):
    """Vietnamese → English using a local HuggingFace translation model.

    The model identifier is configurable.  The model is loaded once on
    construction and reused across all translate() calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pipeline = self._load(model_name, device)

    @staticmethod
    def _load(model_name: str, device: str):
        from transformers import pipeline as hf_pipeline

        return hf_pipeline(
            "translation",
            model=model_name,
            device=device if device != "auto" else None,
            # Deterministic generation — no sampling.
            num_beams=4,
        )

    @property
    def name(self) -> str:
        return "local_vi2en"

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        results = self._pipeline(
            text,
            max_length=512,
            num_beams=4,
            do_sample=False,
        )

        if not results:
            raise RuntimeError("local model returned no output")

        return str(results[0]["translation_text"]).strip()
