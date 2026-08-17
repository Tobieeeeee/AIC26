"""Local HuggingFace Vi→En translation backend."""
from __future__ import annotations

from .base import Translator

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-vi-en"


class LocalTranslator(Translator):
    """Vietnamese → English using a local HuggingFace translation model.

    The tokenizer and model are loaded once on construction and reused
    across all translate() calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._tokenizer, self._model = self._load(
            model_name, self._device
        )

    @staticmethod
    def _resolve_device(device: str):
        import torch

        if device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(device)

    @staticmethod
    def _load(model_name: str, device):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        model.to(device)
        return tokenizer, model

    @property
    def name(self) -> str:
        return "local_vi2en"

    def translate(self, text: str) -> str:
        import torch

        if not text.strip():
            return ""

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        gen_kwargs = {
            "max_length": 512,
            "num_beams": 4,
            "do_sample": False,
        }

        if hasattr(self._tokenizer, "lang_code_to_id"):
            for lang in ["en_XX", "eng_Latn", "en"]:
                if lang in self._tokenizer.lang_code_to_id:
                    gen_kwargs["forced_bos_token_id"] = self._tokenizer.lang_code_to_id[lang]
                    break

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs
            )

        decoded = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        if not decoded:
            raise RuntimeError("local model returned no output")

        return decoded[0].strip()
