"""DeepL Translation backend."""
from __future__ import annotations

import os

from .base import Translator


class DeepLTranslator(Translator):
    """Vietnamese → English through the DeepL API.

    The API key is read from the ``DEEPL_API_KEY`` environment variable.

    If the ``deepl`` package is missing or the API key is not set the
    constructor raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        try:
            import deepl as deepl_lib
        except ImportError as exc:
            raise RuntimeError(
                "deepl package is not installed"
            ) from exc

        api_key = os.environ.get("DEEPL_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError(
                "DEEPL_API_KEY is not configured"
            )

        self._translator = deepl_lib.Translator(api_key)

    @property
    def name(self) -> str:
        return "deepl"

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        result = self._translator.translate_text(
            text,
            source_lang="VI",
            target_lang="EN-US",
        )

        return str(result.text).strip()
