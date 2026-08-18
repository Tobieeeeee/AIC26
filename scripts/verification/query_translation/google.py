"""Google Cloud Translation backend."""
from __future__ import annotations

import os

from .base import Translator


class GoogleTranslator(Translator):
    """Vietnamese → English through Google Cloud Translation API.

    Credentials are read from the environment:

    - ``GOOGLE_APPLICATION_CREDENTIALS`` — path to service-account JSON, or
    - Application Default Credentials configured via ``gcloud auth``.

    If the ``google-cloud-translate`` package is missing or credentials are
    unavailable the constructor raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        try:
            from google.cloud import translate_v2 as translate
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-translate is not installed"
            ) from exc

        try:
            self._client = translate.Client()
            # Quick connectivity/credential check.
            self._client.get_languages()
        except Exception as exc:
            raise RuntimeError(
                "Google Translation API is not available "
                "(check credentials / network)"
            ) from exc

    @property
    def name(self) -> str:
        return "google_translate"

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        result = self._client.translate(
            text,
            source_language="vi",
            target_language="en",
            format_="text",
        )

        return str(result["translatedText"]).strip()
