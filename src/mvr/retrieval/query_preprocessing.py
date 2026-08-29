"""Shared query preprocessing for the KIS / Q&A / TRAKE retrieval paths.

One translation policy, used by every retrieval entry point so the dense
(semantic) branch and the lexical (BM25) branch see consistent inputs:

- dense  : English (Vietnamese query translated, fail-closed to the original)
- lexical: original Vietnamese (OCR/ASR text is indexed in Vietnamese)

Translation sources are the verified local Helsinki-NLP Vi→En model and the
Qwen-based query decomposer that both live under ``scripts/verification/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from mvr.enrichment.qna_translation import translate_for_retrieval

DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"


@dataclass(frozen=True)
class ProcessedQuery:
    """Dense (translated) and lexical (original) forms of a query."""

    dense: str
    lexical: str


class QueryPreprocessingService:
    """Single source of truth for query translation policy.

    Reads the same env vars the TRAKE path already honours:
    ``MVR_TRAKE_TRANSLATE_QUERIES``, ``MVR_TRAKE_TRANSLATION_MODEL``,
    ``MVR_TRAKE_ALLOW_MODEL_DOWNLOAD``, ``MVR_DEVICE``.
    """

    def __init__(self, translator_factory=None) -> None:
        self._translator_factory = translator_factory
        self._translator = None
        self._load_error: Optional[BaseException] = None

    def _should_translate(self) -> bool:
        if self._translator_factory is not None:
            return True
        flag = os.environ.get("MVR_TRAKE_TRANSLATE_QUERIES", "").strip().lower()
        return flag in ("1", "true", "yes")

    def _get_translator(self):
        if self._translator is not None:
            return self._translator
        if self._load_error is not None:
            return None
        factory = self._translator_factory or _default_translator_factory
        try:
            self._translator = factory()
        except BaseException as exc:
            self._load_error = exc
            return None
        return self._translator

    def translate(self, query: str) -> str:
        """English query for the dense branch; falls back to the original."""
        if not self._should_translate():
            return query
        translator = self._get_translator()
        return translate_for_retrieval(query, translator)

    def prepare(self, query: str) -> ProcessedQuery:
        """Dense (translated) + lexical (original) query pair."""
        dense = self.translate(query)
        return ProcessedQuery(dense=dense, lexical=query)


def _default_translator_factory():
    """Local Helsinki-NLP Vi→En model, cached unless download is opted in."""
    from mvr.enrichment.qna_translation import HelsinkiViEnTranslator

    model_name = (
        os.environ.get("MVR_TRAKE_TRANSLATION_MODEL", "").strip()
        or DEFAULT_TRANSLATION_MODEL
    )
    allow_download = (
        os.environ.get("MVR_TRAKE_ALLOW_MODEL_DOWNLOAD", "").strip().lower()
        in ("1", "true", "yes")
    )
    device = os.environ.get("MVR_DEVICE", "auto")
    return HelsinkiViEnTranslator(
        model_name=model_name,
        allow_model_download=allow_download,
        device=device,
    )


def build_keyword_query(concepts) -> str:
    """Deterministic, deduplicated keyword string from decomposed concepts.

    Source: ``scripts/verification/query_decomposition/utils.py``.
    """
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
    seen: set[str] = set()
    keywords: list[str] = []
    for concept_list in order:
        for concept in concept_list:
            lower = concept.lower()
            if lower not in seen:
                seen.add(lower)
                keywords.append(concept)
    return ", ".join(keywords)
