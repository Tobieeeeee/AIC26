"""Shared text normalization for Task05A searchable feature preparation."""
from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def clean_display_text(value: object) -> str:
    """Return NFC text with surrounding/repeated whitespace normalized.

    The returned value is the human-readable/searchable display form.  It
    intentionally does not remove Vietnamese diacritics or otherwise alter
    lexical content.
    """

    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFC", value)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_text(value: object) -> str:
    """Return the deterministic lexical identity used for deduplication."""

    return clean_display_text(value).casefold()


def normalize_lexical_query(value: object) -> str:
    """Normalize a lexical query without removing Vietnamese diacritics.

    The policy is deliberately identical at every Elasticsearch entry point:
    Unicode NFC, collapsed whitespace, trim, then Unicode casefold. Accent
    folding belongs exclusively to Elasticsearch's ``vi_folded`` analyzer.
    """

    return normalize_text(value)


def is_punctuation_or_symbol_only(value: object) -> bool:
    """Whether non-whitespace characters are all Unicode punctuation/symbols."""

    text = clean_display_text(value)
    if not text:
        return False
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return False
    return all(unicodedata.category(char)[0] in {"P", "S"} for char in non_space)
