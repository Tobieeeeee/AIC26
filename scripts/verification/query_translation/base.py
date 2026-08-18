"""Abstract base for all translation backends."""
from __future__ import annotations

import abc


class Translator(abc.ABC):
    """Common interface that every translation backend must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this translation method."""

    @abc.abstractmethod
    def translate(self, text: str) -> str:
        """Translate *text* from Vietnamese to English.

        Must return plain English text.
        Raises on unrecoverable failure.
        """
