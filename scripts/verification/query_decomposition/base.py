from typing import Tuple, Optional
from .models import QueryConcepts

class QueryDecomposer:
    """Base interface for query decomposition."""

    @property
    def name(self) -> str:
        """Name of the decomposer."""
        raise NotImplementedError

    def decompose(self, text: str) -> Tuple[QueryConcepts, str, Optional[str]]:
        """
        Decompose English query text into structured concepts and a raw output string.
        Returns:
            concepts: Parsed QueryConcepts object.
            keyword_query: The deterministic keyword string.
            raw_output: Raw model output string (for debugging).
        Raises:
            Exception if generation or parsing fails permanently.
        """
        raise NotImplementedError
