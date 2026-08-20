"""TRAKE event/gap input contract and gap-phrase translation.

Represents a TRAKE query that has ALREADY been decomposed by the organizer
into k numbered events (E1..Ek) plus the inter-event timing/spatial clues
found in the text between them. Automatic decomposition of a raw query into
this shape (e.g. via an LLM) is out of scope here -- see
agent_task/trake_competition_ready/03_EVENT_INPUT_CONTRACT_AND_GAP_TRANSLATION.md.

The gap-phrase table below is a heuristic starting point calibrated by the
course lecturer from real AIC 2025 queries, not an organizer-published
spec -- kept as an overridable module-level constant so it can be tuned
later without touching alignment logic.
"""
from __future__ import annotations

from dataclasses import dataclass

# Order matters: matching is substring-based and first-match-wins, so a more
# specific phrase that CONTAINS a shorter one (e.g. "ngay sau đó" contains
# "sau đó") must be declared first, or the shorter phrase would shadow it.
GAP_PHRASE_TABLE: dict[str, tuple[float, float]] = {
    "ngay sau đó": (0.0, 10.0),
    "liền": (0.0, 10.0),
    "rồi": (0.0, 60.0),
    "sau đó": (0.0, 60.0),
    "một lúc sau": (30.0, 300.0),
    "lát sau": (30.0, 300.0),
}

DEFAULT_GAP_SECONDS: tuple[float, float] = (0.0, 300.0)


@dataclass(frozen=True)
class TrakeEventQuery:
    """One event's own sub-query text, ready to pass to search_kis()."""

    index: int
    text: str

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError(f"event index must be >= 1, got {self.index}")
        if not self.text or not self.text.strip():
            raise ValueError("event text must be a non-empty string")


@dataclass(frozen=True)
class TrakeGapConstraint:
    min_gap_seconds: float
    max_gap_seconds: float

    def __post_init__(self) -> None:
        if self.min_gap_seconds < 0 or self.max_gap_seconds < 0:
            raise ValueError(
                "gap bounds must be >= 0, got "
                f"min={self.min_gap_seconds}, max={self.max_gap_seconds}"
            )
        if self.min_gap_seconds > self.max_gap_seconds:
            raise ValueError(
                f"min_gap_seconds ({self.min_gap_seconds}) must be <= "
                f"max_gap_seconds ({self.max_gap_seconds})"
            )


@dataclass(frozen=True)
class TrakeQuery:
    """A decomposed k-event TRAKE query with k-1 inter-event gap constraints.

    ``gaps[i]`` is the constraint between ``events[i]`` and ``events[i+1]``.
    """

    events: tuple[TrakeEventQuery, ...]
    gaps: tuple[TrakeGapConstraint, ...]

    def __post_init__(self) -> None:
        if len(self.events) < 2:
            raise ValueError(
                f"a TRAKE query must have at least 2 events (got {len(self.events)}); "
                "a single-event chain is plain KIS -- use search_kis() directly"
            )
        if len(self.gaps) != len(self.events) - 1:
            raise ValueError(
                f"expected {len(self.events) - 1} gaps for {len(self.events)} "
                f"events, got {len(self.gaps)}"
            )
        for position, event in enumerate(self.events, start=1):
            if event.index != position:
                raise ValueError(
                    "events must be strictly sequential starting at 1; "
                    f"event at position {position} has index {event.index}"
                )


def translate_gap_phrase(text_between_events: str) -> TrakeGapConstraint:
    """Translate the text connecting two consecutive events into a gap
    constraint, via first-match-wins substring lookup against
    ``GAP_PHRASE_TABLE``, falling back to ``DEFAULT_GAP_SECONDS`` when no
    phrase matches."""
    for phrase, (min_gap, max_gap) in GAP_PHRASE_TABLE.items():
        if phrase in text_between_events:
            return TrakeGapConstraint(min_gap_seconds=min_gap, max_gap_seconds=max_gap)
    default_min, default_max = DEFAULT_GAP_SECONDS
    return TrakeGapConstraint(min_gap_seconds=default_min, max_gap_seconds=default_max)
