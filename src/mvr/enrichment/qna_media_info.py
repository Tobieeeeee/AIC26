"""Organizer ``media_info`` as a Q&A evidence and diagnostic source.

Each AIC video ships a YouTube-style metadata JSON (title, description,
keywords, publish date, channel). Unlike the object-detection files -- which
cover only 59 of 873 videos -- **media_info covers all 873**, making it the
one per-video signal available across the whole corpus without running any
model.

Two uses, both in Q&A scope:

1. **Diagnostic**: search the corpus for terms taken from a Q&A query to
   find which video probably contains the answer. This answers "is the
   answer video even in our semantic index?" -- a question that otherwise
   requires indexing all 873 videos to settle. Measured on real queries:
   the FANA charity query resolves to ``L30_V072`` and the Nguyễn Trung Trực
   query to ``L24_V011``, neither of which is in the L21+L22 test index,
   which explains those queries' failures without implicating the model.

2. **Context**: ``MediaInfo.as_prompt_text()`` renders a compact block for
   the answer prompt. Several real Q&A questions ask about programme,
   place, or date context that the video's own title/description states
   directly.

Matching is diacritic-insensitive in both directions: organizer titles are
inconsistently accented (``"TIN TỨC 60 GI Y"`` appears in real keyword
lists), and a searcher typing Vietnamese with or without tone marks should
find the same videos.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_WHITESPACE_RE = re.compile(r"\s+")



def normalize_text(value: object) -> str:
    """Casefold, strip Vietnamese diacritics, and collapse whitespace."""
    decomposed = unicodedata.normalize("NFD", str(value)).casefold()
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return _WHITESPACE_RE.sub(" ", stripped).strip()


@dataclass(frozen=True)
class MediaInfo:
    """One video's organizer descriptive metadata."""

    video_id: str
    title: str = ""
    description: str = ""
    keywords: tuple[str, ...] = ()
    author: str = ""
    publish_date: str = ""
    watch_url: str = ""

    @classmethod
    def from_payload(cls, video_id: str, payload: dict) -> "MediaInfo":
        raw_keywords = payload.get("keywords") or []
        keywords = tuple(str(k) for k in raw_keywords) if isinstance(raw_keywords, list) else ()
        return cls(
            video_id=video_id,
            title=str(payload.get("title") or ""),
            description=str(payload.get("description") or ""),
            keywords=keywords,
            author=str(payload.get("author") or ""),
            publish_date=str(payload.get("publish_date") or ""),
            watch_url=str(payload.get("watch_url") or ""),
        )

    def searchable_text(self) -> str:
        return normalize_text(
            " ".join([self.title, self.description, " ".join(self.keywords), self.author])
        )

    def as_prompt_text(self, *, max_description_chars: int = 300) -> str:
        """Compact block for the answer prompt.

        Descriptions are mostly boilerplate (subscribe links, hashtags) and
        run ~1000 characters, so they are truncated: the informative part is
        at the front, and a long tail would crowd out the OCR evidence that
        matters more.
        """
        description = self.description.strip()
        if len(description) > max_description_chars:
            description = description[:max_description_chars].rstrip() + "..."
        lines = []
        if self.title.strip():
            lines.append(f"Video title: {self.title.strip()}")
        if self.author.strip():
            lines.append(f"Channel: {self.author.strip()}")
        if self.publish_date.strip():
            lines.append(f"Published: {self.publish_date.strip()}")
        if description:
            lines.append(f"Video description: {description}")
        return "\n".join(lines)


@dataclass(frozen=True)
class MediaInfoHit:
    video_id: str
    matched_terms: int
    info: MediaInfo


class MediaInfoSearchIndex:
    """In-memory, normalized search over the corpus' descriptive metadata.

    The whole corpus is ~1.3MB of text, so it is loaded eagerly rather than
    indexed on disk.
    """

    def __init__(self, entries: dict[str, MediaInfo]) -> None:
        self._entries = dict(entries)
        self._normalized = {vid: info.searchable_text() for vid, info in self._entries.items()}

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_directory(cls, directory: Path | str) -> "MediaInfoSearchIndex":
        """Load every ``<video_id>.json`` under ``directory``.

        A malformed or unreadable file is skipped rather than aborting the
        whole corpus load -- organizer metadata is explicitly optional and
        some videos ship none.
        """
        entries: dict[str, MediaInfo] = {}
        for path in sorted(Path(directory).glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            entries[path.stem] = MediaInfo.from_payload(path.stem, payload)
        return cls(entries)

    def get(self, video_id: str) -> MediaInfo | None:
        return self._entries.get(video_id)

    def video_ids(self) -> Iterable[str]:
        return self._entries.keys()

    def search(
        self, terms: Sequence[str], *, limit: int | None = None
    ) -> list[MediaInfoHit]:
        """Rank videos by how many of ``terms`` appear in their metadata.

        Substring matching on normalized text -- deliberately simple, since
        this is a lookup aid for a human, not a retrieval component. Videos
        matching no term are excluded entirely.
        """
        normalized_terms = [normalize_text(t) for t in terms]
        normalized_terms = [t for t in normalized_terms if t]
        if not normalized_terms:
            return []

        hits: list[MediaInfoHit] = []
        for video_id, text in self._normalized.items():
            matched = sum(1 for term in normalized_terms if term in text)
            if matched:
                hits.append(
                    MediaInfoHit(
                        video_id=video_id,
                        matched_terms=matched,
                        info=self._entries[video_id],
                    )
                )
        hits.sort(key=lambda h: (-h.matched_terms, h.video_id))
        return hits if limit is None else hits[:limit]
