"""AIC organizer support-data adapter and original-video frame registry.

Resolves KIS candidates to an authoritative ``(video_id, frame_idx)`` pair on
the ORIGINAL video. Organizer-provided keyframes, the ``map_keyframes`` CSV
mapping, and ``media_info`` descriptive metadata are optional accelerators;
the raw-video-only path (probing the video file directly) must keep working
when none of them are present for a given video.

Critical rule verified against real AIC data: the CSV column ``frame_idx`` is
the original-video frame index and is the only valid source for the KIS
``frame_id``. The column ``n`` (organizer keyframe ordinal) is only ever used
as a lookup key into the mapping, never as the KIS ``frame_id`` itself.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .models import VideoInfo
from .probe import probe_video


@dataclass(frozen=True)
class AicFrameMapping:
    """One organizer keyframe resolved to its original-video frame position."""

    video_id: str
    keyframe_ref: str
    frame_idx: int
    timestamp: float
    fps: float


class MissingFrameMappingError(LookupError):
    """A ``(video_id, keyframe_ref)`` mapping was requested but is unknown.

    Deliberately never guessed or interpolated: callers must handle this
    explicitly rather than receive a fabricated frame index.
    """


def _normalize_ref(value: str) -> str:
    """Normalize a keyframe reference (CSV ``n``, "003.jpg", or "003") to one key."""
    text = str(value).strip()
    if text.lower().endswith(".jpg"):
        text = text[: -len(".jpg")]
    try:
        return str(int(text))
    except ValueError as exc:
        raise ValueError(f"keyframe reference must be numeric, got {value!r}") from exc


class AicDatasetRegistry:
    """Original-video frame registry backed by an optional AIC dataset root.

    Expected directory layout (every subdirectory is optional except at least
    one video file under ``videos/``)::

        <root>/videos/<video_id>.mp4
        <root>/keyframes/<video_id>/<NNN>.jpg
        <root>/map_keyframes/<video_id>.csv   (n, pts_time, fps, frame_idx)
        <root>/media_info/<video_id>.json     (YouTube descriptive metadata only)
    """

    def __init__(self, root: Path | str | None) -> None:
        self.root = Path(root) if root is not None else None
        self._frame_maps: dict[str, dict[str, AicFrameMapping]] = {}

    # -- path helpers --------------------------------------------------

    def video_path(self, video_id: str) -> Path | None:
        if self.root is None:
            return None
        candidate = self.root / "videos" / f"{video_id}.mp4"
        return candidate if candidate.is_file() else None

    def keyframe_path(self, video_id: str, keyframe_ref: str) -> Path | None:
        """Locate the organizer keyframe JPEG by numeric match, not by guessed
        zero-padding width (observed AIC exports are 3-digit, e.g. "003.jpg",
        but this must not assume that width is universal).
        """
        if self.root is None:
            return None
        directory = self.root / "keyframes" / video_id
        if not directory.is_dir():
            return None
        target = _normalize_ref(keyframe_ref)
        for candidate in sorted(directory.glob("*.jpg")):
            try:
                if _normalize_ref(candidate.stem) == target:
                    return candidate
            except ValueError:
                continue
        return None

    def _map_keyframes_csv(self, video_id: str) -> Path | None:
        if self.root is None:
            return None
        candidate = self.root / "map_keyframes" / f"{video_id}.csv"
        return candidate if candidate.is_file() else None

    def _media_info_path(self, video_id: str) -> Path | None:
        if self.root is None:
            return None
        candidate = self.root / "media_info" / f"{video_id}.json"
        return candidate if candidate.is_file() else None

    # -- frame mapping (organizer support data) -------------------------

    def _load_frame_map(self, video_id: str) -> dict[str, AicFrameMapping]:
        if video_id in self._frame_maps:
            return self._frame_maps[video_id]
        mapping: dict[str, AicFrameMapping] = {}
        csv_path = self._map_keyframes_csv(video_id)
        if csv_path is not None:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    ref = _normalize_ref(row["n"])
                    mapping[ref] = AicFrameMapping(
                        video_id=video_id,
                        keyframe_ref=ref,
                        frame_idx=int(row["frame_idx"]),
                        timestamp=float(row["pts_time"]),
                        fps=float(row["fps"]),
                    )
        self._frame_maps[video_id] = mapping
        return mapping

    def resolve_frame(self, video_id: str, keyframe_ref: str) -> AicFrameMapping:
        """Resolve an organizer keyframe to its verified KIS ``(video_id, frame_idx)``.

        ``keyframe_ref`` may be the CSV ``n`` column, a keyframe filename such
        as ``"003.jpg"``, or the bare stem ``"003"``; all normalize to the same
        lookup. Keying is always composite on ``(video_id, keyframe_ref)``, so
        the same ``keyframe_ref`` in two different videos never collides.

        Raises ``MissingFrameMappingError`` rather than guessing when no
        verified mapping exists for this pair.
        """
        mapping = self._load_frame_map(video_id)
        ref = _normalize_ref(keyframe_ref)
        try:
            return mapping[ref]
        except KeyError as exc:
            raise MissingFrameMappingError(
                f"no verified frame_idx mapping for (video_id={video_id!r}, "
                f"keyframe={keyframe_ref!r}); refusing to guess"
            ) from exc

    # -- video-level info: the raw-video-first path ----------------------

    def video_info(self, video_id: str) -> VideoInfo:
        """Probe the original video directly.

        This is the raw-video-first path: it works even when no organizer
        support assets (keyframes, CSV mapping, media_info) exist for this
        video, as long as the video file itself is present under
        ``<root>/videos/``.
        """
        path = self.video_path(video_id)
        if path is None:
            root_desc = self.root / "videos" if self.root is not None else "<no dataset root configured>"
            raise FileNotFoundError(
                f"no original video found for video_id={video_id!r} under {root_desc}"
            )
        return probe_video(path)

    # -- optional descriptive metadata -----------------------------------

    def media_info(self, video_id: str) -> dict | None:
        """Optional YouTube-style descriptive metadata (title, channel, etc.).

        This is never a source of frame-mapping data (it carries no fps or
        frame information) and its absence must not affect ``resolve_frame()``
        or ``video_info()``.
        """
        path = self._media_info_path(video_id)
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
