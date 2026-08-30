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
    """Normalize a keyframe reference (CSV ``n``, "003.jpg", or "003") to one key.

    Raises ``MissingFrameMappingError`` (never a bare ``ValueError``) for a
    non-numeric reference: a reference this repo's own pipeline produced
    (e.g. ``"L01_V001_S0002_K01"``) is not a verified AIC mapping, and
    callers rely on this specific exception type to drop the candidate
    rather than crash -- see ``resolve_original_frame_id()`` in
    ``mvr.retrieval.kis``.
    """
    text = str(value).strip()
    if text.lower().endswith(".jpg"):
        text = text[: -len(".jpg")]
    try:
        return str(int(text))
    except ValueError as exc:
        raise MissingFrameMappingError(
            f"keyframe reference must be numeric, got {value!r}; refusing to guess"
        ) from exc


class AicDatasetRegistry:
    """Original-video frame registry backed by an optional AIC dataset root.

    Two directory layouts are read, normalized first::

        normalized  <root>/videos/<video_id>.mp4
                    <root>/keyframes/<video_id>/<NNN>.jpg
                    <root>/map_keyframes/<video_id>.csv
                    <root>/media_info/<video_id>.json

        raw         <root>/Videos_L21_a/video/<video_id>.mp4
                    <root>/Keyframes_L26_a/keyframes/<video_id>/<NNN>.jpg
                    <root>/map-keyframes-aic25-b1/map-keyframes/<video_id>.csv
                    <root>/media-info-aic25-b1/media-info/<video_id>.json

        R2 mirror   <root>/<batch>/videos/<collection>/<video_id>.mp4

    The R2 reorg (main ``712274d``) retired the organizer's ``Videos_*/video/``
    shape inside the bucket in favour of
    ``data/raw/{batch}/videos/{collection}/`` -- see
    ``mvr.operations.r2_layout.raw_video_key()``. A root mirrored from R2 after
    that move nests videos one level deeper and groups them by collection, so
    neither older shape finds them. Only videos moved; the reorg plan
    (``scripts/operations/r2_reng.py``) touches nothing else.

    The raw shape is what the organizer actually ships: one folder per asset
    kind, carrying a batch suffix, with an inner directory whose name uses
    hyphens where the normalized tree uses underscores. Reading only the
    normalized tree meant a registry pointed at the real download resolved
    nothing at all -- every TRAKE run returned "(no candidate chains found)",
    and Q&A saw no object detections. ``build_keyframe_mapping.py`` has read
    both layouts all along, which is why KIS was unaffected.

    Every subdirectory is optional.
    """

    # kind -> directory globs relative to the dataset root, best first.
    RAW_LAYOUT = {
        "videos": ("videos", "Videos_*/video", "*/videos/*"),
        "keyframes": ("keyframes", "Keyframes_*/keyframes"),
        "map_keyframes": ("map_keyframes", "map-keyframes*/map-keyframes"),
        "media_info": ("media_info", "media-info*/media-info"),
    }

    def __init__(self, root: Path | str | None) -> None:
        self.root = Path(root) if root is not None else None
        self._frame_maps: dict[str, dict[str, AicFrameMapping]] = {}

    def _directories(self, kind: str) -> list[Path]:
        """Every directory that could hold this asset kind, best first.

        The normalized directory leads: a normalized tree is something a
        person built deliberately from the raw download, and letting the raw
        copy win would silently discard that work.

        Deliberately re-globbed on every call rather than cached. The
        ingestion pipeline writes keyframes and then asks for them through a
        registry that already existed, so a list of directories captured
        earlier would answer "nothing here" about files that are now on disk.
        The cost is one listing of the dataset root beside the per-video
        ``*.jpg`` scan ``keyframe_path`` already does.
        """
        if self.root is None:
            return []
        found: list[Path] = []
        for pattern in self.RAW_LAYOUT[kind]:
            for candidate in sorted(self.root.glob(pattern)):
                if candidate not in found and candidate.is_dir():
                    found.append(candidate)
        return found

    # -- path helpers --------------------------------------------------

    def video_path(self, video_id: str) -> Path | None:
        return self._first_file("videos", f"{video_id}.mp4")

    def _first_file(self, kind: str, filename: str) -> Path | None:
        for directory in self._directories(kind):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
        return None

    def keyframe_path(self, video_id: str, keyframe_ref: str) -> Path | None:
        """Locate the organizer keyframe JPEG by numeric match, not by guessed
        zero-padding width (observed AIC exports are 3-digit, e.g. "003.jpg",
        but this must not assume that width is universal).
        """
        target = _normalize_ref(keyframe_ref)
        for root in self._directories("keyframes"):
            directory = root / video_id
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.jpg")):
                try:
                    if _normalize_ref(candidate.stem) == target:
                        return candidate
                except MissingFrameMappingError:
                    continue
        return None

    def _map_keyframes_csv(self, video_id: str) -> Path | None:
        return self._first_file("map_keyframes", f"{video_id}.csv")

    def _media_info_path(self, video_id: str) -> Path | None:
        return self._first_file("media_info", f"{video_id}.json")

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
