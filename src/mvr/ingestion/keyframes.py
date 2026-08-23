from pathlib import Path

from .models import Keyframe, Shot


def select_frame_indices(
    shot: Shot,
    long_shot_seconds: float = 12.0,
    sample_every_seconds: float = 8.0,
) -> list[int]:
    """Chọn frame giữa và bổ sung frame định kỳ cho shot dài."""

    if shot.fps <= 0:
        raise ValueError("shot.fps phải lớn hơn 0")

    if shot.end_frame < shot.start_frame:
        raise ValueError("end_frame phải >= start_frame")

    if long_shot_seconds < 0:
        raise ValueError("long_shot_seconds không được âm")

    if sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds phải lớn hơn 0")

    # Luôn lấy frame giữa.
    midpoint = (shot.start_frame + shot.end_frame) // 2
    selected = {midpoint}

    # Shot dài thì lấy thêm một frame sau mỗi khoảng sample_every_seconds.
    if shot.duration >= long_shot_seconds:
        step = max(1, round(sample_every_seconds * shot.fps))

        frame_idx = shot.start_frame + step

        while frame_idx < shot.end_frame:
            selected.add(frame_idx)
            frame_idx += step

    # Bảo đảm sorted, unique và nằm trong shot.
    return sorted(
        index
        for index in selected
        if shot.start_frame <= index <= shot.end_frame
    )


def make_keyframes(
    shots: list[Shot],
    long_shot_seconds: float,
    sample_every_seconds: float,
) -> list[Keyframe]:
    """Tạo metadata và đường dẫn cho các keyframe."""

    keyframes: list[Keyframe] = []

    for shot in shots:
        indices = select_frame_indices(
            shot,
            long_shot_seconds,
            sample_every_seconds,
        )

        for ordinal, frame_idx in enumerate(indices, start=1):
            keyframe_id = (
                f"{shot.video_id}_"
                f"S{shot.shot_id:04d}_"
                f"K{ordinal:02d}"
            )

            relative_path = (
                Path("keyframes")
                / shot.video_id
                / f"{keyframe_id}.jpg"
            )

            keyframes.append(
                Keyframe(
                    keyframe_id=keyframe_id,
                    video_id=shot.video_id,
                    shot_id=shot.shot_id,
                    ordinal=ordinal,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / shot.fps,
                    image_path=relative_path.as_posix(),
                )
            )

    return keyframes


def make_keyframes_from_indices(
    shots: list[Shot],
    selected_by_shot: dict[tuple[str, int], list[int]],
) -> list[Keyframe]:
    """Build canonical identities for an externally selected frame set.

    The formatter is shared by the hybrid selector so changing selection policy
    cannot change the established ``K01``, ``K02``, ... identity semantics.
    """

    keyframes: list[Keyframe] = []
    for shot in shots:
        indices = sorted(set(selected_by_shot.get((shot.video_id, shot.shot_id), [])))
        if not indices:
            raise ValueError(
                f"shot {shot.video_id}/S{shot.shot_id:04d} has no selected keyframe"
            )
        if indices[0] < shot.start_frame or indices[-1] > shot.end_frame:
            raise ValueError(
                f"selected frame is outside shot {shot.video_id}/S{shot.shot_id:04d}"
            )
        for ordinal, frame_idx in enumerate(indices, start=1):
            keyframe_id = (
                f"{shot.video_id}_S{shot.shot_id:04d}_K{ordinal:02d}"
            )
            relative_path = (
                Path("keyframes") / shot.video_id / f"{keyframe_id}.jpg"
            )
            keyframes.append(
                Keyframe(
                    keyframe_id=keyframe_id,
                    video_id=shot.video_id,
                    shot_id=shot.shot_id,
                    ordinal=ordinal,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / shot.fps,
                    image_path=relative_path.as_posix(),
                )
            )
    return keyframes
