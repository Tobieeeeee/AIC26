from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

from .models import Keyframe


def _probe_resolution(video_path: Path) -> tuple[int, int]:
    """Đọc chiều rộng và chiều cao video bằng ffprobe."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe thất bại với {video_path}:\n"
            f"{result.stderr.strip()}"
        )

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]

        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Không đọc được kích thước video: {video_path}"
        ) from error

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Kích thước video không hợp lệ: {width}x{height}"
        )

    return width, height


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Đọc đúng số byte của một raw frame."""

    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            break

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Dừng tiến trình FFmpeg an toàn."""

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def extract_keyframes(
    video_path: Path,
    frames: list[Keyframe],
    output_root: Path,
) -> None:
    """
    Trích keyframe chính xác bằng một lần scan video.

    FFmpeg thứ nhất:
        đọc video đúng một lần và xuất raw frame.

    Python:
        chỉ chuyển các frame cần lấy.

    FFmpeg thứ hai:
        nhận các raw frame đã chọn và ghi thành JPEG.

    Không sử dụng OpenCV.imwrite và không tạo biểu thức
    select=eq(n,...)+eq(n,...) quá dài.
    """

    if not frames:
        return

    video_path = Path(video_path)
    output_root = Path(output_root)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy video: {video_path}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    # Một frame index có thể được dùng bởi nhiều Keyframe.
    by_frame_idx: dict[int, list[Keyframe]] = defaultdict(list)

    for frame in frames:
        if frame.frame_idx < 0:
            raise ValueError(
                f"frame_idx không được âm: {frame.frame_idx}"
            )

        by_frame_idx[frame.frame_idx].append(frame)

        destination = output_root / frame.image_path
        destination.parent.mkdir(parents=True, exist_ok=True)

    target_indices = sorted(by_frame_idx)

    if not target_indices:
        return

    width, height = _probe_resolution(video_path)

    # rgb24: mỗi pixel gồm 3 byte.
    bytes_per_frame = width * height * 3

    with tempfile.TemporaryDirectory(
        prefix="shotlab_",
        dir=output_root,
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        temp_pattern = temp_dir / "selected_%06d.jpg"

        # FFmpeg 1: scan video đúng một lần.
        decoder_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-noautorotate",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "rgb24",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        # FFmpeg 2: lưu riêng các raw frame được Python chuyển qua.
        encoder_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-an",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(temp_pattern),
        ]

        decoder = subprocess.Popen(
            decoder_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        encoder = subprocess.Popen(
            encoder_command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if decoder.stdout is None or decoder.stderr is None:
            _stop_process(decoder)
            _stop_process(encoder)
            raise RuntimeError(
                "Không tạo được pipe đầu ra cho FFmpeg decoder."
            )

        if encoder.stdin is None or encoder.stderr is None:
            _stop_process(decoder)
            _stop_process(encoder)
            raise RuntimeError(
                "Không tạo được pipe đầu vào cho FFmpeg encoder."
            )

        target_position = 0
        current_frame_idx = 0
        extracted_count = 0

        try:
            while target_position < len(target_indices):
                raw_frame = _read_exact(
                    decoder.stdout,
                    bytes_per_frame,
                )

                # Video đã kết thúc trước frame cần lấy.
                if len(raw_frame) != bytes_per_frame:
                    break

                target_frame_idx = target_indices[target_position]

                if current_frame_idx == target_frame_idx:
                    encoder.stdin.write(raw_frame)
                    extracted_count += 1
                    target_position += 1

                current_frame_idx += 1

        except BrokenPipeError as error:
            raise RuntimeError(
                "FFmpeg encoder đóng pipe khi đang ghi keyframe."
            ) from error

        finally:
            encoder.stdin.close()
            decoder.stdout.close()

        # Đã lấy được frame cuối cần thiết nên không cần decode
        # phần video còn lại.
        if target_position == len(target_indices):
            _stop_process(decoder)
        else:
            decoder.wait()

        encoder_return_code = encoder.wait()

        decoder_error = decoder.stderr.read().decode(
            errors="replace"
        )
        encoder_error = encoder.stderr.read().decode(
            errors="replace"
        )

        decoder.stderr.close()
        encoder.stderr.close()

        if encoder_return_code != 0:
            raise RuntimeError(
                f"FFmpeg encoder thất bại với {video_path}:\n"
                f"{encoder_error.strip()}"
            )

        if extracted_count != len(target_indices):
            missing_indices = target_indices[target_position:]

            raise RuntimeError(
                f"Không đọc đủ keyframe từ {video_path}.\n"
                f"Các frame còn thiếu: {missing_indices[:20]}\n"
                f"FFmpeg decoder: {decoder_error.strip()}"
            )

        extracted_files = sorted(
            temp_dir.glob("selected_*.jpg")
        )

        if len(extracted_files) != len(target_indices):
            raise RuntimeError(
                f"Yêu cầu {len(target_indices)} ảnh nhưng "
                f"FFmpeg chỉ tạo {len(extracted_files)} ảnh."
            )

        # Đổi từ tên tuần tự sang đúng keyframe_id/image_path.
        for source, frame_idx in zip(
            extracted_files,
            target_indices,
            strict=True,
        ):
            for keyframe in by_frame_idx[frame_idx]:
                destination = output_root / keyframe.image_path

                if destination.exists():
                    destination.unlink()

                shutil.copy2(source, destination)

    missing_outputs = [
        frame.image_path
        for frame in frames
        if not (output_root / frame.image_path).is_file()
    ]

    if missing_outputs:
        raise RuntimeError(
            "Một số keyframe chưa được tạo:\n"
            + "\n".join(missing_outputs[:20])
        )
