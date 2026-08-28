"""Measure real OCR throughput on this machine, then extrapolate to the corpus.

Fifteen minutes here decides whether OCR over the whole corpus is an
afternoon or a week, so it is worth measuring rather than assuming. The
script runs the same call the pipeline runs -- `easyocr.Reader(["vi","en"],
gpu=...).readtext(...)` from `scene_text.py` -- on a random sample of real
keyframes, so the number predicts the real job rather than a toy.

Three things are timed separately, because they scale differently:

**Model load.** `process_video_ocr` constructs a fresh `easyocr.Reader` on
every call, and `process_dataset_ocr` calls it once per video. That is 873
model loads over the corpus, and at 8 seconds each it is two hours of
nothing but loading. Timed on its own so the report can say how much of the
total is this and not OCR.

**Warm-up.** The first inference pays for CUDA context creation and kernel
autotuning. Folding it into the average makes a 200-image sample look far
slower than a 166,550-image job actually is.

**Steady state.** What everything after the warm-up costs. This is the
number the extrapolation uses, reported as a median so one pathological
image cannot move it.

Example::

    python scripts/measure_ocr_throughput.py --dataset-root <your-dataset-root>

Reads only; writes nothing except an optional JSON report.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

#: The corpus this extrapolates to. Overridable because a subset run should
#: extrapolate to the subset, not silently to the whole thing.
DEFAULT_TOTAL_KEYFRAMES = 166_550
DEFAULT_TOTAL_VIDEOS = 873

#: `generate_sampling_frames` defaults to [-1, 0, 1]: three OCR calls per
#: keyframe, each on a frame decoded out of the source video.
DEFAULT_TEMPORAL_OFFSETS = 3


def find_keyframe_images(dataset_root: Path, count: int, seed: int) -> list[Path]:
    """Sample across videos, not within one.

    OCR cost tracks image content and resolution, and a single video is far
    more uniform than the corpus. Sampling one video would measure that
    video.
    """
    roots = sorted(dataset_root.glob("Keyframes_*"))
    if not roots:
        # Normalized layout: <root>/keyframes/<video_id>/001.jpg
        roots = [dataset_root] if (dataset_root / "keyframes").is_dir() else []
    if not roots:
        raise SystemExit(
            f"no keyframe directories under {dataset_root}; expected "
            f"<root>/Keyframes_L21/keyframes/<video_id>/*.jpg or "
            f"<root>/keyframes/<video_id>/*.jpg"
        )

    video_dirs: list[Path] = []
    for root in roots:
        base = root / "keyframes"
        if base.is_dir():
            video_dirs.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    if not video_dirs:
        raise SystemExit(f"no video directories found under {dataset_root}")

    rng = random.Random(seed)
    rng.shuffle(video_dirs)

    images: list[Path] = []
    for video_dir in video_dirs:
        found = sorted(video_dir.glob("*.jpg"))
        if found:
            images.append(rng.choice(found))
        if len(images) >= count:
            break

    # Fewer videos than requested images: top up from the videos we have.
    index = 0
    while len(images) < count and video_dirs:
        found = sorted(video_dirs[index % len(video_dirs)].glob("*.jpg"))
        if found:
            images.append(rng.choice(found))
        index += 1
        if index > count * 4:
            break

    if not images:
        raise SystemExit(f"found video directories but no .jpg under {dataset_root}")
    return images[:count]


def describe_device() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False}

    info = {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available())}
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    return info


def _hms(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} phút"
    return f"{seconds / 3600:.1f} giờ"


def measure(images: list[Path], *, gpu: bool, warmup: int) -> dict:
    import numpy as np
    from PIL import Image

    try:
        import easyocr
    except ImportError as exc:
        raise SystemExit(
            "EasyOCR is missing; install the project's ocr extra"
        ) from exc

    print(f"Nạp model EasyOCR (gpu={gpu})...", file=sys.stderr)
    load_start = time.perf_counter()
    reader = easyocr.Reader(["vi", "en"], gpu=gpu)
    first_load_s = time.perf_counter() - load_start
    print(f"  lần đầu: {first_load_s:.1f}s", file=sys.stderr)

    # A second construction shows the per-video reload cost with the weights
    # already in the OS page cache -- the realistic figure for videos 2..873,
    # which is what `process_dataset_ocr` actually pays 872 times.
    reload_start = time.perf_counter()
    easyocr.Reader(["vi", "en"], gpu=gpu)
    reload_s = time.perf_counter() - reload_start
    print(f"  nạp lại : {reload_s:.1f}s", file=sys.stderr)

    def read(path: Path) -> int:
        image = np.asarray(Image.open(path).convert("RGB"))
        return len(reader.readtext(image))

    warmup = min(warmup, len(images))
    print(f"Khởi động {warmup} ảnh...", file=sys.stderr)
    warmup_start = time.perf_counter()
    for path in images[:warmup]:
        read(path)
    warmup_s = time.perf_counter() - warmup_start

    timed = images[warmup:]
    if not timed:
        raise SystemExit(
            f"chỉ có {len(images)} ảnh, không còn ảnh nào sau khi khởi động "
            f"{warmup}; tăng --count hoặc giảm --warmup"
        )

    print(f"Đo trên {len(timed)} ảnh...", file=sys.stderr)
    per_image: list[float] = []
    detections = 0
    for index, path in enumerate(timed, start=1):
        start = time.perf_counter()
        detections += read(path)
        per_image.append(time.perf_counter() - start)
        if index % 50 == 0:
            print(f"  {index}/{len(timed)}", file=sys.stderr)

    per_image.sort()
    return {
        "model_load_first_s": first_load_s,
        "model_reload_s": reload_s,
        "warmup_images": warmup,
        "warmup_total_s": warmup_s,
        "measured_images": len(per_image),
        "median_s": statistics.median(per_image),
        "mean_s": statistics.fmean(per_image),
        "p90_s": per_image[int(len(per_image) * 0.9) - 1],
        "min_s": per_image[0],
        "max_s": per_image[-1],
        "detections_total": detections,
    }


def render(result: dict, device: dict, *, keyframes: int, videos: int) -> str:
    median = result["median_s"]
    reload_total = result["model_reload_s"] * videos

    ocr_single = median * keyframes
    ocr_triple = median * keyframes * DEFAULT_TEMPORAL_OFFSETS

    lines = [
        "",
        "=" * 66,
        "T0 — TỐC ĐỘ OCR ĐO ĐƯỢC",
        "=" * 66,
        f"  Thiết bị      : {device.get('gpu') or 'CPU'}"
        + (f"  ({device['vram_gb']} GB VRAM)" if device.get("vram_gb") else ""),
        f"  torch         : {device.get('torch')}  CUDA={device.get('cuda')}",
        "",
        f"  Nạp model lần đầu : {result['model_load_first_s']:.1f}s",
        f"  Nạp lại mỗi video : {result['model_reload_s']:.1f}s",
        f"  Khởi động         : {result['warmup_total_s']:.1f}s cho "
        f"{result['warmup_images']} ảnh",
        "",
        f"  Đo trên {result['measured_images']} ảnh, "
        f"{result['detections_total']} vùng chữ:",
        f"    trung vị  {median * 1000:7.1f} ms/ảnh   "
        f"({1 / median:5.1f} ảnh/giây)",
        f"    trung bình{result['mean_s'] * 1000:7.1f} ms/ảnh",
        f"    p90       {result['p90_s'] * 1000:7.1f} ms/ảnh",
        f"    nhanh/chậm{result['min_s'] * 1000:7.1f} / "
        f"{result['max_s'] * 1000:.1f} ms",
        "",
        "-" * 66,
        f"SUY RA CHO {keyframes:,} KEYFRAME / {videos} VIDEO".replace(",", "."),
        "-" * 66,
        "",
        f"  offsets=[0]        OCR {_hms(ocr_single):>10}"
        f"  + nạp model {_hms(reload_total):>10}"
        f"  = {_hms(ocr_single + reload_total)}",
        f"  offsets=[-1,0,1]   OCR {_hms(ocr_triple):>10}"
        f"  + nạp model {_hms(reload_total):>10}"
        f"  = {_hms(ocr_triple + reload_total)}  + giải mã video",
        "",
    ]

    share = reload_total / (ocr_single + reload_total)
    if share > 0.15:
        lines += [
            f"  ⚠ Nạp model chiếm {share:.0%} tổng thời gian của phương án đầu.",
            "    `process_video_ocr` dựng một `easyocr.Reader` mới mỗi video",
            f"    (scene_text.py:316), nên trả giá đó {videos} lần. Dựng một lần",
            f"    rồi truyền vào sẽ tiết kiệm khoảng {_hms(reload_total)}.",
            "",
        ]

    if ocr_single + reload_total < 4 * 3600:
        verdict = "CHẠY ĐƯỢC TOÀN CORPUS. Dùng offsets=[0], không cần cắt tập con."
    elif ocr_single + reload_total < 12 * 3600:
        verdict = "CHẠY ĐƯỢC QUA ĐÊM với offsets=[0]. Không đủ thời gian cho [-1,0,1]."
    else:
        verdict = "QUÁ CHẬM cho toàn corpus. Cắt tập con, hoặc chạy nhiều GPU."
    lines += [f"  → {verdict}", "", "=" * 66, ""]

    lines += [
        "  Lưu ý: con số này chỉ tính OCR trên ảnh có sẵn. Phương án",
        "  offsets=[-1,0,1] còn phải giải mã ~500.000 frame từ 78 GB video,",
        "  chưa đo ở đây và thường là phần đắt hơn cả OCR.",
        "",
    ]
    return "\n".join(lines)


def _force_utf8_output() -> None:
    """Keep a Windows console from throwing away the whole measurement.

    The report is in Vietnamese and the default Windows code page is cp1252,
    which cannot encode it. Printing would raise `UnicodeEncodeError` on the
    LAST line of the script -- after the fifteen minutes of measuring -- and
    the numbers would be gone. `errors="replace"` degrades a character rather
    than losing the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - exotic stream
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--cpu", action="store_true", help="ép chạy CPU, để so với GPU"
    )
    parser.add_argument("--total-keyframes", type=int, default=DEFAULT_TOTAL_KEYFRAMES)
    parser.add_argument("--total-videos", type=int, default=DEFAULT_TOTAL_VIDEOS)
    parser.add_argument("--report", type=Path, help="ghi kết quả ra JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)

    device = describe_device()
    gpu = bool(device.get("cuda")) and not args.cpu
    if not gpu and not args.cpu:
        print(
            "CẢNH BÁO: không thấy CUDA, sẽ đo bằng CPU. Con số suy ra sẽ không\n"
            "         mô tả máy có GPU. Kiểm lại torch trước khi tin kết quả.\n",
            file=sys.stderr,
        )

    images = find_keyframe_images(args.dataset_root, args.count, args.seed)
    print(f"Chọn {len(images)} ảnh từ {args.dataset_root}", file=sys.stderr)

    result = measure(images, gpu=gpu, warmup=args.warmup)
    report = render(
        result, device, keyframes=args.total_keyframes, videos=args.total_videos
    )
    print(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "device": device,
                    "gpu_used": gpu,
                    "sample_size": len(images),
                    "timings": result,
                    "total_keyframes": args.total_keyframes,
                    "total_videos": args.total_videos,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Đã ghi {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
