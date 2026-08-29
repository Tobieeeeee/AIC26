import argparse
import csv
from pathlib import Path

def load_gt(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def load_mapping(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--mapping", required=True)
    args = p.parse_args()

    gt = load_gt(args.groundtruth)
    mapping = load_mapping(args.mapping)

    by_video = {}
    for r in mapping:
        try:
            frame = int(float(r["frame_idx"]))
        except Exception:
            continue
        by_video.setdefault(r["video_id"], []).append(frame)

    hits = []
    for r in gt:
        video = r["video_id"]
        lo, hi = int(r["start_frame"]), int(r["end_frame"])
        frames = by_video.get(video, [])
        covered = any(lo <= f <= hi for f in frames)
        hits.append((r["query_id"], covered))

    covered = sum(x[1] for x in hits)
    total = len(hits)
    print(f"covered={covered}")
    print(f"total={total}")
    print(f"selection_recall={covered/total if total else 0:.6f}")

    misses = [qid for qid, ok in hits if not ok]
    if misses:
        print("misses=" + ",".join(misses))

if __name__ == "__main__":
    main()
