# L21_V001 Keyframe-Selection Ground Truth v0.1

## Video facts

- Video: `L21_V001.mp4`
- Resolution: 1280x720
- FPS: 30
- Video-stream frames: 37849
- Video-stream duration: 1261.633333 s
- Annotated visual queries: 32

## What this ground truth means

Each query maps to a **relevant temporal segment**, not to one "correct keyframe".

A selector covers a query when it keeps at least one keyframe whose:

```text
video_id == L21_V001
AND
start_frame <= frame_idx <= end_frame
```

This allows direct computation of:

```text
Selection Recall
= covered GT segments / total GT segments
```

before running FAISS or any retrieval model.

## Annotation basis

The video was manually inspected across the full ~21-minute duration using
5-second visual contact sheets. Scene-change timestamps were also inspected to
tighten many boundaries.

Queries intentionally describe **visible content only**. No claim is made about
audio-only facts.

The files use half-open temporal intervals `[start_time, end_time)` and inclusive
frame intervals `[start_frame, end_frame]`.

## Important limitation

This is **research ground truth v0.1**, generated for comparing keyframe
selection strategies. It is not official AIC/BTC ground truth.

Rows marked `medium` should be manually reviewed if this dataset is used for
publication-grade benchmarking. A ±1 s tolerance is recorded for high-confidence
rows and ±2 s for medium-confidence rows.

## Suggested first benchmark

Compare on the same frozen shot boundaries:

- midpoint-only
- 12s/8s
- 15s/5s
- adaptive PE
- adaptive MetaCLIP

Report:

- Selection Recall
- number of keyframes
- Selection Recall per 1,000 keyframes
- storage / compute

Only after that should you run Recall@K / MRR using the retrieval pipeline.
