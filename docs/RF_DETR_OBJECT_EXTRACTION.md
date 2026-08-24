# RF-DETR object extraction

Production object detection uses the canonical
`mvr.enrichment.objects.rfdetr.RFDETRDetector`: `RFDETRLarge` from
`rfdetr==1.9.3`, the released `rf-detr-large-2026.pth` checkpoint, 704 x 704
input, CUDA batch 1, and the existing confidence threshold 0.4. The older
`rf-detr-large.pth` release is legacy and superseded; it is not a runtime
candidate.

The released detector emits the MS COCO 2017 vocabulary (80 names using
sparse COCO category IDs in 90 model slots). It does not emit OpenImages V4
labels and does not run NMS.

## Installation and checkpoint

Install the exact production package:

```powershell
py -3.12 -m pip install -e ".[objects]"
```

Download the official checkpoint once through an approved provisioning path,
then verify it against `configs/rfdetr_model.lock.json`. The required identity
is:

```text
filename: rf-detr-large-2026.pth
bytes:    135954129
sha256:   0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38
```

Runtime downloads are disabled. Supply the verified local path with
`--detector-checkpoint` or `MVR_RFDETR_CHECKPOINT`. A missing, renamed,
wrong-size, or wrong-hash checkpoint fails closed; no model fallback occurs.

## Example

```powershell
python scripts/run_feature_extraction.py `
  --mapping-csv <canonical-keyframe-mapping.csv> `
  --keyframe-root <canonical-keyframes> `
  --output <feature-root> `
  --embedding-providers clip `
  --detector-model rfdetr_large `
  --detector-checkpoint <local-cache>/rf-detr-large-2026.pth `
  --detector-threshold 0.4 `
  --device cuda
```

`--detector-model` accepts only the production `rfdetr_large` identity. Faster
R-CNN and YOLO remain separate reference/legacy implementations and are never
automatic fallbacks. Pass `--skip-objects` to run only embedding providers.

## Output contracts

Task05A-compatible envelopes remain under:

```text
<feature-root>/objects/
|-- <video_id>/<keyframe_id>.json
|-- shot_summary.json
`-- manifest.json
```

Every envelope preserves `keyframe_id`, `video_id`, and both `frame_id` and
`frame_idx`, where `frame_id == frame_idx` from the canonical mapping. Its
`objects` list retains `score`/`bbox` for existing consumers and also includes
the canonical flat fields `confidence,x1,y1,x2,y2` plus the identity fields.

DATA-02 staging is written separately so Task05A never mistakes flat rows for
additional keyframe artifacts:

```text
<feature-root>/objects_staging/
|-- <video_id>/<keyframe_id>.json  # list of detection rows
`-- manifest.json
```

Consolidate this directory with schema `objects`. Object deduplication uses
`video_id,keyframe_id,class_id,x1,y1,x2,y2`, preserving multiple objects in
one frame and repeated keyframe IDs across different videos.

The pin and real runtime evidence are recorded in
`docs/reviews/RFDETR_PRODUCTION_PIN_REPORT.md`. Full Batch1 extraction was not
run as part of the pin task.
