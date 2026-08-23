# RF-DETR object extraction

Object detection for keyframes uses `RfDetr`
(`src/mvr/enrichment/objects/rf_detr.py`), a COCO-pretrained RF-DETR model.
It replaced the previous live `FasterRCNN` detector (also COCO-pretrained,
not OpenImages V4) as of 2026-08-23; `faster_rcnn.py` was removed from the
repo (still recoverable from git history if ever needed).

Prerequisites:

1. Install the `objects` dependency group, which now includes `rfdetr`
   alongside the existing `ultralytics` (YOLO) dependency:

   ```powershell
   py -3.12 -m pip install -e ".[objects]"
   ```

2. No manual checkpoint download is required. Unlike the `beit3` /
   `metaclip2` / `perception-encoder` providers (which use a locked,
   SHA256-pinned local checkpoint — see `docs/BEIT3_OFFLINE_EXTRACTION.md`),
   RF-DETR downloads its own official Roboflow-hosted COCO weights
   automatically on first construction of the model. Check available disk
   space before the first run.

Example offline object extraction:

```powershell
python scripts/run_feature_extraction.py `
  --mapping-csv <canonical-keyframe-mapping.csv> `
  --keyframe-root <canonical-keyframes> `
  --output <feature-root> `
  --embedding-providers clip `
  --detector-model rfdetr_medium `
  --detector-threshold 0.5 `
  --device cuda
```

`--detector-model` accepts `rfdetr_nano`, `rfdetr_small`, `rfdetr_medium`
(default), or `rfdetr_large` — larger variants trade latency for accuracy.
Pass `--skip-objects` to run only the embedding providers.

RF-DETR is a set-prediction (DETR-style) detector: it does not run NMS.
`--detector-nms-threshold` is accepted for interface parity with the old
Faster R-CNN flags but has no effect.

Output layout (unchanged from the previous Faster R-CNN detector):

```text
<feature-root>/objects/
|-- <video_id>/<keyframe_id>.json   # one file per keyframe
|-- shot_summary.json
`-- manifest.json
```

Each per-keyframe JSON has the shape
`{keyframe_id, video_id, shot_id, frame_idx, timestamp, image_path,
person_count, car_count, objects: [{label, class_id, score, bbox}]}`,
with `bbox` as pixel-space `[x1, y1, x2, y2]` in the original image. This
contract is what other pipeline code and tests rely on; it did not change
when the detector backend changed. `manifest.json` now reports the
detector's own `model_family`/`nms_*` fields instead of hardcoded
Faster R-CNN values.
