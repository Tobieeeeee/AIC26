"""Object detection backends for the MVR competition system.

Production default: pinned RF-DETR Large (see rfdetr.py)
Legacy/reference only: Faster R-CNN and YOLO
Organizer precomputed: AIC objects loader (see aic_objects.py)
"""

PRODUCTION_BACKEND = "rfdetr"

AVAILABLE_BACKENDS = {
    "rfdetr": "mvr.enrichment.objects.rfdetr.RFDETRDetector",
}

LEGACY_BACKENDS = frozenset({"faster_rcnn", "yolov8"})
