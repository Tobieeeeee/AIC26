"""Object detection backends for the MVR competition system.

Production default: RF-DETR (see rfdetr.py)
Legacy/experimental: Faster R-CNN (see faster_rcnn.py)
Organizer precomputed: AIC objects loader (see aic_objects.py)
"""

PRODUCTION_BACKEND = "rfdetr"

AVAILABLE_BACKENDS = {
    "rfdetr": "mvr.enrichment.objects.rfdetr.RFDETRDetector",
    "faster_rcnn": "mvr.enrichment.objects.faster_rcnn.FasterRCNN",
}

LEGACY_BACKENDS = frozenset({"faster_rcnn"})
