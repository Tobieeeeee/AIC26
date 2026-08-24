"""Compatibility import for the canonical pinned RF-DETR adapter.

New code must import :mod:`mvr.enrichment.objects.rfdetr` directly. This
module contains no alternate loader, variant defaults, or download behavior.
"""

from .rfdetr import RFDETRConfig, RFDETRDetector

RfDetr = RFDETRDetector

__all__ = ["RFDETRConfig", "RFDETRDetector", "RfDetr"]
