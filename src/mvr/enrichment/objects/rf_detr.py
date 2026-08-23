from pathlib import Path
from typing import Any, Union

from PIL import Image

_VARIANTS = {
    "rfdetr_nano": "RFDETRNano",
    "rfdetr_small": "RFDETRSmall",
    "rfdetr_medium": "RFDETRMedium",
    "rfdetr_large": "RFDETRLarge",
}


class RfDetr:
    """Live RF-DETR object detector (COCO-pretrained). Drop-in replacement for
    ``FasterRCNN``: same ``detect()`` contract, so ``extract_objects()``'s
    on-disk JSON structure is unchanged.

    RF-DETR is a set-prediction (DETR-style) detector -- it does not run NMS,
    unlike torchvision's Faster R-CNN. ``nms_threshold`` is accepted for
    interface parity but unused.
    """

    def __init__(
        self,
        model_name: str = "rfdetr_medium",
        device: str = "auto",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> None:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr is missing; install the project's objects extra"
            ) from exc

        if model_name not in _VARIANTS:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.model_family = "RF-DETR"
        self.nms_enabled = False
        self.nms_mode = "none"
        self.nms_implementation = "nms_free_set_prediction"

        variant_cls = getattr(rfdetr, _VARIANTS[model_name])
        # device="auto" is left unpassed so RF-DETR's own NVML-backed
        # accelerator detection picks cuda/mps/cpu (more robust than a naive
        # torch.cuda.is_available() check -- see rfdetr.config._detect_device).
        self.model = variant_cls() if device == "auto" else variant_cls(device=device)
        self.device = self.model.model_config.device
        self.categories = self.model.class_names

    def detect(self, image_path: Union[Path, str]) -> list[dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not open image {image_path}: {e}")

        detections = self.model.predict(image, threshold=self.score_threshold)
        labels = detections.data["class_name"]

        results = []
        for i in range(len(detections)):
            results.append({
                "label": str(labels[i]),
                "class_id": int(detections.class_id[i]),
                "score": round(float(detections.confidence[i]), 4),
                "bbox": [round(float(c), 2) for c in detections.xyxy[i].tolist()],
            })
        return results
