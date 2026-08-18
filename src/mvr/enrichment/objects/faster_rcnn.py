from pathlib import Path
from typing import Any
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

class FasterRCNN:
    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn", device: str = "auto", score_threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device in ("auto", "cuda") else "cpu")
        self.score_threshold = score_threshold
        self.model_name = model_name
        
        # Load the model with pretrained weights
        if model_name == "fasterrcnn_resnet50_fpn":
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.categories = weights.meta["categories"]
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image_path: Path | str) -> list[dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not open image {image_path}: {e}")

        img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        predictions = self.model(img_tensor)[0]
        
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        results = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.score_threshold:
                # Convert back to standard native Python types for JSON serialization
                results.append({
                    "label": self.categories[int(label)],
                    "class_id": int(label),
                    "score": round(float(score), 4),
                    "bbox": [round(float(c), 2) for c in box.tolist()]
                })
        return results
