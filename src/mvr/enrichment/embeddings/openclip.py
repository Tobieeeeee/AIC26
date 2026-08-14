"""OpenCLIP encoders adapted from the completed Lab 3 notebook."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


def _resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class OpenClipEncoder:
    """Encode images and text with Lab 3 preprocessing and L2 normalization."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str = "auto",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "OpenCLIP dependencies are missing; install with "
                "pip install -e \".[semantic]\""
            ) from exc

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = _resolve_device(torch, device)
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    def encode_images(
        self,
        image_paths: Iterable[Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        from PIL import Image

        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(self._preprocess(image.convert("RGB")))
            batch = self._torch.stack(tensors).to(self.device)
            with self._torch.no_grad():
                features = self._model.encode_image(batch)
                features = self._torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy())
        return np.ascontiguousarray(np.vstack(batches), dtype=np.float32)

    def encode_text(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        values = list(texts)
        if not values:
            raise ValueError("texts must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            tokens = self._tokenizer(values[start : start + batch_size]).to(self.device)
            with self._torch.no_grad():
                features = self._model.encode_text(tokens)
                features = self._torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy())
        return np.ascontiguousarray(np.vstack(batches), dtype=np.float32)
