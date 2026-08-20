"""Perception Encoder image adapter for the locked PE Core checkpoint."""
from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .provider_specs import ProviderSpec


class PerceptionEncoderDependencyError(RuntimeError):
    """Raised when the official PE runtime or checkpoint is unavailable."""


class PerceptionEncoderImageEncoder:
    def __init__(
        self,
        *,
        spec: ProviderSpec,
        checkpoint_path: Path,
        device: str = "auto",
        runtime_loader: Callable[..., tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        if spec.model_id != "perception_encoder":
            raise ValueError(
                "PerceptionEncoderImageEncoder requires perception_encoder spec, "
                f"got {spec.model_id}"
            )
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.device_request = device
        self._runtime_loader = runtime_loader or self._load_official_runtime
        self._runtime: tuple[Any, Any, Any, str] | None = None

    @staticmethod
    def _load_official_runtime(spec: ProviderSpec, checkpoint: Path, device: str):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Perception Encoder checkpoint not found: {checkpoint}")
        try:
            torch = importlib.import_module("torch")
            pe = importlib.import_module("core.vision_encoder.pe")
            transforms = importlib.import_module("core.vision_encoder.transforms")
        except ImportError as exc:
            raise PerceptionEncoderDependencyError(
                "Perception Encoder runtime is missing. Install the pinned optional "
                "dependencies and place the locked perception_models source on PYTHONPATH."
            ) from exc
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = pe.CLIP.from_config(
                "PE-Core-L14-336",
                pretrained=True,
                checkpoint_path=str(checkpoint),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise PerceptionEncoderDependencyError(
                "cannot load the locked PE-Core-L14-336 checkpoint; verify its "
                "revision and SHA256"
            ) from exc
        model.eval().to(resolved_device)
        preprocess = transforms.get_image_transform(model.image_size)
        return torch, model, preprocess, resolved_device

    def _get_runtime(self):
        if self._runtime is None:
            self._runtime = self._runtime_loader(
                self.spec, self.checkpoint_path, self.device_request
            )
        return self._runtime

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        torch, model, preprocess, device = self._get_runtime()
        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise PerceptionEncoderDependencyError(
                "Pillow is required for Perception Encoder images"
            ) from exc
        tensors = []
        for path in image_paths:
            with image_module.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        inference_context = getattr(torch, "inference_mode", torch.no_grad)
        with inference_context():
            features = model.encode_image(batch, normalize=True)
        return np.ascontiguousarray(features.detach().cpu().numpy(), dtype=np.float32)
