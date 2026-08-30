"""MetaCLIP-2 image encoder backed by the locked Transformers checkpoint."""
from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .provider_specs import ProviderSpec


class MetaClip2DependencyError(RuntimeError):
    """Raised when the pinned MetaCLIP-2 runtime cannot be loaded."""


class MetaClip2ImageEncoder:
    def __init__(
        self,
        *,
        spec: ProviderSpec,
        model_location: str | Path | None = None,
        device: str = "auto",
        allow_download: bool = False,
        runtime_loader: Callable[..., tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        if spec.model_id != "metaclip2":
            raise ValueError(
                f"MetaClip2ImageEncoder requires metaclip2 spec, got {spec.model_id}"
            )
        self.spec = spec
        self.model_location = str(model_location or spec.checkpoint_or_pretrained)
        self.device_request = device
        self.allow_download = allow_download
        self._runtime_loader = runtime_loader or self._load_official_runtime
        self._runtime: tuple[Any, Any, Any, str] | None = None

    @staticmethod
    def _load_official_runtime(
        spec: ProviderSpec,
        model_location: str,
        device: str,
        allow_download: bool,
    ):
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise MetaClip2DependencyError(
                "MetaCLIP-2 requires the optional 'metaclip2' dependencies. "
                "Install them in an isolated environment."
            ) from exc

        local_path = Path(model_location)
        revision = None if local_path.exists() else spec.model_revision
        load_kwargs = {
            "revision": revision,
            "local_files_only": not allow_download,
        }
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                model_location, **load_kwargs
            )
            # The checkpoint's own transformersInfo pins auto_model to this
            # class, not the generic AutoModel; AutoModel fails to resolve
            # the metaclip_2 architecture and raises a misleading "pytorch_model.bin"
            # OSError regardless of allow_download.
            model = transformers.AutoModelForZeroShotImageClassification.from_pretrained(
                model_location, **load_kwargs
            )
        except (OSError, ValueError) as exc:
            mode = "local cache/path" if not allow_download else "pinned repository"
            raise MetaClip2DependencyError(
                f"cannot load locked MetaCLIP-2 from {mode}: {model_location}; "
                "provide the pinned cache or explicitly authorize downloads"
            ) from exc
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval().to(resolved_device)
        return torch, model, processor, resolved_device

    def _get_runtime(self):
        if self._runtime is None:
            self._runtime = self._runtime_loader(
                self.spec,
                self.model_location,
                self.device_request,
                self.allow_download,
            )
        return self._runtime

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        torch, model, processor, device = self._get_runtime()
        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise MetaClip2DependencyError("Pillow is required for MetaCLIP-2") from exc
        images = []
        for path in image_paths:
            with image_module.open(path) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        inference_context = getattr(torch, "inference_mode", torch.no_grad)
        with inference_context():
            features = model.get_image_features(**inputs)
        # This transformers version's get_image_features returns the full
        # BaseModelOutputWithPooling (projected embedding in pooler_output),
        # not a bare tensor as its own docstring example implies.
        vector = getattr(features, "pooler_output", features)
        return np.ascontiguousarray(vector.detach().cpu().numpy(), dtype=np.float32)
