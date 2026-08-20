"""BEiT-3 image encoder using the locked Microsoft UniLM contract."""
from __future__ import annotations

import importlib
import math
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import verify_file_sha256
from .provider_specs import ProviderSpec


class Beit3DependencyError(RuntimeError):
    """Raised when the isolated BEiT-3 runtime is unavailable."""


def _install_torch_six_inf_compat() -> None:
    """Provide the sole legacy torch._six symbol used by pinned UniLM code."""
    try:
        importlib.import_module("torch._six")
    except ImportError:
        compatibility_module = types.ModuleType("torch._six")
        compatibility_module.inf = math.inf
        sys.modules["torch._six"] = compatibility_module


class Beit3ImageEncoder:
    def __init__(
        self,
        *,
        spec: ProviderSpec,
        checkpoint_path: Path,
        device: str = "auto",
        runtime_loader: Callable[[ProviderSpec, Path, str], tuple[Any, Any, Any]] | None = None,
    ) -> None:
        if spec.model_id != "beit3":
            raise ValueError(f"Beit3ImageEncoder requires beit3 spec, got {spec.model_id}")
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.device_request = device
        self._runtime_loader = runtime_loader or self._load_official_runtime
        self._runtime: tuple[Any, Any, Any] | None = None

    @staticmethod
    def _load_official_runtime(spec: ProviderSpec, checkpoint: Path, device: str):
        verify_file_sha256(
            checkpoint, spec.checkpoint_sha256, "BEiT-3 checkpoint"
        )
        try:
            torch = importlib.import_module("torch")
            timm = importlib.import_module("timm")
            _install_torch_six_inf_compat()
            importlib.import_module("modeling_finetune")
            transforms = importlib.import_module("torchvision.transforms")
        except ImportError as exc:
            raise Beit3DependencyError(
                "BEiT-3 runtime is missing. Use the isolated versions in "
                "configs/embedding_providers.json, place microsoft/unilm/beit3 "
                "on PYTHONPATH, and provide the locked checkpoint explicitly."
            ) from exc

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model("beit3_base_patch16_224_retrieval", pretrained=False)
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload.get("module", payload))
        missing, unexpected = model.load_state_dict(state, strict=False)
        critical_missing = [
            key for key in missing if key.startswith(("beit3.", "vision_head.", "language_head."))
        ]
        if critical_missing or unexpected:
            raise RuntimeError(
                "BEiT-3 checkpoint is incompatible with the locked retrieval model: "
                f"missing={critical_missing[:5]}, unexpected={unexpected[:5]}"
            )
        model.eval().to(resolved_device)
        preprocess = transforms.Compose(
            [
                transforms.Resize(
                    tuple(spec.image_preprocessing["size"]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=spec.image_preprocessing["mean"],
                    std=spec.image_preprocessing["std"],
                ),
            ]
        )
        return torch, model, (preprocess, resolved_device)

    def _get_runtime(self):
        if self._runtime is None:
            self._runtime = self._runtime_loader(
                self.spec, self.checkpoint_path, self.device_request
            )
        return self._runtime

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        torch, model, runtime = self._get_runtime()
        preprocess, device = runtime
        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise Beit3DependencyError("Pillow is required for BEiT-3 images") from exc
        tensors = []
        for path in image_paths:
            with image_module.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        inference_context = getattr(torch, "inference_mode", torch.no_grad)
        with inference_context():
            features, _ = model(image=batch, only_infer=True)
        return np.ascontiguousarray(features.detach().cpu().numpy(), dtype=np.float32)
