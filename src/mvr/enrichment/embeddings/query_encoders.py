"""Lazy, provider-matched online text query encoders."""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .artifacts import verify_file_sha256
from .beit3 import Beit3ImageEncoder
from .metaclip2 import MetaClip2ImageEncoder
from .perception_encoder import PerceptionEncoderImageEncoder
from .provider_specs import ProviderSpec, ProviderSpecRegistry


class QueryEncoderError(RuntimeError):
    """Base error for provider query encoding."""


class QueryProviderUnavailableError(QueryEncoderError):
    """Raised when the requested provider cannot be loaded."""


class QueryRoutingError(QueryEncoderError):
    """Raised before a query can cross provider embedding spaces."""


@dataclass(frozen=True)
class QueryEmbedding:
    model_id: str
    vector: np.ndarray
    dimension: int
    dtype: str
    normalized: bool


@dataclass(frozen=True)
class ProviderQueryEncoderConfig:
    device: str = "auto"
    beit3_checkpoint: Path | None = None
    beit3_tokenizer: Path | None = None
    metaclip2_model: str | Path | None = None
    metaclip2_allow_download: bool = False
    perception_encoder_checkpoint: Path | None = None
    disabled_providers: frozenset[str] = frozenset()


class TextEncoderAdapter(Protocol):
    model_id: str

    def encode_text(self, text: str) -> np.ndarray: ...


def _inference(torch: Any):
    return getattr(torch, "inference_mode", torch.no_grad)()


class Beit3TextEncoder:
    model_id = "beit3"

    def __init__(
        self,
        *,
        spec: ProviderSpec,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device: str = "auto",
        runtime_loader: Callable[..., tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device
        self._loader = runtime_loader or self._load_runtime
        self._runtime: tuple[Any, Any, Any, str] | None = None

    @staticmethod
    def _load_runtime(spec, checkpoint_path, tokenizer_path, device):
        verify_file_sha256(
            tokenizer_path, spec.tokenizer_sha256, "BEiT-3 tokenizer"
        )
        image_encoder = Beit3ImageEncoder(
            spec=spec, checkpoint_path=checkpoint_path, device=device
        )
        torch, model, runtime = image_encoder._get_runtime()
        _, resolved_device = runtime
        try:
            transformers = importlib.import_module("transformers")
            tokenizer = transformers.XLMRobertaTokenizer(str(tokenizer_path))
        except (ImportError, OSError, ValueError) as exc:
            raise QueryProviderUnavailableError(
                "cannot load the locked BEiT-3 beit3.spm tokenizer"
            ) from exc
        return torch, model, tokenizer, resolved_device

    def _get_runtime(self):
        if self._runtime is None:
            self._runtime = self._loader(
                self.spec,
                self.checkpoint_path,
                self.tokenizer_path,
                self.device,
            )
        return self._runtime

    def encode_text(self, text: str) -> np.ndarray:
        torch, model, tokenizer, device = self._get_runtime()
        max_length = int(self.spec.tokenizer["max_length"])
        token_ids = tokenizer.encode(text, add_special_tokens=False)[: max_length - 2]
        token_ids = [tokenizer.bos_token_id, *token_ids, tokenizer.eos_token_id]
        padding = max_length - len(token_ids)
        language_tokens = torch.tensor(
            [token_ids + [tokenizer.pad_token_id] * padding], device=device
        )
        padding_mask = torch.tensor(
            [[0] * len(token_ids) + [1] * padding], device=device
        )
        with _inference(torch):
            _, features = model(
                text_description=language_tokens,
                padding_mask=padding_mask,
                only_infer=True,
            )
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)


class MetaClip2TextEncoder:
    model_id = "metaclip2"

    def __init__(
        self,
        *,
        spec: ProviderSpec,
        model_location: str | Path | None = None,
        device: str = "auto",
        allow_download: bool = False,
        runtime_loader: Callable[..., tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        self.spec = spec
        self.image_encoder = MetaClip2ImageEncoder(
            spec=spec,
            model_location=model_location,
            device=device,
            allow_download=allow_download,
            runtime_loader=runtime_loader,
        )

    def encode_text(self, text: str) -> np.ndarray:
        torch, model, processor, device = self.image_encoder._get_runtime()
        inputs = processor(
            text=[text],
            padding=True,
            truncation=True,
            max_length=int(self.spec.tokenizer["max_length"]),
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with _inference(torch):
            features = model.get_text_features(**inputs)
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)


class PerceptionEncoderTextEncoder:
    model_id = "perception_encoder"

    def __init__(
        self,
        *,
        spec: ProviderSpec,
        checkpoint_path: Path,
        device: str = "auto",
        runtime_loader: Callable[..., tuple[Any, Any, Any, str]] | None = None,
    ) -> None:
        self.spec = spec
        self.image_encoder = PerceptionEncoderImageEncoder(
            spec=spec,
            checkpoint_path=checkpoint_path,
            device=device,
            runtime_loader=runtime_loader,
        )
        self._tokenizer: Any | None = None

    def encode_text(self, text: str) -> np.ndarray:
        torch, model, _, device = self.image_encoder._get_runtime()
        if self._tokenizer is None:
            try:
                transforms = importlib.import_module("core.vision_encoder.transforms")
                self._tokenizer = transforms.get_text_tokenizer(model.context_length)
            except (ImportError, AttributeError, ValueError) as exc:
                raise QueryProviderUnavailableError(
                    "cannot load the official PE Core text tokenizer"
                ) from exc
        tokens = self._tokenizer([text]).to(device)
        with _inference(torch):
            features = model.encode_text(tokens, normalize=True)
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)


class QueryEncoderRegistry:
    """Resolve and cache exactly one adapter instance per provider."""

    def __init__(
        self,
        specs: ProviderSpecRegistry,
        config: ProviderQueryEncoderConfig = ProviderQueryEncoderConfig(),
        factories: Mapping[str, Callable[[ProviderSpec], TextEncoderAdapter]] | None = None,
    ) -> None:
        self.specs = specs
        self.config = config
        self._cache: dict[str, TextEncoderAdapter] = {}
        self._factories = dict(factories or self._default_factories())

    def _default_factories(self) -> dict[str, Callable[[ProviderSpec], TextEncoderAdapter]]:
        def beit3(spec: ProviderSpec) -> TextEncoderAdapter:
            if self.config.beit3_checkpoint is None or self.config.beit3_tokenizer is None:
                raise QueryProviderUnavailableError(
                    "BEiT-3 requires explicit local checkpoint and beit3.spm paths"
                )
            return Beit3TextEncoder(
                spec=spec,
                checkpoint_path=self.config.beit3_checkpoint,
                tokenizer_path=self.config.beit3_tokenizer,
                device=self.config.device,
            )

        def metaclip2(spec: ProviderSpec) -> TextEncoderAdapter:
            return MetaClip2TextEncoder(
                spec=spec,
                model_location=self.config.metaclip2_model,
                device=self.config.device,
                allow_download=self.config.metaclip2_allow_download,
            )

        def perception_encoder(spec: ProviderSpec) -> TextEncoderAdapter:
            if self.config.perception_encoder_checkpoint is None:
                raise QueryProviderUnavailableError(
                    "Perception Encoder requires an explicit local checkpoint path"
                )
            return PerceptionEncoderTextEncoder(
                spec=spec,
                checkpoint_path=self.config.perception_encoder_checkpoint,
                device=self.config.device,
            )

        return {
            "beit3": beit3,
            "metaclip2": metaclip2,
            "perception_encoder": perception_encoder,
        }

    def _adapter(self, model_id: str) -> TextEncoderAdapter:
        if model_id in self.config.disabled_providers:
            raise QueryProviderUnavailableError(f"query provider is disabled: {model_id}")
        try:
            spec = self.specs.get(model_id)
        except (KeyError, ValueError) as exc:
            raise QueryProviderUnavailableError(f"unknown query provider: {model_id}") from exc
        if model_id not in self._cache:
            try:
                factory = self._factories[model_id]
                adapter = factory(spec)
            except KeyError as exc:
                raise QueryProviderUnavailableError(
                    f"no text encoder adapter is registered for {model_id}"
                ) from exc
            except QueryEncoderError:
                raise
            except Exception as exc:
                raise QueryProviderUnavailableError(
                    f"cannot initialize query provider {model_id}: {type(exc).__name__}"
                ) from exc
            if adapter.model_id != model_id:
                raise QueryRoutingError(
                    f"factory returned {adapter.model_id} for requested {model_id}"
                )
            self._cache[model_id] = adapter
        return self._cache[model_id]

    def encode_query(
        self,
        model_id: str,
        text: str,
        *,
        index_contract: Any | None = None,
    ) -> QueryEmbedding:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        try:
            spec = self.specs.get(model_id)
        except (KeyError, ValueError) as exc:
            raise QueryProviderUnavailableError(
                f"unknown query provider: {model_id}"
            ) from exc
        if index_contract is not None:
            if index_contract.model_id != model_id:
                raise QueryRoutingError(
                    f"query provider {model_id} cannot search {index_contract.model_id} index"
                )
            if index_contract.dimension != spec.embedding_dimension:
                raise QueryRoutingError(
                    f"query/index dimension mismatch for {model_id}: "
                    f"{spec.embedding_dimension} != {index_contract.dimension}"
                )
        try:
            raw = self._adapter(model_id).encode_text(text.strip())
        except QueryEncoderError:
            raise
        except Exception as exc:
            raise QueryProviderUnavailableError(
                f"query encoding failed for {model_id}: {type(exc).__name__}"
            ) from exc
        vector = np.asarray(raw)
        if vector.shape == (1, spec.embedding_dimension):
            vector = vector[0]
        if vector.shape != (spec.embedding_dimension,):
            raise QueryEncoderError(
                f"{model_id} query dimension {vector.shape} != ({spec.embedding_dimension},)"
            )
        if not np.issubdtype(vector.dtype, np.number) or not np.isfinite(vector).all():
            raise QueryEncoderError(f"{model_id} query vector must be finite numeric data")
        norm = float(np.linalg.norm(vector))
        if spec.normalized:
            if norm <= 0 or not np.isfinite(norm):
                raise QueryEncoderError(f"{model_id} query vector has invalid norm")
            vector = vector / norm
        vector = np.ascontiguousarray(vector, dtype=np.dtype(spec.dtype))
        return QueryEmbedding(
            model_id=model_id,
            vector=vector,
            dimension=spec.embedding_dimension,
            dtype=spec.dtype,
            normalized=spec.normalized,
        )
