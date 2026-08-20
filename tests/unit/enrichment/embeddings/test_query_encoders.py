from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.embeddings.query_encoders import (
    Beit3TextEncoder,
    MetaClip2TextEncoder,
    PerceptionEncoderTextEncoder,
    ProviderQueryEncoderConfig,
    QueryEncoderError,
    QueryEncoderRegistry,
    QueryProviderUnavailableError,
    QueryRoutingError,
)
from mvr.indexing.provider_indexes import ProviderIndexContract


SPECS = load_provider_specs(Path("configs/embedding_providers.json"))


class FakeAdapter:
    def __init__(self, model_id: str, dimension: int, *, fail: bool = False) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.fail = fail
        self.queries: list[str] = []

    def encode_text(self, text: str) -> np.ndarray:
        self.queries.append(text)
        if self.fail:
            raise RuntimeError("provider offline")
        vector = np.zeros(self.dimension, dtype=np.float64)
        vector[0] = 2.0
        return vector


def _registry(*, fail: str | None = None, disabled=frozenset()):
    created: dict[str, int] = {}
    adapters: dict[str, FakeAdapter] = {}

    def factory(model_id: str):
        def create(spec):
            created[model_id] = created.get(model_id, 0) + 1
            adapter = FakeAdapter(
                model_id, spec.embedding_dimension, fail=model_id == fail
            )
            adapters[model_id] = adapter
            return adapter

        return create

    registry = QueryEncoderRegistry(
        SPECS,
        ProviderQueryEncoderConfig(disabled_providers=disabled),
        factories={model_id: factory(model_id) for model_id in SPECS.model_ids},
    )
    return registry, created, adapters


def test_fasttest_query_produces_three_matching_normalized_vectors() -> None:
    registry, _, _ = _registry()
    for model_id in ("beit3", "metaclip2", "perception_encoder"):
        spec = SPECS.get(model_id)
        contract = ProviderIndexContract(
            model_id=model_id,
            collection_name=f"test__fasttest__{model_id}",
            dimension=spec.embedding_dimension,
            metric=spec.similarity_metric,
            normalized=spec.normalized,
            backend="fake",
        )
        result = registry.encode_query(model_id, "a red bus", index_contract=contract)
        assert result.model_id == model_id
        assert result.vector.shape == (spec.embedding_dimension,)
        assert result.dimension == spec.embedding_dimension
        assert result.dtype == "float32"
        assert result.vector.dtype == np.float32
        assert np.isclose(np.linalg.norm(result.vector), 1.0)


def test_adapter_is_cached_and_query_order_is_deterministic() -> None:
    registry, created, adapters = _registry()
    registry.encode_query("beit3", "first")
    registry.encode_query("beit3", "second")
    assert created == {"beit3": 1}
    assert adapters["beit3"].queries == ["first", "second"]


@pytest.mark.parametrize("text", ["", "  ", None])
def test_empty_query_policy(text) -> None:
    registry, _, _ = _registry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.encode_query("beit3", text)


def test_unknown_disabled_and_unavailable_providers_are_typed() -> None:
    registry, _, _ = _registry(disabled=frozenset({"beit3"}))
    with pytest.raises(QueryProviderUnavailableError, match="unknown"):
        registry.encode_query("unknown", "query")
    with pytest.raises(QueryProviderUnavailableError, match="disabled"):
        registry.encode_query("beit3", "query")
    failing, _, _ = _registry(fail="metaclip2")
    with pytest.raises(QueryProviderUnavailableError, match="metaclip2"):
        failing.encode_query("metaclip2", "query")
    assert failing.encode_query("beit3", "query").model_id == "beit3"


def test_cross_model_and_dimension_routes_are_rejected_before_encoding() -> None:
    registry, created, _ = _registry()
    wrong_model = ProviderIndexContract(
        "metaclip2", "test__fasttest__metaclip2", 512, "inner_product", True, "fake"
    )
    with pytest.raises(QueryRoutingError, match="cannot search"):
        registry.encode_query("beit3", "query", index_contract=wrong_model)
    wrong_dimension = ProviderIndexContract(
        "beit3", "test__fasttest__beit3", 999, "inner_product", True, "fake"
    )
    with pytest.raises(QueryRoutingError, match="dimension mismatch"):
        registry.encode_query("beit3", "query", index_contract=wrong_dimension)
    assert created == {}


def test_factory_cross_model_adapter_is_rejected() -> None:
    registry = QueryEncoderRegistry(
        SPECS,
        factories={"beit3": lambda spec: FakeAdapter("metaclip2", spec.embedding_dimension)},
    )
    with pytest.raises(QueryRoutingError, match="returned metaclip2"):
        registry.encode_query("beit3", "query")


class FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def to(self, device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    inference_mode = staticmethod(nullcontext)
    no_grad = staticmethod(nullcontext)

    @staticmethod
    def tensor(value, device=None):
        return FakeTensor(value)


def test_beit3_adapter_uses_locked_tokenizer_sequence_and_model_arguments() -> None:
    calls = {}

    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 2
        pad_token_id = 1

        def encode(self, text, *, add_special_tokens):
            calls["tokenizer"] = (text, add_special_tokens)
            return [7, 8]

    class Model:
        def __call__(self, **kwargs):
            calls["model"] = kwargs
            return None, FakeTensor(np.ones((1, 768), dtype=np.float32))

    adapter = Beit3TextEncoder(
        spec=SPECS.get("beit3"),
        checkpoint_path=Path("checkpoint"),
        tokenizer_path=Path("beit3.spm"),
        runtime_loader=lambda *args: (FakeTorch, Model(), Tokenizer(), "cpu"),
    )
    assert adapter.encode_text("hello").shape == (1, 768)
    assert calls["tokenizer"] == ("hello", False)
    assert set(calls["model"]) == {"text_description", "padding_mask", "only_infer"}
    assert calls["model"]["only_infer"] is True


def test_metaclip2_adapter_uses_paired_processor_and_get_text_features() -> None:
    calls = {}

    class Processor:
        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"input_ids": FakeTensor([[1, 2]])}

    class Model:
        def get_text_features(self, **kwargs):
            calls["model"] = kwargs
            return FakeTensor(np.ones((1, 512), dtype=np.float32))

    adapter = MetaClip2TextEncoder(
        spec=SPECS.get("metaclip2"),
        runtime_loader=lambda *args: (FakeTorch, Model(), Processor(), "cpu"),
    )
    assert adapter.encode_text("hello").shape == (1, 512)
    assert calls["processor"] == {
        "text": ["hello"],
        "padding": True,
        "truncation": True,
        "max_length": 77,
        "return_tensors": "pt",
    }
    assert "input_ids" in calls["model"]


def test_pe_adapter_uses_official_context_tokenizer_and_encode_text(monkeypatch) -> None:
    calls = {}

    class Tokenizer:
        def __call__(self, texts):
            calls["texts"] = texts
            return FakeTensor([[1, 2]])

    tokenizer = Tokenizer()
    transforms = SimpleNamespace(
        get_text_tokenizer=lambda context_length: (
            calls.setdefault("context_length", context_length),
            tokenizer,
        )[1]
    )
    monkeypatch.setattr(
        "mvr.enrichment.embeddings.query_encoders.importlib.import_module",
        lambda name: transforms,
    )

    class Model:
        context_length = 32

        def encode_text(self, tokens, *, normalize):
            calls["normalize"] = normalize
            return FakeTensor(np.ones((1, 1024), dtype=np.float32))

    adapter = PerceptionEncoderTextEncoder(
        spec=SPECS.get("perception_encoder"),
        checkpoint_path=Path("checkpoint"),
        runtime_loader=lambda *args: (FakeTorch, Model(), object(), "cpu"),
    )
    assert adapter.encode_text("hello").shape == (1, 1024)
    assert calls == {
        "context_length": 32,
        "texts": ["hello"],
        "normalize": True,
    }


def test_invalid_dimension_and_nonfinite_output_are_rejected() -> None:
    bad_dimension = QueryEncoderRegistry(
        SPECS,
        factories={"beit3": lambda spec: FakeAdapter("beit3", 2)},
    )
    with pytest.raises(QueryEncoderError, match="dimension"):
        bad_dimension.encode_query("beit3", "query")

    class NanAdapter(FakeAdapter):
        def encode_text(self, text):
            return np.full(self.dimension, np.nan, dtype=np.float32)

    nonfinite = QueryEncoderRegistry(
        SPECS,
        factories={"beit3": lambda spec: NanAdapter("beit3", spec.embedding_dimension)},
    )
    with pytest.raises(QueryEncoderError, match="finite"):
        nonfinite.encode_query("beit3", "query")
