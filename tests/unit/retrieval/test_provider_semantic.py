from __future__ import annotations

import numpy as np
import pytest

from mvr.enrichment.embeddings.query_encoders import QueryEmbedding, QueryRoutingError
from mvr.indexing.provider_indexes import ProviderIndexContract
from mvr.retrieval.provider_semantic import (
    BackendSearchHit,
    IndependentSemanticRetriever,
    ProviderSearchError,
)


PROVIDERS = {"beit3": 3, "metaclip2": 2, "perception_encoder": 4}


class FakeEncoders:
    def __init__(self, *, fail: str | None = None, wrong_model: str | None = None) -> None:
        self.fail = fail
        self.wrong_model = wrong_model
        self.calls = []

    def encode_query(self, model_id, text, *, index_contract):
        self.calls.append((model_id, text, index_contract.collection_name))
        if model_id == self.fail:
            raise TimeoutError("encoder timeout")
        dimension = index_contract.dimension
        vector = np.zeros(dimension, dtype=np.float32)
        vector[0] = 1
        return QueryEmbedding(
            model_id=self.wrong_model or model_id,
            vector=vector,
            dimension=dimension,
            dtype="float32",
            normalized=True,
        )


class FakeBackend:
    def __init__(self, hits=None, *, fail: str | None = None) -> None:
        self.hits = hits or {}
        self.fail = fail
        self.calls = []

    def search(self, contract, vector, top_k):
        self.calls.append((contract.model_id, contract.collection_name, vector.shape, top_k))
        if contract.model_id == self.fail:
            raise TimeoutError("backend timeout")
        return self.hits.get(contract.model_id, [])[:top_k]


def _metadata(keyframe_id: str, frame_id: int):
    return {
        "keyframe_id": keyframe_id,
        "video_id": "V1",
        "frame_id": frame_id,
        "shot_id": frame_id // 10,
        "timestamp": frame_id / 25,
    }


def _contracts(metric="inner_product"):
    return {
        model_id: ProviderIndexContract(
            model_id=model_id,
            collection_name=f"test__fasttest__{model_id}",
            dimension=dimension,
            metric=metric,
            normalized=True,
            backend="fake",
        )
        for model_id, dimension in PROVIDERS.items()
    }


def _retriever(encoders=None, backend=None, **kwargs):
    return IndependentSemanticRetriever(
        query_encoders=encoders or FakeEncoders(),
        backend=backend or FakeBackend(),
        contracts=_contracts(kwargs.pop("metric", "inner_product")),
        required_providers=tuple(PROVIDERS),
        **kwargs,
    )


def test_three_providers_route_to_matching_collections_and_keep_lists_independent() -> None:
    hits = {
        model_id: [
            BackendSearchHit(1.0 - index / 10, _metadata(f"K{index}", index * 10))
            for index in range(1, 6)
        ]
        for model_id in PROVIDERS
    }
    backend = FakeBackend(hits)
    retriever = _retriever(backend=backend)
    response = retriever.search(
        "a red bus", provider_ids=tuple(PROVIDERS), top_k=5
    )
    assert set(response.ranked_lists) == set(PROVIDERS)
    assert len(response.ranked_lists) == 3
    for model_id, results in response.ranked_lists.items():
        assert len(results) == 5
        assert [result.rank for result in results] == [1, 2, 3, 4, 5]
        assert all(result.model_id == model_id for result in results)
        assert all(result.video_id == "V1" for result in results)
    assert [call[1] for call in backend.calls] == [
        f"test__fasttest__{model_id}" for model_id in PROVIDERS
    ]
    payload = response.to_dict()
    assert set(payload["ranked_lists"]) == set(PROVIDERS)
    assert "fused" not in payload


def test_top_k_cap_ties_and_duplicate_hits_are_deterministic() -> None:
    backend = FakeBackend(
        {
            "beit3": [
                BackendSearchHit(0.7, _metadata("K2", 20)),
                BackendSearchHit(0.9, _metadata("K3", 30)),
                BackendSearchHit(0.8, _metadata("K2", 20)),
                BackendSearchHit(0.9, _metadata("K1", 10)),
            ]
        }
    )
    retriever = _retriever(backend=backend, max_top_k=3)
    results = retriever.search_provider("beit3", "query", top_k=99)
    assert [(item.rank, item.keyframe_id, item.score) for item in results] == [
        (1, "K1", 0.9),
        (2, "K3", 0.9),
        (3, "K2", 0.8),
    ]
    assert backend.calls[0][3] == 6


def test_lower_score_metric_direction_is_supported() -> None:
    backend = FakeBackend(
        {
            "beit3": [
                BackendSearchHit(0.4, _metadata("K2", 20)),
                BackendSearchHit(0.1, _metadata("K3", 30)),
                BackendSearchHit(0.1, _metadata("K1", 10)),
            ]
        }
    )
    results = _retriever(backend=backend, metric="l2").search_provider(
        "beit3", "query", top_k=3
    )
    assert [item.keyframe_id for item in results] == ["K1", "K3", "K2"]


def test_empty_hits_return_an_empty_ranked_list() -> None:
    response = _retriever().search(
        "query", provider_ids=["beit3"], top_k=5
    )
    assert response.ranked_lists == {"beit3": ()}
    assert response.successful_providers == ("beit3",)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        _retriever().search_provider("beit3", "query", top_k=top_k)


def test_encoder_result_model_mismatch_stops_before_backend() -> None:
    backend = FakeBackend()
    retriever = _retriever(
        encoders=FakeEncoders(wrong_model="metaclip2"), backend=backend
    )
    with pytest.raises(QueryRoutingError, match="does not match"):
        retriever.search_provider("beit3", "query", top_k=5)
    assert backend.calls == []


def test_contract_cross_route_stops_before_encoder_or_backend() -> None:
    encoders = FakeEncoders()
    backend = FakeBackend()
    contracts = _contracts()
    contracts["beit3"] = ProviderIndexContract(
        "beit3", "test__fasttest__metaclip2", 3, "inner_product", True, "fake"
    )
    retriever = IndependentSemanticRetriever(
        query_encoders=encoders, backend=backend, contracts=contracts
    )
    with pytest.raises(QueryRoutingError, match="cross-provider"):
        retriever.search_provider("beit3", "query", top_k=5)
    assert encoders.calls == []
    assert backend.calls == []


def test_required_provider_error_fail_closed() -> None:
    retriever = _retriever(
        backend=FakeBackend(fail="metaclip2"), failure_policy="fail_closed"
    )
    with pytest.raises(ProviderSearchError, match="metaclip2"):
        retriever.search(
            "query", provider_ids=["beit3", "metaclip2", "perception_encoder"], top_k=5
        )


def test_provider_error_degraded_reports_coverage_without_fallback() -> None:
    encoders = FakeEncoders(fail="metaclip2")
    response = _retriever(
        encoders=encoders, failure_policy="degraded"
    ).search(
        "query", provider_ids=["beit3", "metaclip2", "perception_encoder"], top_k=5
    )
    assert response.successful_providers == ("beit3", "perception_encoder")
    assert response.failed_providers == ("metaclip2",)
    assert set(response.ranked_lists) == {"beit3", "perception_encoder"}
    assert "TimeoutError" in response.errors["metaclip2"]
    assert encoders.calls == [
        ("beit3", "query", "test__fasttest__beit3"),
        ("metaclip2", "query", "test__fasttest__metaclip2"),
        ("perception_encoder", "query", "test__fasttest__perception_encoder"),
    ]
