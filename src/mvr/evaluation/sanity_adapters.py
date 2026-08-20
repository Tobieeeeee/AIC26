"""Retriever adapters used by the GT-free sanity harness."""
from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mvr.retrieval.lexical_bm25 import search_task05b_lexical

from .sanity_contracts import RetrievalHit


class ProviderUnavailableError(RuntimeError):
    """Raised when a configured retrieval provider or index is unavailable."""


@runtime_checkable
class RetrieverAdapter(Protocol):
    provider_id: str

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        ...

    def provenance(self) -> dict[str, Any]:
        ...


class BM25Adapter:
    """Thin adapter over the existing Task05B Elasticsearch/BM25 retriever."""

    provider_id = "bm25-task05b"

    def __init__(
        self,
        client: Any,
        *,
        index_name: str,
        manifest_path: Path | None = None,
    ) -> None:
        if not isinstance(index_name, str) or not index_name.strip():
            raise ValueError("index_name must be non-empty")
        self._client = client
        self._index_name = index_name.strip()
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self._provenance: dict[str, Any] | None = None

    def ensure_available(self) -> None:
        try:
            if not bool(self._client.ping()):
                raise ProviderUnavailableError("Elasticsearch did not respond to ping")
            if not bool(self._client.indices.exists(index=self._index_name)):
                raise ProviderUnavailableError(
                    f"Elasticsearch index does not exist: {self._index_name}"
                )
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Elasticsearch availability check failed: {exc}"
            ) from exc

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        hits = search_task05b_lexical(
            self._client,
            query,
            index_name=self._index_name,
            top_k=top_k,
        )
        return [
            RetrievalHit(
                rank=hit.rank,
                score=hit.score,
                keyframe_id=hit.keyframe_id,
                video_id=hit.video_id,
                frame_id=hit.frame_idx,
                shot_id=hit.shot_id,
                timestamp=hit.timestamp,
                provider=self.provider_id,
                source="task05b_lexical",
            )
            for hit in hits
        ]

    def provenance(self) -> dict[str, Any]:
        if self._provenance is not None:
            return dict(self._provenance)
        manifest: dict[str, Any] | None = None
        manifest_sha256: str | None = None
        if self._manifest_path is not None and self._manifest_path.is_file():
            manifest_bytes = self._manifest_path.read_bytes()
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            payload = json.loads(manifest_bytes.decode("utf-8"))
            if isinstance(payload, Mapping):
                manifest = dict(payload)
        try:
            info = self._client.info()
            if not isinstance(info, Mapping):
                info = getattr(info, "body", {})
            count = int(self._client.count(index=self._index_name)["count"])
            version = (
                info.get("version", {}).get("number")
                if isinstance(info, Mapping)
                else None
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"unable to read Elasticsearch provenance: {exc}"
            ) from exc
        self._provenance = {
            "provider_id": self.provider_id,
            "backend": "elasticsearch",
            "index_name": self._index_name,
            "index_document_count": count,
            "elasticsearch_version": version,
            "task05b_manifest_path": (
                str(self._manifest_path.resolve()) if self._manifest_path else None
            ),
            "task05b_manifest_sha256": manifest_sha256,
            "task05b_manifest": manifest,
        }
        return dict(self._provenance)
