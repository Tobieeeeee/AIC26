"""FastAPI serving layer adapted from the FastAPI Lab and Lab 6."""
from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from ..enrichment.qna_answer import AnswerReasoner, AnswerReasonerUnavailable, HuggingFaceCausalLmBackend
from ..ingestion.aic_dataset import AicDatasetRegistry
from ..retrieval.hybrid import rrf_fuse
from ..retrieval.kis import KeyframeFrameIndexResolver, search_kis
from ..retrieval.lexical import search_lexical
from ..retrieval.object_search import parse_object_conditions, search_objects
from ..retrieval.qna import search_qna
from .schemas import (
    ElasticsearchRuntimeStatus,
    HealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    KisHit,
    KisSearchRequest,
    KisSearchResponse,
    ObjectSearchRequest,
    QnaHit,
    QnaSearchRequest,
    QnaSearchResponse,
    SearchHit,
    SearchResponse,
    SemanticRuntimeStatus,
    SemanticSearchRequest,
    TextSearchRequest,
)


DEFAULT_ES_INDEX = "keyframes"
TOP_N_EACH = 30
_UNSET = object()
logger = logging.getLogger(__name__)


class LazyFaissSemanticService:
    """Load the persisted Phase 03 index and encoder only on first search."""

    def __init__(self, index_root: Path | None, *, device: str = "auto") -> None:
        self.index_root = index_root
        self.device = device
        self._index: Any | None = None
        self._retriever: Any | None = None
        self._readiness_error: Exception | None = None
        self._runtime_error: Exception | None = None

    @property
    def configured(self) -> bool:
        return self.index_root is not None

    @property
    def available(self) -> bool:
        if self._runtime_error is not None:
            return False
        try:
            self._load_index()
        except RuntimeError:
            return False
        return True

    @property
    def loaded(self) -> bool:
        return self._retriever is not None

    def _load_index(self):
        if self._index is not None:
            return self._index
        if self._readiness_error is not None:
            raise RuntimeError(
                "semantic index is unavailable"
            ) from self._readiness_error
        if self.index_root is None:
            raise RuntimeError("semantic index is unavailable")
        required = ("keyframes.faiss", "metadata.json", "manifest.json")
        if not all((self.index_root / name).is_file() for name in required):
            raise RuntimeError("semantic index is unavailable")
        if find_spec("faiss") is None or find_spec("open_clip") is None:
            raise RuntimeError("semantic runtime dependencies are unavailable")
        try:
            from ..indexing.faiss_index import FaissIndex

            self._index = FaissIndex.load(self.index_root)
        except Exception as exc:
            self._readiness_error = exc
            raise RuntimeError("semantic index is unavailable") from exc
        return self._index

    def search(self, query: str, top_k: int = 5):
        index = self._load_index()
        if self._retriever is None:
            try:
                from ..enrichment.embeddings.openclip import OpenClipEncoder
                from ..retrieval.semantic import SemanticRetriever

                model_name = index.manifest.get("embedding_model")
                pretrained = index.manifest.get("pretrained")
                if not isinstance(model_name, str) or not model_name.strip():
                    raise RuntimeError("semantic index manifest has no embedding model")
                if not isinstance(pretrained, str) or not pretrained.strip():
                    raise RuntimeError("semantic index manifest has no pretrained source")
                encoder = OpenClipEncoder(
                    model_name=model_name,
                    pretrained=pretrained,
                    device=self.device,
                )
                self._retriever = SemanticRetriever(encoder, index)
            except Exception as exc:
                self._runtime_error = exc
                raise RuntimeError("semantic runtime is unavailable") from exc
        return self._retriever.search(query, top_k=top_k)


class LazyAnswerReasoner:
    """Load the local-LLM Q&A backend only on first use, and only once.

    Fails closed via ``HuggingFaceCausalLmBackend`` (Task 03): unless
    ``MVR_QNA_ALLOW_MODEL_DOWNLOAD`` is explicitly set, the model must
    already be cached locally or every ``answer()`` call raises
    ``RuntimeError`` (mapped to HTTP 503 by the route), matching the lazy
    singleton pattern ``LazyFaissSemanticService`` already uses for the
    semantic index -- never reloaded per request.
    """

    def __init__(
        self, *, model_name: str, allow_model_download: bool, device: str = "auto"
    ) -> None:
        self.model_name = model_name
        self.allow_model_download = allow_model_download
        self.device = device
        self._reasoner: AnswerReasoner | None = None
        self._error: Exception | None = None

    @property
    def configured(self) -> bool:
        return bool(self.model_name)

    def _load(self) -> AnswerReasoner:
        if self._reasoner is not None:
            return self._reasoner
        if self._error is not None:
            raise RuntimeError("answer reasoner is unavailable") from self._error
        try:
            backend = HuggingFaceCausalLmBackend(
                model_name=self.model_name,
                allow_model_download=self.allow_model_download,
                device=self.device,
            )
        except AnswerReasonerUnavailable as exc:
            self._error = exc
            raise RuntimeError("answer reasoner is unavailable") from exc
        self._reasoner = AnswerReasoner(backend)
        return self._reasoner

    def answer(self, question: str, context: Any):
        return self._load().answer(question, context)


class ElasticsearchMultimodalService:
    """Call the already-reused Lab 6 retrieval modules without changing ranking."""

    available = True

    def __init__(
        self,
        client: Any,
        *,
        index_name: str = DEFAULT_ES_INDEX,
        device: str = "auto",
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.device = device
        self._semantic: Any | None = None

    def document_count(self) -> int:
        return int(self.client.count(index=self.index_name)["count"])

    def search_text(
        self,
        query: str,
        *,
        top_k: int,
        person_min: int,
        car_min: int,
    ) -> dict[str, Any]:
        return search_lexical(
            self.client,
            query,
            top_k=top_k,
            person_min=person_min,
            car_min=car_min,
            index_name=self.index_name,
        )

    def search_object(self, want: str, *, top_k: int) -> dict[str, Any]:
        return search_objects(
            self.client,
            want,
            top_k=top_k,
            index_name=self.index_name,
        )

    def _semantic_retriever(self):
        if self._semantic is None:
            from ..retrieval.semantic import (
                ElasticsearchSemanticRetriever,
                create_lab6_openclip_encoder,
            )

            encoder = create_lab6_openclip_encoder(device=self.device)
            self._semantic = ElasticsearchSemanticRetriever(
                encoder,
                self.client,
                index_name=self.index_name,
                num_candidates=200,
            )
        return self._semantic

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        lexical = self.search_text(
            query,
            top_k=TOP_N_EACH,
            person_min=0,
            car_min=0,
        )
        semantic = self._semantic_retriever().search(
            query,
            top_k=TOP_N_EACH,
        )
        lexical_hits = list(lexical["hits"]["hits"])
        semantic_hits = list(semantic["hits"]["hits"])
        ranked_lists = [
            [hit["_source"]["frame_id"] for hit in lexical_hits],
            [hit["_source"]["frame_id"] for hit in semantic_hits],
        ]
        fused = rrf_fuse(ranked_lists, rrf_k=rrf_k, top_k=top_k)

        payloads: dict[str, dict[str, Any]] = {}
        for hits, rank_name in (
            (lexical_hits, "rank_lexical"),
            (semantic_hits, "rank_semantic"),
        ):
            for rank, hit in enumerate(hits, start=1):
                frame_id = str(hit["_source"]["frame_id"])
                payloads.setdefault(frame_id, _hit_payload(hit, rank=1))[
                    rank_name
                ] = rank

        results = []
        for rank, (frame_id, score) in enumerate(fused, start=1):
            payload = dict(payloads[frame_id])
            payload.update(rank=rank, score=float(score))
            results.append(payload)
        return results


def _image_path(source: dict[str, Any]) -> str:
    existing = source.get("image_path")
    if existing:
        return str(existing)
    frame_id = str(source["frame_id"])
    video_id = str(source["video_id"])
    return f"/keyframes/{video_id}/{frame_id}.jpg"


def _hit_payload(hit: dict[str, Any], *, rank: int) -> dict[str, Any]:
    source = hit["_source"]
    return {
        "rank": rank,
        "frame_id": str(source["frame_id"]),
        "video_id": str(source["video_id"]),
        "timestamp": float(source["ts"]),
        "score": float(hit.get("_score") or 0.0),
        "image_path": _image_path(source),
        "ocr_text": str(source.get("ocr_text", "")),
        "caption": str(source.get("caption", "")),
        "objects": {
            str(name): int(count)
            for name, count in source.get("objects", {}).items()
        },
        "highlight": dict(hit.get("highlight", {})),
    }


def _servable_keyframe_url(raw_path: str | None, keyframe_dir: Path | None) -> str | None:
    """Map a local keyframe path to the URL this app actually serves it at.

    Returns ``None`` -- rather than the local path -- whenever the image
    cannot be served (no mounted directory, or the file lives outside it).
    Q&A candidates routinely come from an organizer dataset root that is
    not the mounted keyframe directory, and leaking absolute local paths to
    an API client would be both useless to them and an unnecessary
    disclosure of the local filesystem layout.
    """
    if raw_path is None or keyframe_dir is None:
        return None
    try:
        relative_path = Path(raw_path).resolve().relative_to(keyframe_dir.resolve())
    except (ValueError, OSError):
        return None
    return f"/keyframes/{relative_path.as_posix()}"


def _semantic_payload(result: Any, keyframe_dir: Path | None) -> dict[str, Any]:
    payload = result.to_dict()
    if keyframe_dir is None:
        return payload
    image_path = Path(payload["image_path"]).resolve()
    try:
        relative_path = image_path.relative_to(keyframe_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "semantic result image is outside the configured keyframe directory"
        ) from exc
    payload["image_path"] = f"/keyframes/{relative_path.as_posix()}"
    return payload


def _response_from_es(query: str, response: dict[str, Any]) -> SearchResponse:
    raw_hits = list(response["hits"]["hits"])
    total = response["hits"]["total"]
    count = int(total["value"] if isinstance(total, dict) else total)
    return SearchResponse(
        query=query,
        count=count,
        results=[
            SearchHit(**_hit_payload(hit, rank=rank))
            for rank, hit in enumerate(raw_hits, start=1)
        ],
    )


def _default_semantic_service() -> LazyFaissSemanticService:
    raw_path = os.environ.get("MVR_SEMANTIC_INDEX", "").strip()
    index_root = Path(raw_path).resolve() if raw_path else None
    return LazyFaissSemanticService(
        index_root,
        device=os.environ.get("MVR_DEVICE", "auto"),
    )


def _default_multimodal_service() -> tuple[Any | None, bool]:
    es_url = os.environ.get("MVR_ES_URL", "").strip()
    if not es_url:
        return None, False
    try:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(
            es_url,
            request_timeout=120,
            max_retries=3,
            retry_on_timeout=True,
        )
        index_name = os.environ.get("MVR_ES_INDEX", DEFAULT_ES_INDEX)
        if not client.ping() or not client.indices.exists(index=index_name):
            return None, True
        return (
            ElasticsearchMultimodalService(
                client,
                index_name=index_name,
                device=os.environ.get("MVR_DEVICE", "auto"),
            ),
            True,
        )
    except Exception:
        return None, True


def _default_kis_pipeline_resolver() -> KeyframeFrameIndexResolver | None:
    raw_path = os.environ.get("MVR_KEYFRAME_MAPPING_CSV", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    if not path.is_file():
        return None
    return KeyframeFrameIndexResolver.from_csv(path)


def _default_aic_registry() -> AicDatasetRegistry | None:
    raw_path = os.environ.get("MVR_AIC_DATASET_ROOT", "").strip()
    if not raw_path:
        return None
    return AicDatasetRegistry(Path(raw_path).resolve())


def _default_qna_answer_reasoner() -> LazyAnswerReasoner | None:
    model_name = os.environ.get("MVR_QNA_MODEL", "").strip()
    if not model_name:
        return None
    allow_download = os.environ.get("MVR_QNA_ALLOW_MODEL_DOWNLOAD", "").strip().lower() in (
        "1", "true", "yes",
    )
    return LazyAnswerReasoner(
        model_name=model_name,
        allow_model_download=allow_download,
        device=os.environ.get("MVR_DEVICE", "auto"),
    )


def _semantic_status(service: Any | None) -> SemanticRuntimeStatus:
    if service is None:
        return SemanticRuntimeStatus(
            configured=False,
            available=False,
            loaded=False,
        )
    return SemanticRuntimeStatus(
        configured=bool(getattr(service, "configured", True)),
        available=bool(getattr(service, "available", True)),
        loaded=bool(getattr(service, "loaded", True)),
    )


def create_app(
    *,
    semantic_service: Any = _UNSET,
    multimodal_service: Any = _UNSET,
    elasticsearch_configured: bool | None = None,
    keyframe_dir: Path | None | object = _UNSET,
    kis_pipeline_resolver: Any = _UNSET,
    kis_aic_registry: Any = _UNSET,
    qna_answer_reasoner: Any = _UNSET,
) -> FastAPI:
    """Build an app with injectable services for lightweight contract tests."""
    if semantic_service is _UNSET:
        semantic_service = _default_semantic_service()
    if multimodal_service is _UNSET:
        multimodal_service, detected_configuration = _default_multimodal_service()
        if elasticsearch_configured is None:
            elasticsearch_configured = detected_configuration
    if elasticsearch_configured is None:
        elasticsearch_configured = multimodal_service is not None
    if keyframe_dir is _UNSET:
        raw_keyframes = os.environ.get("MVR_KEYFRAME_DIR", "").strip()
        keyframe_dir = Path(raw_keyframes).resolve() if raw_keyframes else None
    if kis_pipeline_resolver is _UNSET:
        kis_pipeline_resolver = _default_kis_pipeline_resolver()
    if kis_aic_registry is _UNSET:
        kis_aic_registry = _default_aic_registry()
    if qna_answer_reasoner is _UNSET:
        qna_answer_reasoner = _default_qna_answer_reasoner()

    application = FastAPI(
        title="MVR Competition Search API",
        description="Serving layer for the reused semantic and multimodal baseline.",
        version="0.1.0",
    )
    if isinstance(keyframe_dir, Path) and keyframe_dir.is_dir():
        application.mount(
            "/keyframes",
            StaticFiles(directory=keyframe_dir),
            name="keyframes",
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        response_model_exclude_none=True,
    )
    def health() -> HealthResponse:
        es_available = multimodal_service is not None
        documents = None
        if es_available:
            try:
                documents = multimodal_service.document_count()
            except Exception:
                es_available = False
        semantic = _semantic_status(semantic_service)
        status = "ok" if semantic.available else "degraded"
        return HealthResponse(
            status=status,
            semantic=semantic,
            elasticsearch=ElasticsearchRuntimeStatus(
                configured=bool(elasticsearch_configured),
                available=es_available,
                documents=documents,
            ),
        )

    @application.post("/search/semantic", response_model=SearchResponse)
    def search_semantic(request: SemanticSearchRequest) -> SearchResponse:
        if semantic_service is None:
            raise HTTPException(status_code=503, detail="semantic index is unavailable")
        try:
            results = semantic_service.search(request.query, top_k=request.top_k)
            payloads = [
                _semantic_payload(result, keyframe_dir)
                for result in results
            ]
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SearchResponse(
            query=request.query,
            count=len(results),
            results=[SearchHit(**payload) for payload in payloads],
        )

    @application.post("/search/kis", response_model=KisSearchResponse)
    def search_kis_route(request: KisSearchRequest) -> KisSearchResponse:
        try:
            candidates = search_kis(
                request.query,
                top_k=request.top_k,
                semantic_service=semantic_service,
                multimodal_service=multimodal_service,
                pipeline_resolver=kis_pipeline_resolver,
                aic_registry=kis_aic_registry,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return KisSearchResponse(
            query=request.query,
            count=len(candidates),
            results=[
                KisHit(
                    rank=candidate.rank,
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    score=candidate.score,
                    keyframe_id=candidate.keyframe_id,
                    timestamp=candidate.timestamp,
                )
                for candidate in candidates
            ],
        )

    @application.post("/search/qna", response_model=QnaSearchResponse)
    def search_qna_route(request: QnaSearchRequest) -> QnaSearchResponse:
        try:
            candidates = search_qna(
                request.event_description,
                request.question,
                top_k=request.top_k,
                semantic_service=semantic_service,
                multimodal_service=multimodal_service,
                pipeline_resolver=kis_pipeline_resolver,
                aic_registry=kis_aic_registry,
                answer_reasoner=qna_answer_reasoner,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return QnaSearchResponse(
            event_description=request.event_description,
            question=request.question,
            count=len(candidates),
            results=[
                QnaHit(
                    rank=candidate.rank,
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    score=candidate.score,
                    keyframe_id=candidate.keyframe_id,
                    timestamp=candidate.timestamp,
                    answer=candidate.answer,
                    answer_source=candidate.answer_source,
                    image_path=_servable_keyframe_url(
                        candidate.image_path,
                        keyframe_dir if isinstance(keyframe_dir, Path) else None,
                    ),
                )
                for candidate in candidates
            ],
        )

    if multimodal_service is not None:

        @application.post("/search/text", response_model=SearchResponse)
        def search_text(request: TextSearchRequest) -> SearchResponse:
            try:
                response = multimodal_service.search_text(
                    request.query,
                    top_k=request.top_k,
                    person_min=request.person_min,
                    car_min=request.car_min,
                )
            except Exception as exc:
                logger.exception("text retrieval failed")
                raise HTTPException(
                    status_code=503,
                    detail="text retrieval service is unavailable",
                ) from exc
            return _response_from_es(request.query, response)

        @application.post("/search/object", response_model=SearchResponse)
        def search_object(request: ObjectSearchRequest) -> SearchResponse:
            try:
                parse_object_conditions(request.want)
                response = multimodal_service.search_object(
                    request.want,
                    top_k=request.top_k,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("object retrieval failed")
                raise HTTPException(
                    status_code=503,
                    detail="object retrieval service is unavailable",
                ) from exc
            return _response_from_es(request.want, response)

        @application.post("/search/hybrid", response_model=HybridSearchResponse)
        def search_hybrid(request: HybridSearchRequest) -> HybridSearchResponse:
            try:
                results = multimodal_service.search_hybrid(
                    request.query,
                    top_k=request.top_k,
                    rrf_k=request.rrf_k,
                )
            except Exception as exc:
                logger.exception("hybrid retrieval failed")
                raise HTTPException(
                    status_code=503,
                    detail="hybrid retrieval service is unavailable",
                ) from exc
            return HybridSearchResponse(
                query=request.query,
                count=len(results),
                rrf_k=request.rrf_k,
                results=[SearchHit(**result) for result in results],
            )

    return application


app = create_app()
