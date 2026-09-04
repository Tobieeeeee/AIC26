"""FastAPI serving layer adapted from the FastAPI Lab and Lab 6."""
from __future__ import annotations

import logging
import os
import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..enrichment.qna_answer import AnswerReasoner, AnswerReasonerUnavailable, HuggingFaceCausalLmBackend
from ..enrichment.qna_translation import DEFAULT_TRANSLATION_MODEL, HelsinkiViEnTranslator, TranslatorUnavailable
from ..ingestion.aic_dataset import AicDatasetRegistry
from ..ingestion.video_summary import build_video_summary
from ..retrieval.hybrid import rrf_fuse
from ..retrieval.kis import KeyframeFrameIndexResolver, search_kis
from ..retrieval.lexical import search_lexical
from ..retrieval.object_search import parse_object_conditions, search_objects
from ..retrieval.qna import search_qna
from ..retrieval.trake import search_trake
from ..retrieval.trake_events import TrakeEventQuery, TrakeQuery, translate_gap_phrase
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
    TrakeChainHit,
    TrakeSearchRequest,
    TrakeSearchResponse,
    UiKeyframeResult,
    UiSearchRequest,
    UiSearchResponse,
    UiSubmitRequest,
    UiSubmitResponse,
    VideoSummaryKeyframe as ApiVideoSummaryKeyframe,
    VideoSummaryResponse,
)


DEFAULT_ES_INDEX = "keyframes"
DEFAULT_SEMANTIC_BACKEND = "milvus"
TOP_N_EACH = 30
_UNSET = object()
# Evidence-backed default from the real-data benchmark (see
# docs/AIC_TRAKE_BENCHMARK_RESULTS.md) -- not loaded from configs/default.yaml
# because nothing in this codebase actually reads that file; every other
# "config" value here is likewise a plain Python default (see e.g. each
# TrakeAlignmentStrategy's own __init__ defaults, already set to the tuned
# values). Introducing a YAML-loading mechanism for just this one value
# would be new, unprecedented machinery, not a missing wire-up.
TRAKE_DEFAULT_ALIGNMENT_STRATEGY = "atlas"
logger = logging.getLogger(__name__)


def _build_trake_alignment_strategy(name: str):
    from ..retrieval.trake_align import AtlasAlignment, LuciferAlignment, UceseAlignment

    strategies = {"atlas": AtlasAlignment, "lucifer": LuciferAlignment, "ucese": UceseAlignment}
    return strategies[name]()


def _resolve_trake_semantic_service(service: Any) -> Any:
    """Return something exposing ``.text_encoder`` and ``.index``, which
    TRAKE's dense scoring needs but plain ``.search()`` does not provide.

    Accepts either shape, because ``create_app`` documents its services as
    injectable and both are legitimate injections:

    - ``LazyFaissSemanticService`` (the explicit reference backend) hides them behind
      a lazy ``retriever()`` so the model/index still load on first use only.
    - A plain ``SemanticRetriever`` already exposes them directly -- this is
      exactly what the real-data integration tests construct.

    Raises ``RuntimeError`` (-> HTTP 503) for a service supporting neither,
    rather than letting an ``AttributeError`` surface as an opaque 500.
    """
    if hasattr(service, "retriever"):
        return service.retriever()
    if hasattr(service, "text_encoder") and hasattr(service, "index"):
        return service
    raise RuntimeError(
        "configured semantic service does not support TRAKE dense scoring "
        "(needs either a retriever() method or .text_encoder/.index attributes)"
    )


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
            self._read_manifest()
        except RuntimeError:
            return False
        return True

    @property
    def loaded(self) -> bool:
        return self._retriever is not None

    def _load_index(self):
        if self._index is not None:
            return self._index
        self._read_manifest()
        try:
            from ..indexing.faiss_index import FaissIndex

            self._index = FaissIndex.load(self.index_root)
        except Exception as exc:
            self._readiness_error = exc
            raise RuntimeError("semantic index is unavailable") from exc
        return self._index

    def _read_manifest(self) -> dict[str, Any]:
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
            manifest = json.loads(
                (self.index_root / "manifest.json").read_text(encoding="utf-8")
            )
        except Exception as exc:
            self._readiness_error = exc
            raise RuntimeError("semantic index is unavailable") from exc
        if not isinstance(manifest, dict):
            self._readiness_error = ValueError("semantic index manifest must be an object")
            raise RuntimeError("semantic index is unavailable") from self._readiness_error
        return manifest

    def _load_retriever(self):
        if self._retriever is None:
            try:
                from ..enrichment.embeddings.openclip import OpenClipEncoder
                from ..retrieval.semantic import SemanticRetriever

                manifest = self._read_manifest()
                model_name = manifest.get("embedding_model")
                pretrained = manifest.get("pretrained")
                if not isinstance(model_name, str) or not model_name.strip():
                    raise RuntimeError("semantic index manifest has no embedding model")
                if not isinstance(pretrained, str) or not pretrained.strip():
                    raise RuntimeError("semantic index manifest has no pretrained source")
                encoder = OpenClipEncoder(
                    model_name=model_name,
                    pretrained=pretrained,
                    device=self.device,
                )
                index = self._load_index()
                self._retriever = SemanticRetriever(encoder, index)
            except Exception as exc:
                self._runtime_error = exc
                raise RuntimeError("semantic runtime is unavailable") from exc
        return self._retriever

    def search(self, query: str, top_k: int = 5):
        return self._load_retriever().search(query, top_k=top_k)

    def retriever(self):
        """Expose the underlying ``SemanticRetriever`` (``.text_encoder`` /
        ``.index``) for callers that need dense per-frame scoring (TRAKE's
        ``search_trake()``), not just top-k ``search()``. Same lazy-load
        guarantee and exceptions as ``search()`` -- purely additive, does
        not change ``search()``'s own behavior."""
        return self._load_retriever()


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


class LazyQueryTranslator:
    """Load the Vietnamese->English query translator only on first use.

    Same lazy/fail-closed shape as ``LazyAnswerReasoner``. Kept opt-in
    (``MVR_TRAKE_TRANSLATE_QUERIES``, see ``_default_trake_query_translator``)
    rather than always-on: the measured TRAKE benchmark
    (docs/AIC_TRAKE_BENCHMARK_RESULTS.md) used untranslated English event
    text throughout, and this Vietnamese->English model's behavior on
    already-English input is unverified -- enabling it unconditionally
    could silently change that result. ``translate_for_retrieval`` (used by
    ``search_trake``) already falls back to the original text on any
    failure here, so a missing/uncached model degrades to "no translation"
    rather than a 503.
    """

    def __init__(
        self, *, model_name: str, allow_model_download: bool, device: str = "cpu"
    ) -> None:
        self.model_name = model_name
        self.allow_model_download = allow_model_download
        self.device = device
        self._translator: HelsinkiViEnTranslator | None = None
        self._error: Exception | None = None

    def _load(self) -> HelsinkiViEnTranslator:
        if self._translator is not None:
            return self._translator
        if self._error is not None:
            raise RuntimeError("query translator is unavailable") from self._error
        try:
            self._translator = HelsinkiViEnTranslator(
                model_name=self.model_name,
                allow_model_download=self.allow_model_download,
                device=self.device,
            )
        except TranslatorUnavailable as exc:
            self._error = exc
            raise RuntimeError("query translator is unavailable") from exc
        return self._translator

    def translate(self, text: str) -> str:
        return self._load().translate(text)


class ElasticsearchMultimodalService:
    """Call the already-reused Lab 6 retrieval modules without changing ranking."""

    available = True

    def __init__(
        self,
        client: Any,
        *,
        index_name: str = DEFAULT_ES_INDEX,
        device: str = "auto",
        semantic_service: Any | None = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.device = device
        self.semantic_service = semantic_service

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
        if self.semantic_service is None:
            raise RuntimeError("semantic retrieval service is unavailable")
        semantic_results = self.semantic_service.search(query, top_k=TOP_N_EACH)
        lexical_hits = list(lexical["hits"]["hits"])
        lexical_payloads = [
            _hit_payload(hit, rank=rank)
            for rank, hit in enumerate(lexical_hits, start=1)
        ]
        semantic_payloads = []
        for result in semantic_results:
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            if not payload.get("image_path") and payload.get("keyframe_id"):
                payload["image_path"] = (
                    f"/keyframes/{payload['video_id']}/{payload['keyframe_id']}.jpg"
                )
            semantic_payloads.append(payload)

        def identity(payload: dict[str, Any]) -> str:
            keyframe_id = payload.get("keyframe_id", payload["frame_id"])
            return f"{payload['video_id']}\x1f{keyframe_id}"

        ranked_lists = [
            [identity(payload) for payload in lexical_payloads],
            [identity(payload) for payload in semantic_payloads],
        ]
        fused = rrf_fuse(ranked_lists, rrf_k=rrf_k, top_k=top_k)

        payloads: dict[str, dict[str, Any]] = {}
        for hits, rank_name in (
            (lexical_payloads, "rank_lexical"),
            (semantic_payloads, "rank_semantic"),
        ):
            for rank, payload in enumerate(hits, start=1):
                candidate_id = identity(payload)
                payloads.setdefault(candidate_id, dict(payload))[rank_name] = rank

        results = []
        for rank, (candidate_id, score) in enumerate(fused, start=1):
            payload = dict(payloads[candidate_id])
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
    if not payload.get("image_path") and payload.get("keyframe_id"):
        payload["image_path"] = (
            f"/keyframes/{payload['video_id']}/{payload['keyframe_id']}.jpg"
        )
    if keyframe_dir is None or str(payload.get("image_path", "")).startswith(
        "/keyframes/"
    ):
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


def _default_semantic_service() -> Any:
    backend_name = os.environ.get(
        "MVR_SEMANTIC_BACKEND", DEFAULT_SEMANTIC_BACKEND
    ).strip().lower()
    if backend_name == "milvus":
        from ..retrieval.milvus_semantic import MilvusSemanticService

        return MilvusSemanticService.from_env(
            index_type=os.environ.get("MVR_MILVUS_INDEX_TYPE", "FLAT").strip().upper()
        )
    if backend_name == "faiss_reference":
        raw_path = os.environ.get("MVR_SEMANTIC_INDEX", "").strip()
        index_root = Path(raw_path).resolve() if raw_path else None
        return LazyFaissSemanticService(
            index_root,
            device=os.environ.get("MVR_DEVICE", "auto"),
        )
    raise RuntimeError(
        "MVR_SEMANTIC_BACKEND must be 'milvus' or 'faiss_reference'"
    )


def _default_multimodal_service(
    semantic_service: Any | None = None,
) -> tuple[Any | None, bool]:
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
                semantic_service=semantic_service,
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


def _default_trake_query_translator() -> LazyQueryTranslator | None:
    if os.environ.get("MVR_TRAKE_TRANSLATE_QUERIES", "").strip().lower() not in (
        "1", "true", "yes",
    ):
        return None
    allow_download = os.environ.get("MVR_TRAKE_ALLOW_MODEL_DOWNLOAD", "").strip().lower() in (
        "1", "true", "yes",
    )
    model_name = os.environ.get("MVR_TRAKE_TRANSLATION_MODEL", "").strip() or DEFAULT_TRANSLATION_MODEL
    return LazyQueryTranslator(
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
    runtime_status = getattr(service, "runtime_status", None)
    if callable(runtime_status):
        return SemanticRuntimeStatus(**runtime_status())
    return SemanticRuntimeStatus(
        configured=bool(getattr(service, "configured", True)),
        available=bool(getattr(service, "available", True)),
        loaded=bool(getattr(service, "loaded", True)),
    )


def _format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _summary_mapping_csv(
    video_id: str,
    keyframe_dir: Path | None,
    aic_registry: Any | None,
) -> Path | None:
    registry_root = getattr(aic_registry, "root", None)
    if isinstance(registry_root, Path):
        candidate = registry_root / "map_keyframes" / f"{video_id}.csv"
        if candidate.is_file():
            return candidate
    if keyframe_dir is not None:
        candidate = keyframe_dir.parent / "map_keyframes" / f"{video_id}.csv"
        if candidate.is_file():
            return candidate
    return None


def _frame_asset_urls(
    *,
    video_id: str,
    frame_id: int,
    keyframe_id: str | None,
    image_path: str | None,
    keyframe_dir: Path | None,
    video_dir: Path | None,
) -> tuple[str, str]:
    thumbnail = _servable_keyframe_url(image_path, keyframe_dir)
    if thumbnail is None and image_path and str(image_path).startswith("/keyframes/"):
        thumbnail = str(image_path)
    if thumbnail is None and keyframe_id:
        keyframe_name = keyframe_id if Path(keyframe_id).suffix else f"{keyframe_id}.jpg"
        thumbnail = f"/keyframes/{video_id}/{keyframe_name}"
    if thumbnail is None:
        thumbnail = f"/keyframes/{video_id}/{frame_id}.jpg"

    video_url = f"/videos/{video_id}.mp4" if video_dir is not None else ""
    return thumbnail, video_url


def _ui_result(
    *,
    rank: int,
    video_id: str,
    frame_id: int,
    score: float,
    timestamp: float | None,
    keyframe_id: str | None,
    image_path: str | None,
    keyframe_dir: Path | None,
    video_dir: Path | None,
    answer: str | None = None,
    objects: list[str] | None = None,
    frames: list[int | None] | None = None,
) -> UiKeyframeResult:
    thumbnail_url, video_url = _frame_asset_urls(
        video_id=video_id,
        frame_id=frame_id,
        keyframe_id=keyframe_id,
        image_path=image_path,
        keyframe_dir=keyframe_dir,
        video_dir=video_dir,
    )
    timestamp_sec = float(timestamp if timestamp is not None else 0.0)
    return UiKeyframeResult(
        id=f"{video_id}:{frame_id}:{rank}",
        video_id=video_id,
        frame_id=frame_id,
        timestamp=_format_timestamp(timestamp),
        timestamp_sec=timestamp_sec,
        thumbnail_url=thumbnail_url,
        video_url=video_url,
        score=score,
        objects=objects or [],
        answer=answer,
        frames=frames,
    )


def create_app(
    *,
    semantic_service: Any = _UNSET,
    multimodal_service: Any = _UNSET,
    elasticsearch_configured: bool | None = None,
    keyframe_dir: Path | None | object = _UNSET,
    video_dir: Path | None | object = _UNSET,
    kis_pipeline_resolver: Any = _UNSET,
    kis_aic_registry: Any = _UNSET,
    qna_answer_reasoner: Any = _UNSET,
    trake_query_translator: Any = _UNSET,
) -> FastAPI:
    """Build an app with injectable services for lightweight contract tests."""
    if semantic_service is _UNSET:
        semantic_service = _default_semantic_service()
    if multimodal_service is _UNSET:
        multimodal_service, detected_configuration = _default_multimodal_service(
            semantic_service
        )
        if elasticsearch_configured is None:
            elasticsearch_configured = detected_configuration
    if elasticsearch_configured is None:
        elasticsearch_configured = multimodal_service is not None
    if keyframe_dir is _UNSET:
        raw_keyframes = os.environ.get("MVR_KEYFRAME_DIR", "").strip()
        keyframe_dir = Path(raw_keyframes).resolve() if raw_keyframes else None
    if video_dir is _UNSET:
        raw_videos = os.environ.get("MVR_VIDEO_DIR", "").strip()
        video_dir = Path(raw_videos).resolve() if raw_videos else None
    if kis_pipeline_resolver is _UNSET:
        kis_pipeline_resolver = _default_kis_pipeline_resolver()
    if kis_aic_registry is _UNSET:
        kis_aic_registry = _default_aic_registry()
    if qna_answer_reasoner is _UNSET:
        qna_answer_reasoner = _default_qna_answer_reasoner()
    if trake_query_translator is _UNSET:
        trake_query_translator = _default_trake_query_translator()

    application = FastAPI(
        title="MVR Competition Search API",
        description="Serving layer for the reused semantic and multimodal baseline.",
        version="0.1.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if isinstance(keyframe_dir, Path) and keyframe_dir.is_dir():
        application.mount(
            "/keyframes",
            StaticFiles(directory=keyframe_dir),
            name="keyframes",
        )
    if isinstance(video_dir, Path) and video_dir.is_dir():
        application.mount(
            "/videos",
            StaticFiles(directory=video_dir),
            name="videos",
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

    @application.post("/api/v1/search", response_model=UiSearchResponse)
    def ui_search(request: UiSearchRequest) -> UiSearchResponse:
        served_keyframes = keyframe_dir if isinstance(keyframe_dir, Path) else None
        served_videos = video_dir if isinstance(video_dir, Path) else None

        if request.task_mode == "Q&A":
            question = (request.question or "").strip() or request.query
            event_description = (request.event_description or request.query).strip()
            try:
                candidates = search_qna(
                    event_description,
                    question,
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

            results = [
                _ui_result(
                    rank=candidate.rank,
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    score=candidate.score,
                    timestamp=candidate.timestamp,
                    keyframe_id=candidate.keyframe_id,
                    image_path=candidate.image_path,
                    keyframe_dir=served_keyframes,
                    video_dir=served_videos,
                    answer=candidate.answer,
                )
                for candidate in candidates
            ]
            return UiSearchResponse(total_results=len(results), results=results)

        if request.task_mode == "TRAKE":
            event_texts = [
                event.strip()
                for event in (request.trake_events or [request.query, request.event_description or ""])
                if event and event.strip()
            ]
            if len(event_texts) < 2:
                event_texts = []
            else:
                gap_phrases = request.gap_phrases
                if gap_phrases is None:
                    gap_phrases = [""] * (len(event_texts) - 1)
                if len(gap_phrases) != len(event_texts) - 1:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"expected {len(event_texts) - 1} gap_phrases for "
                            f"{len(event_texts)} TRAKE events"
                        ),
                    )
                try:
                    query = TrakeQuery(
                        events=tuple(
                            TrakeEventQuery(index, text)
                            for index, text in enumerate(event_texts, start=1)
                        ),
                        gaps=tuple(translate_gap_phrase(phrase) for phrase in gap_phrases),
                    )
                    trake_semantic_service = _resolve_trake_semantic_service(semantic_service)
                    chains = search_trake(
                        query,
                        alignment_strategy=_build_trake_alignment_strategy(
                            TRAKE_DEFAULT_ALIGNMENT_STRATEGY
                        ),
                        semantic_service=trake_semantic_service,
                        aic_registry=kis_aic_registry,
                        top_k=request.top_k,
                        pipeline_resolver=kis_pipeline_resolver,
                        query_translator=trake_query_translator,
                    )
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                results: list[UiKeyframeResult] = []
                for rank, chain in enumerate(chains, start=1):
                    first_frame = next((frame for frame in chain.frames if frame is not None), 0)
                    results.append(
                        _ui_result(
                            rank=rank,
                            video_id=chain.video_id,
                            frame_id=int(first_frame or 0),
                            score=chain.score,
                            timestamp=None,
                            keyframe_id=None,
                            image_path=None,
                            keyframe_dir=served_keyframes,
                            video_dir=served_videos,
                            frames=list(chain.frames),
                        )
                    )
                return UiSearchResponse(total_results=len(results), results=results)

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

        results = [
            _ui_result(
                rank=candidate.rank,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                score=candidate.score,
                timestamp=candidate.timestamp,
                keyframe_id=candidate.keyframe_id,
                image_path=None,
                keyframe_dir=served_keyframes,
                video_dir=served_videos,
            )
            for candidate in candidates
        ]
        return UiSearchResponse(total_results=len(results), results=results)

    @application.get("/api/v1/videos/{video_id}/summary", response_model=VideoSummaryResponse)
    def ui_video_summary(video_id: str) -> VideoSummaryResponse:
        served_keyframes = keyframe_dir if isinstance(keyframe_dir, Path) else None
        served_videos = video_dir if isinstance(video_dir, Path) else None
        if served_keyframes is None:
            raise HTTPException(status_code=503, detail="keyframe directory is unavailable")
        try:
            video_id = video_id.strip()
            if video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
                raise ValueError("video_id must be a single video name, not a path")
            summary = build_video_summary(
                video_id=video_id,
                keyframe_root=served_keyframes,
                mapping_csv=_summary_mapping_csv(video_id, served_keyframes, kis_aic_registry),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        results = []
        for item in summary.keyframes:
            thumbnail = _servable_keyframe_url(str(item.image_path), served_keyframes)
            if thumbnail is None:
                thumbnail = f"/keyframes/{summary.video_id}/{item.keyframe_id}.jpg"
            video_url = f"/videos/{summary.video_id}.mp4" if served_videos is not None else ""
            results.append(
                ApiVideoSummaryKeyframe(
                    keyframe_id=item.keyframe_id,
                    frame_id=item.frame_id,
                    timestamp=_format_timestamp(item.timestamp) if item.timestamp is not None else None,
                    timestamp_sec=item.timestamp,
                    thumbnail_url=thumbnail,
                    video_url=video_url,
                    mapped=item.frame_id is not None,
                )
            )
        return VideoSummaryResponse(
            video_id=summary.video_id,
            keyframe_count=summary.keyframe_count,
            mapped_count=summary.mapped_count,
            results=results,
        )

    @application.post("/api/v1/submit", response_model=UiSubmitResponse)
    def ui_submit(request: UiSubmitRequest) -> UiSubmitResponse:
        if request.task_mode == "Q&A":
            lines = [
                f'{item.video_id},{item.frame_id},"{(item.answer or "").replace(chr(34), chr(34) + chr(34))}"'
                for item in request.items
            ]
        elif request.task_mode == "TRAKE":
            frames = ",".join(str(item.frame_id) for item in request.items)
            video_id = request.items[0].video_id if request.items else ""
            lines = [f"{video_id},{frames}" if frames else video_id]
        else:
            lines = [f"{item.video_id},{item.frame_id}" for item in request.items]
        return UiSubmitResponse(
            query_id=request.query_id,
            task_mode=request.task_mode,
            row_count=len(request.items),
            csv_preview="\n".join(lines),
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

    @application.post("/search/trake", response_model=TrakeSearchResponse)
    def search_trake_route(request: TrakeSearchRequest) -> TrakeSearchResponse:
        """Qualifying-round TRAKE search: ranked candidate chains for an
        already-decomposed k-event query.

        Returns RAW chains, including unresolved (``None`` -> JSON ``null``)
        markers, as-is from ``search_trake()``. The never-blank fallback
        (Phase 07, ``apply_never_blank_fallback``) is deliberately NOT run
        here -- it is a submission-time concern for whichever tool actually
        produces the competition entry, not a retrieval-time concern, and
        applying it silently here would hide which markers were genuinely
        found versus fabricated.
        """
        if semantic_service is None:
            raise HTTPException(status_code=503, detail="trake semantic index is unavailable")
        try:
            query = TrakeQuery(
                events=tuple(
                    TrakeEventQuery(index, event.text)
                    for index, event in enumerate(request.events, start=1)
                ),
                gaps=tuple(translate_gap_phrase(phrase) for phrase in request.gap_phrases),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        strategy_name = request.alignment_strategy or TRAKE_DEFAULT_ALIGNMENT_STRATEGY
        strategy = _build_trake_alignment_strategy(strategy_name)

        try:
            trake_semantic_service = _resolve_trake_semantic_service(semantic_service)
            chains = search_trake(
                query,
                alignment_strategy=strategy,
                semantic_service=trake_semantic_service,
                aic_registry=kis_aic_registry,
                top_k=request.top_k,
                pipeline_resolver=kis_pipeline_resolver,
                query_translator=trake_query_translator,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return TrakeSearchResponse(
            event_count=len(request.events),
            alignment_strategy=strategy_name,
            count=len(chains),
            results=[
                TrakeChainHit(
                    rank=rank,
                    video_id=chain.video_id,
                    frames=list(chain.frames),
                    score=chain.score,
                )
                for rank, chain in enumerate(chains, start=1)
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
