"""Tests for canonical production system defaults.

Assert that the production config and defaults resolve to the canonical
stack and that no legacy provider is the default.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def production_config() -> dict:
    path = REPO_ROOT / "configs" / "production.yaml"
    assert path.is_file(), f"production.yaml not found at {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def default_config() -> dict:
    path = REPO_ROOT / "configs" / "default.yaml"
    assert path.is_file(), f"default.yaml not found at {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Ingestion defaults
# --------------------------------------------------------------------------

class TestIngestionDefaults:
    def test_detector_is_transnetv2(self, production_config):
        assert production_config["ingestion"]["detector"] == "transnetv2"

    def test_detector_threshold(self, production_config):
        assert production_config["ingestion"]["detector_threshold"] == 0.5

    def test_selector_is_periodic(self, production_config):
        assert production_config["ingestion"]["selector"] == "periodic"

    def test_long_threshold(self, production_config):
        assert production_config["ingestion"]["long_threshold_sec"] == 12.0

    def test_interval(self, production_config):
        assert production_config["ingestion"]["interval_sec"] == 8.0

    def test_no_default_pyscenedetect(self, production_config):
        assert production_config["ingestion"]["detector"] != "pyscenedetect"

    def test_default_yaml_aligned(self, default_config):
        assert default_config["ingestion"]["shot_detector"] == "transnetv2"


# --------------------------------------------------------------------------
# Object detection defaults
# --------------------------------------------------------------------------

class TestObjectDefaults:
    def test_provider_is_rfdetr(self, production_config):
        assert production_config["objects"]["provider"] == "rfdetr"

    def test_no_default_faster_rcnn(self, production_config):
        assert production_config["objects"]["provider"] != "faster_rcnn"

    def test_no_default_yolov8(self, production_config):
        assert production_config["objects"]["provider"] != "yolov8n"

    def test_pin_status_documented(self, production_config):
        assert "pin_status" in production_config["objects"]

    def test_faster_rcnn_is_legacy(self, production_config):
        legacy = production_config["objects"].get("legacy_backends", [])
        assert "faster_rcnn" in legacy

    def test_default_yaml_aligned(self, default_config):
        assert default_config["enrichment"]["object_detector"] == "rfdetr"


# --------------------------------------------------------------------------
# Embedding defaults
# --------------------------------------------------------------------------

class TestEmbeddingDefaults:
    def test_three_providers(self, production_config):
        providers = production_config["embeddings"]["providers"]
        assert len(providers) == 3

    def test_beit3_present(self, production_config):
        providers = production_config["embeddings"]["providers"]
        assert "beit3" in providers

    def test_metaclip2_present(self, production_config):
        providers = production_config["embeddings"]["providers"]
        assert "metaclip2" in providers

    def test_perception_encoder_present(self, production_config):
        providers = production_config["embeddings"]["providers"]
        assert "perception_encoder" in providers

    def test_no_single_clip_default(self, production_config):
        providers = production_config["embeddings"]["providers"]
        assert "clip" not in providers

    def test_default_yaml_aligned(self, default_config):
        providers = default_config["enrichment"]["embedding_providers"]
        assert set(providers) == {"beit3", "metaclip2", "perception_encoder"}


# --------------------------------------------------------------------------
# OCR defaults
# --------------------------------------------------------------------------

class TestOCRDefaults:
    def test_detector_is_deepsolo(self, production_config):
        assert production_config["ocr"]["detector"] == "deepsolo"

    def test_recognizer_is_parseq_vietnamese(self, production_config):
        assert production_config["ocr"]["recognizer"] == "parseq_vietnamese"

    def test_temporal_context(self, production_config):
        assert production_config["ocr"]["temporal_context"] == [-1, 0, 1]

    def test_no_default_easyocr(self, production_config):
        assert production_config["ocr"]["detector"] != "easyocr"

    def test_no_default_paddleocr(self, production_config):
        assert production_config["ocr"]["detector"] != "paddleocr"

    def test_default_yaml_aligned(self, default_config):
        assert default_config["enrichment"]["ocr_engine"] == "deepsolo"


# --------------------------------------------------------------------------
# ASR defaults
# --------------------------------------------------------------------------

class TestASRDefaults:
    def test_vad_is_silero(self, production_config):
        assert production_config["asr"]["vad"]["provider"] == "silero"

    def test_recognizer_is_chunkformer(self, production_config):
        assert production_config["asr"]["recognizer"] == "chunkformer_ctc_large_vie"

    def test_model_pinned(self, production_config):
        assert production_config["asr"]["model"] == "khanhld/chunkformer-ctc-large-vie"

    def test_revision_pinned(self, production_config):
        assert production_config["asr"]["revision"] is not None
        assert len(production_config["asr"]["revision"]) == 40  # SHA-1


# --------------------------------------------------------------------------
# RF-DETR provider module
# --------------------------------------------------------------------------

class TestRFDETRProvider:
    def test_rfdetr_module_exists(self):
        from mvr.enrichment.objects.rfdetr import RFDETRDetector, RFDETRConfig
        assert RFDETRDetector is not None
        assert RFDETRConfig is not None

    def test_default_config_unpinned(self):
        from mvr.enrichment.objects.rfdetr import RFDETRConfig
        config = RFDETRConfig()
        assert config.provider_id == "rfdetr"
        assert not config.is_pinned
        assert config.pin_status == "pending"

    def test_detect_fails_without_pin(self):
        from mvr.enrichment.objects.rfdetr import RFDETRDetector, RFDETRPinError
        detector = RFDETRDetector()
        with pytest.raises(RFDETRPinError):
            detector.detect("fake_image.jpg")

    def test_no_silent_fallback(self):
        """RF-DETR must not silently fall back to Faster R-CNN."""
        from mvr.enrichment.objects import PRODUCTION_BACKEND
        assert PRODUCTION_BACKEND == "rfdetr"


# --------------------------------------------------------------------------
# Production config schema
# --------------------------------------------------------------------------

class TestProductionConfigSchema:
    def test_schema_version(self, production_config):
        assert production_config.get("schema_version") == "1.0"

    def test_all_top_level_keys(self, production_config):
        required = {"ingestion", "objects", "embeddings", "ocr", "asr"}
        assert required.issubset(set(production_config.keys()))

    def test_defaults_doc_exists(self):
        path = REPO_ROOT / "docs" / "architecture" / "CANONICAL_SYSTEM_DEFAULTS.md"
        assert path.is_file(), "CANONICAL_SYSTEM_DEFAULTS.md must exist"
