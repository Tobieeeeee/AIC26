import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from mvr.enrichment.ocr.model_integrity import (
    ModelIntegrityError,
    fetch_models,
    load_model_lock,
    model_manifest_provenance,
    validate_lock_document,
    verify_model_directory,
    verify_model_file,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(
    *,
    filename: str,
    directory: str,
    sha256: str,
    acquisition: dict | None = None,
) -> dict:
    return {
        "model": "Synthetic model",
        "variant": "test",
        "purpose": "test",
        "checkpoint_identity": filename,
        "source_repo": "https://example.invalid/repository",
        "source_revision": "a" * 40,
        "download_source": "https://example.invalid/artifact",
        "license": "test-only",
        "filename": filename,
        "dir": directory,
        "runtime_path": f"/models/{directory}/{filename}",
        "runtime_config": "test-config",
        "sha256": sha256,
        "acquisition": acquisition or {"type": "file"},
    }


def _lock_document(deepsolo: dict, parseq_vi: dict) -> dict:
    return {
        "schema_version": "1.0",
        "models": {"deepsolo": deepsolo, "parseq_vi": parseq_vi},
    }


def _write_lock(path: Path, deepsolo: dict, parseq_vi: dict) -> None:
    path.write_text(
        json.dumps(_lock_document(deepsolo, parseq_vi)), encoding="utf-8"
    )


def test_correct_sha256_is_accepted(tmp_path):
    payload = b"trusted checkpoint bytes"
    artifact = tmp_path / "model.pth"
    artifact.write_bytes(payload)

    assert verify_model_file(artifact, _sha256(payload), label="model") == _sha256(
        payload
    )


def test_wrong_sha256_is_rejected(tmp_path):
    artifact = tmp_path / "model.pth"
    artifact.write_bytes(b"wrong bytes")

    with pytest.raises(ModelIntegrityError, match="SHA256 mismatch"):
        verify_model_file(artifact, "0" * 64, label="model")


def test_existing_correct_artifact_is_reused_without_download(tmp_path):
    payload = b"existing trusted bytes"
    lock_path = tmp_path / "lock.json"
    weights_root = tmp_path / "models"
    deepsolo = _entry(filename="detector.pth", directory="detector", sha256=_sha256(payload))
    parseq = _entry(filename="recognizer.ckpt", directory="recognizer", sha256=_sha256(payload))
    _write_lock(lock_path, deepsolo, parseq)
    for entry in (deepsolo, parseq):
        destination = weights_root / entry["dir"] / entry["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    def unexpected_download(url: str, destination: Path) -> None:
        raise AssertionError("downloader must not be called")

    result = fetch_models(lock_path, weights_root, downloader=unexpected_download)

    assert set(result) == {"deepsolo", "parseq_vi"}


def test_existing_corrupt_artifact_fails_closed_and_is_not_overwritten(tmp_path):
    trusted = b"trusted bytes"
    corrupt = b"corrupt bytes"
    lock_path = tmp_path / "lock.json"
    weights_root = tmp_path / "models"
    entry = _entry(filename="model.pth", directory="model", sha256=_sha256(trusted))
    _write_lock(lock_path, entry, entry)
    destination = weights_root / "model" / "model.pth"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(corrupt)

    with pytest.raises(ModelIntegrityError, match="SHA256 mismatch"):
        fetch_models(lock_path, weights_root, downloader=lambda *_: None)

    assert destination.read_bytes() == corrupt


def test_missing_hash_is_rejected_by_lock_schema():
    entry = _entry(filename="model.pth", directory="model", sha256="0" * 64)
    del entry["sha256"]

    with pytest.raises(ModelIntegrityError, match="sha256"):
        validate_lock_document(_lock_document(entry, entry))


def test_failed_download_cleans_temp_and_never_publishes(tmp_path):
    trusted = b"trusted bytes"
    lock_path = tmp_path / "lock.json"
    weights_root = tmp_path / "models"
    entry_a = _entry(filename="a.pth", directory="a", sha256=_sha256(trusted))
    entry_b = _entry(filename="b.ckpt", directory="b", sha256=_sha256(trusted))
    _write_lock(lock_path, entry_a, entry_b)

    def corrupt_download(url: str, destination: Path) -> None:
        destination.write_bytes(b"wrong bytes")

    with pytest.raises(ModelIntegrityError, match="SHA256 mismatch"):
        fetch_models(lock_path, weights_root, downloader=corrupt_download)

    assert not (weights_root / "a" / "a.pth").exists()
    assert list((weights_root / "a").glob("*.tmp")) == []


def test_atomic_publish_happens_only_after_verification(tmp_path, monkeypatch):
    payload = b"verified before publish"
    lock_path = tmp_path / "lock.json"
    weights_root = tmp_path / "models"
    entry_a = _entry(filename="a.pth", directory="a", sha256=_sha256(payload))
    entry_b = _entry(filename="b.ckpt", directory="b", sha256=_sha256(payload))
    _write_lock(lock_path, entry_a, entry_b)
    replacements = []

    from mvr.enrichment.ocr import model_integrity

    real_replace = model_integrity.os.replace

    def recording_replace(source, destination):
        assert verify_model_file(Path(source), _sha256(payload), label="pre-publish")
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(model_integrity.os, "replace", recording_replace)

    def download(url: str, destination: Path) -> None:
        destination.write_bytes(payload)

    fetch_models(lock_path, weights_root, downloader=download)

    assert len(replacements) == 2
    assert (weights_root / "a" / "a.pth").read_bytes() == payload
    assert (weights_root / "b" / "b.ckpt").read_bytes() == payload


def test_zip_bundle_and_exact_member_are_both_verified(tmp_path):
    checkpoint = b"Vietnamese PARSeq checkpoint"
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("weights/rec/best-parseq.ckpt", checkpoint)
    bundle_bytes = bundle.read_bytes()

    lock_path = tmp_path / "lock.json"
    weights_root = tmp_path / "models"
    direct = _entry(filename="detector.pth", directory="detector", sha256=_sha256(b"detector"))
    parseq = _entry(
        filename="best-parseq.ckpt",
        directory="parseq_vi",
        sha256=_sha256(checkpoint),
        acquisition={
            "type": "zip_member",
            "bundle_sha256": _sha256(bundle_bytes),
            "member": "weights/rec/best-parseq.ckpt",
        },
    )
    _write_lock(lock_path, direct, parseq)

    def download(url: str, destination: Path) -> None:
        destination.write_bytes(bundle_bytes if "parseq" in url else b"detector")

    # Distinguish the synthetic URLs without weakening production parsing.
    document = json.loads(lock_path.read_text(encoding="utf-8"))
    document["models"]["parseq_vi"]["download_source"] += "/parseq"
    lock_path.write_text(json.dumps(document), encoding="utf-8")

    fetch_models(lock_path, weights_root, downloader=download)

    assert (weights_root / "parseq_vi" / "best-parseq.ckpt").read_bytes() == checkpoint


def test_runtime_preflight_rejects_wrong_bytes(tmp_path):
    trusted = b"trusted"
    lock_path = tmp_path / "lock.json"
    weights_root = tmp_path / "models"
    detector = _entry(filename="detector.pth", directory="detector", sha256=_sha256(trusted))
    recognizer = _entry(filename="recognizer.ckpt", directory="recognizer", sha256=_sha256(trusted))
    _write_lock(lock_path, detector, recognizer)
    for entry, payload in ((detector, trusted), (recognizer, b"wrong")):
        path = weights_root / entry["dir"] / entry["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    with pytest.raises(ModelIntegrityError, match="parseq_vi: SHA256 mismatch"):
        verify_model_directory(lock_path, weights_root)


def test_production_lock_schema_and_sources_are_pinned():
    models = load_model_lock(Path("docker/ocr/model_sources.lock.json"))

    assert models["deepsolo"]["filename"] == (
        "tt_res50_finetune_synth-tt-mlt-13-15-textocr.pth"
    )
    assert models["deepsolo"]["source_revision"] == (
        "dbadae995035246bad3376c7a44c015c69e9b313"
    )
    assert models["parseq_vi"]["source_revision"] == (
        "76cc5f3cc6268457aac764653400fdff681f8271"
    )
    serialized = json.dumps(models)
    assert "Bao-Minh" not in serialized
    assert "deepsolo_res50_finetune_synth.pth" not in serialized


def test_manifest_provenance_contains_exact_model_identity():
    models = load_model_lock(Path("docker/ocr/model_sources.lock.json"))

    detector = model_manifest_provenance(models["deepsolo"])
    recognizer = model_manifest_provenance(models["parseq_vi"])

    assert detector["checkpoint_identity"].endswith("textocr.pth")
    assert detector["sha256"] == "6342945f34781ea90340d11b089c091327115d381fceb62c87b5cd2a82efeed3"
    assert recognizer["checkpoint_identity"] == "weights/rec/best-parseq.ckpt from weights.zip"
    assert recognizer["sha256"] == "8089b13c5ad115a96a608c6401eaab36b081393ea0a8323537b29a2dc80168f5"
