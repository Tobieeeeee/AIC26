"""Pinned OCR model acquisition and runtime integrity verification."""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Mapping


LOCK_SCHEMA_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_HEADERS = (b"dummy", b"<!doctype", b"<html")
DownloadFunction = Callable[[str, Path], None]


class ModelIntegrityError(RuntimeError):
    """Raised when model provenance or bytes fail closed validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(entry: Mapping[str, object], field: str, model_key: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ModelIntegrityError(f"Model {model_key!r} has missing or invalid {field!r}")
    return value


def _validate_relative_component(value: str, field: str, model_key: str) -> None:
    path = Path(value)
    if path.is_absolute() or value in {".", ".."} or len(path.parts) != 1:
        raise ModelIntegrityError(
            f"Model {model_key!r} {field!r} must be one safe path component"
        )


def validate_lock_document(document: object) -> dict[str, dict[str, object]]:
    if not isinstance(document, dict):
        raise ModelIntegrityError("Model lock must be a JSON object")
    if document.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ModelIntegrityError(
            f"Model lock schema_version must be {LOCK_SCHEMA_VERSION!r}"
        )

    models = document.get("models")
    if not isinstance(models, dict) or set(models) != {"deepsolo", "parseq_vi"}:
        raise ModelIntegrityError(
            "Production model lock must contain exactly deepsolo and parseq_vi"
        )

    validated: dict[str, dict[str, object]] = {}
    required = (
        "model",
        "variant",
        "purpose",
        "checkpoint_identity",
        "source_repo",
        "source_revision",
        "download_source",
        "license",
        "filename",
        "dir",
        "runtime_path",
        "runtime_config",
        "sha256",
    )
    for model_key, raw_entry in models.items():
        if not isinstance(raw_entry, dict):
            raise ModelIntegrityError(f"Model {model_key!r} entry must be an object")
        entry = dict(raw_entry)
        for field in required:
            _require_text(entry, field, model_key)

        filename = str(entry["filename"])
        directory = str(entry["dir"])
        _validate_relative_component(filename, "filename", model_key)
        _validate_relative_component(directory, "dir", model_key)
        expected_runtime = f"/models/{directory}/{filename}"
        if entry["runtime_path"] != expected_runtime:
            raise ModelIntegrityError(
                f"Model {model_key!r} runtime_path must be {expected_runtime!r}"
            )

        expected_sha256 = str(entry["sha256"]).lower()
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise ModelIntegrityError(
                f"Model {model_key!r} sha256 must be 64 lowercase hexadecimal characters"
            )
        entry["sha256"] = expected_sha256

        acquisition = entry.get("acquisition")
        if not isinstance(acquisition, dict):
            raise ModelIntegrityError(f"Model {model_key!r} acquisition must be an object")
        acquisition_type = acquisition.get("type")
        if acquisition_type == "file":
            pass
        elif acquisition_type == "zip_member":
            member = _require_text(acquisition, "member", model_key)
            bundle_sha256 = _require_text(acquisition, "bundle_sha256", model_key).lower()
            if not _SHA256_RE.fullmatch(bundle_sha256):
                raise ModelIntegrityError(
                    f"Model {model_key!r} bundle_sha256 must be 64 lowercase hexadecimal characters"
                )
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise ModelIntegrityError(f"Model {model_key!r} has unsafe ZIP member path")
            acquisition = dict(acquisition)
            acquisition["bundle_sha256"] = bundle_sha256
            entry["acquisition"] = acquisition
        else:
            raise ModelIntegrityError(
                f"Model {model_key!r} has unsupported acquisition type {acquisition_type!r}"
            )
        validated[model_key] = entry
    return validated


def load_model_lock(path: Path) -> dict[str, dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(f"Cannot read model lock {path}: {exc}") from exc
    return validate_lock_document(document)


def verify_model_file(path: Path, expected_sha256: str, *, label: str) -> str:
    expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ModelIntegrityError(f"{label}: missing or malformed expected SHA256")
    if not path.exists() or not path.is_file():
        raise ModelIntegrityError(f"{label}: model file not found: {path}")
    if path.stat().st_size == 0:
        raise ModelIntegrityError(f"{label}: model file is empty: {path}")
    with path.open("rb") as stream:
        header = stream.read(32).lower()
    if any(marker in header for marker in _PLACEHOLDER_HEADERS):
        raise ModelIntegrityError(f"{label}: placeholder or HTML response detected: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"{label}: SHA256 mismatch for {path}; expected {expected}, got {actual}"
        )
    return actual


def verify_model_directory(
    lock_path: Path, weights_root: Path
) -> dict[str, dict[str, object]]:
    models = load_model_lock(lock_path)
    for model_key, entry in models.items():
        host_path = weights_root / str(entry["dir"]) / str(entry["filename"])
        verify_model_file(host_path, str(entry["sha256"]), label=model_key)
        entry["host_path"] = str(host_path.resolve())
    return models


def download_url(url: str, destination: Path) -> None:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 MVR-OCR-Model-Fetch/1.0"},
    )
    with opener.open(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def _new_temp_path(directory: Path, filename: str, role: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{filename}.{role}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    return Path(raw_path)


def fetch_models(
    lock_path: Path,
    weights_root: Path,
    *,
    downloader: DownloadFunction = download_url,
) -> dict[str, Path]:
    models = load_model_lock(lock_path)
    published: dict[str, Path] = {}

    for model_key, entry in models.items():
        model_dir = weights_root / str(entry["dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        destination = model_dir / str(entry["filename"])
        expected_sha256 = str(entry["sha256"])

        if destination.exists():
            verify_model_file(destination, expected_sha256, label=model_key)
            published[model_key] = destination
            continue

        download_temp = _new_temp_path(model_dir, destination.name, "download")
        artifact_temp: Path | None = None
        try:
            downloader(str(entry["download_source"]), download_temp)
            acquisition = entry["acquisition"]
            if not isinstance(acquisition, dict):  # guarded by lock validation
                raise AssertionError("validated acquisition must be an object")

            if acquisition["type"] == "file":
                artifact_temp = download_temp
            else:
                verify_model_file(
                    download_temp,
                    str(acquisition["bundle_sha256"]),
                    label=f"{model_key} distribution bundle",
                )
                artifact_temp = _new_temp_path(model_dir, destination.name, "artifact")
                try:
                    with zipfile.ZipFile(download_temp) as archive:
                        member = str(acquisition["member"])
                        with archive.open(member) as source, artifact_temp.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                except (KeyError, OSError, zipfile.BadZipFile) as exc:
                    raise ModelIntegrityError(
                        f"{model_key}: cannot extract locked ZIP member: {exc}"
                    ) from exc

            verify_model_file(artifact_temp, expected_sha256, label=model_key)
            os.replace(artifact_temp, destination)
            artifact_temp = None
            published[model_key] = destination
        finally:
            for temporary in (artifact_temp, download_temp):
                if temporary is not None and temporary.exists():
                    temporary.unlink()

    return published


def model_manifest_provenance(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": entry["model"],
        "variant": entry["variant"],
        "purpose": entry["purpose"],
        "checkpoint_identity": entry["checkpoint_identity"],
        "sha256": entry["sha256"],
        "source_repo": entry["source_repo"],
        "source_revision": entry["source_revision"],
        "runtime_filename": entry["filename"],
        "runtime_path": entry["runtime_path"],
        "runtime_config": entry["runtime_config"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--lock", type=Path, required=True)
    fetch_parser.add_argument("--weights-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--lock", type=Path, required=True)
    verify_parser.add_argument("--weights-root", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "fetch":
            paths = fetch_models(args.lock, args.weights_root)
        else:
            models = verify_model_directory(args.lock, args.weights_root)
            paths = {key: Path(str(entry["host_path"])) for key, entry in models.items()}
    except ModelIntegrityError as exc:
        parser.error(str(exc))
    for model_key, path in paths.items():
        print(f"{model_key}: verified {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
