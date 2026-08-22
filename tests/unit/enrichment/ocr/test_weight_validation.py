import pytest
import os
import subprocess
import hashlib
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def test_missing_deepsolo_weight(tmp_path):
    req_dir = tmp_path / "req"
    out_dir = tmp_path / "out"
    req_dir.mkdir()
    out_dir.mkdir()

    worker_script = Path("docker/ocr/worker/deepsolo_worker.py").resolve()
    missing_weight = tmp_path / "missing.pth"

    res = subprocess.run([
        "python", str(worker_script),
        "--requests-dir", str(req_dir),
        "--output-dir", str(out_dir),
        "--weights", str(missing_weight),
        "--config", "dummy.yaml",
        "--expected-sha256", "0" * 64,
    ], capture_output=True, text=True)

    assert res.returncode != 0
    assert "not found" in res.stderr

def test_empty_deepsolo_weight(tmp_path):
    req_dir = tmp_path / "req"
    out_dir = tmp_path / "out"
    req_dir.mkdir()
    out_dir.mkdir()

    worker_script = Path("docker/ocr/worker/deepsolo_worker.py").resolve()
    empty_weight = tmp_path / "empty.pth"
    empty_weight.write_bytes(b"")

    res = subprocess.run([
        "python", str(worker_script),
        "--requests-dir", str(req_dir),
        "--output-dir", str(out_dir),
        "--weights", str(empty_weight),
        "--config", "dummy.yaml",
        "--expected-sha256", "0" * 64,
    ], capture_output=True, text=True)

    assert res.returncode != 0
    assert "empty" in res.stderr

def test_mock_file_in_production_fails(tmp_path):
    req_dir = tmp_path / "req"
    out_dir = tmp_path / "out"
    req_dir.mkdir()
    out_dir.mkdir()

    worker_script = Path("docker/ocr/worker/deepsolo_worker.py").resolve()
    mock_weight = tmp_path / "mock.pth"
    mock_weight.write_bytes(b"dummy_weights_here")

    env = os.environ.copy()
    if "MVR_OCR_ALLOW_MOCK" in env:
        del env["MVR_OCR_ALLOW_MOCK"]

    res = subprocess.run([
        "python", str(worker_script),
        "--requests-dir", str(req_dir),
        "--output-dir", str(out_dir),
        "--weights", str(mock_weight),
        "--config", "dummy.yaml",
        "--expected-sha256", _sha256(mock_weight.read_bytes()),
    ], env=env, capture_output=True, text=True)

    assert res.returncode != 0
    assert "MVR_OCR_ALLOW_MOCK is not set to 1" in res.stderr

def test_explicit_mock_env_var(tmp_path):
    req_dir = tmp_path / "req"
    out_dir = tmp_path / "out"
    req_dir.mkdir()
    out_dir.mkdir()

    req_file = req_dir / "test.json"
    req_file.write_text('{"schema_version": "1.0", "artifact_type": "ocr_frame_request", "request_id": "1", "video_id": "V1", "shot_id": 1, "keyframe_id": "K1", "anchor_frame_idx": 1, "anchor_timestamp": 1.0, "source_frame_idx": 1, "source_timestamp": 1.0, "image_path": "a.jpg"}')

    worker_script = Path("docker/ocr/worker/deepsolo_worker.py").resolve()
    mock_weight = tmp_path / "mock.pth"
    mock_weight.write_bytes(b"dummy_weights_here")

    env = os.environ.copy()
    env["MVR_OCR_ALLOW_MOCK"] = "1"

    res = subprocess.run([
        "python", str(worker_script),
        "--requests-dir", str(req_dir),
        "--output-dir", str(out_dir),
        "--weights", str(mock_weight),
        "--config", "dummy.yaml",
        "--expected-sha256", _sha256(mock_weight.read_bytes()),
    ], env=env, capture_output=True, text=True)

    assert res.returncode == 0
    assert "DeepSolo worker finished" in res.stdout
    assert (out_dir / "test.json").exists()
