"""The RTX 5060 Ti wrapper must refuse before it spends R2 transfer.

A missing [operations] extra is invisible until the first Parquet
consolidation, which happens after the run has already downloaded every
keyframe and raw MP4. The wrapper therefore has to check it up front.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = REPOSITORY_ROOT / "scripts" / "operations" / "run_enrichment_rtx5060ti.sh"
BASH = shutil.which("bash")


def _tree(tmp_path: Path, *, import_exit_code: int) -> Path:
    script_dir = tmp_path / "scripts" / "operations"
    script_dir.mkdir(parents=True)
    shutil.copy2(WRAPPER, script_dir / WRAPPER.name)
    (tmp_path / ".env").write_text("R2_BUCKET=fake\n", encoding="utf-8")
    python_shim = tmp_path / ".venv" / "bin" / "python"
    python_shim.parent.mkdir(parents=True)
    python_shim.write_text(f"#!/bin/sh\nexit {import_exit_code}\n", encoding="utf-8")
    python_shim.chmod(0o755)
    return script_dir / WRAPPER.name


def _run(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(script), "L21", "run-id", str(script.parent / "work")],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_wrapper_refuses_when_the_operations_extra_cannot_be_imported(
    tmp_path: Path,
) -> None:
    result = _run(_tree(tmp_path, import_exit_code=1))

    assert result.returncode == 2
    assert "operations" in result.stderr


@pytest.mark.skipif(BASH is None, reason="bash is not available")
def test_wrapper_passes_the_operations_check_when_the_imports_succeed(
    tmp_path: Path,
) -> None:
    result = _run(_tree(tmp_path, import_exit_code=0))

    assert "pip install -e" not in result.stderr
    assert "Missing Docker" in result.stderr  # reached a later preflight
