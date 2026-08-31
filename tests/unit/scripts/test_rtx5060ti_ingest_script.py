from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts"
    / "operations"
    / "run_ingest_rtx5060ti.sh"
)


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_rtx5060ti_runtime_is_blackwell_compatible_and_pinned() -> None:
    text = _script_text()

    assert 'readonly TORCH_VERSION="2.7.1"' in text
    assert 'readonly TORCHVISION_VERSION="0.22.1"' in text
    assert "download.pytorch.org/whl/cu128" in text
    assert 'readonly TRANSNET_VERSION="1.0.5"' in text
    assert 'torch.cuda.get_device_capability(0) == (12, 0)' in text
    assert '"sm_120" in torch.cuda.get_arch_list()' in text
    assert 'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in text
    assert '"${PYTHON}" -m pip check' in text
    assert "CUDA_MATMUL_OK" in text


def test_rtx5060ti_setup_avoids_incompatible_metaclip_extra() -> None:
    text = _script_text()

    assert '"${PYTHON}" -m pip install -e ".[operations]"' in text
    assert '"transformers==4.57.1"' in text
    assert '"Pillow==11.0.0"' in text
    assert not re.search(r"pip install[^\n]*\[[^\]]*metaclip2", text)


def test_rtx5060ti_runner_preserves_ingestion_and_r2_safety_contracts() -> None:
    text = _script_text()

    assert 'die "--collections is required to prevent two machines' in text
    assert 'die "--run-id is required for stable R2 staging and resume"' in text
    assert '"${PYTHON}" -m mvr.cli r2 list' in text
    assert "--ingest-only" in text
    assert "--device cuda" in text
    assert "--evict-artifacts-after-push" in text
    assert 'nohup "${RUNNER_CMD[@]}"' in text
    assert not re.search(r"(?m)^\s*(?:source|\.)\s+[^\n]*\.env", text)


@pytest.mark.skipif(sys.platform == "win32", reason="CI/Linux shell syntax gate")
def test_rtx5060ti_script_passes_bash_syntax_check() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")

    result = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
