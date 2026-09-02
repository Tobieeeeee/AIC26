"""The VM setup script must fail closed on a bad checkpoint download.

Every checkpoint it fetches is pinned by SHA256 in the repository's lock files.
A truncated or substituted download that reaches the destination path anyway
would be discovered much later -- BEiT-3 raises only after the run has already
pulled its keyframes, and RF-DETR would silently detect against the wrong
weights. The script therefore verifies into a temporary file and publishes the
destination only after the hash matches.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY_ROOT / "scripts" / "operations" / "setup_vm.sh"
BASH = shutil.which("bash")

CONTENT = b"real checkpoint bytes\n"
# sha256 of CONTENT, stated literally so the test never trusts the script's own
# hashing to decide whether the script hashed correctly.
DIGEST = "a09340f8b70d82243c08f5b44d47f70b3a57360ba612ff6144eb23ef82413b09"


def _tree(tmp_path: Path) -> Path:
    script_dir = tmp_path / "scripts" / "operations"
    script_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_dir / SCRIPT.name)
    return script_dir / SCRIPT.name


def _curl_stub(tmp_path: Path, *, payload: bytes, marker: Path | None = None) -> Path:
    """A curl that writes ``payload`` to the -o path and records each call."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "curl"
    record = f'printf "call\\n" >> "{marker}"' if marker else ""
    stub.write_text(
        "#!/bin/sh\n"
        "destination=\n"
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -o) destination="$2"; shift 2 ;;\n'
        '    *) shift ;;\n'
        "  esac\n"
        "done\n"
        f"{record}\n"
        f'printf %s {payload.decode().strip()!r} > "$destination"\n'
        'printf "\\n" >> "$destination"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub_dir


def _run(script: Path, stub_dir: Path) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    return subprocess.run(
        [BASH, str(script), "checkpoints"],
        capture_output=True,
        text=True,
        env=environment,
    )


def _checkpoint_table() -> dict[str, tuple[str, str, str]]:
    """Parse the script's pinned checkpoint table."""
    table: dict[str, tuple[str, str, str]] = {}
    inside = False
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        if line.startswith("CHECKPOINTS=("):
            inside = True
            continue
        if inside:
            if line.strip() == ")":
                break
            match = re.match(r'\s*"([^|]+)\|([^|]+)\|([^|]+)\|([^"]+)"', line)
            if match:
                name, url, digest, destination = match.groups()
                table[name] = (url, digest, destination)
    return table


@pytest.mark.skipif(BASH is None, reason="bash is required to execute the script")
def test_a_checksum_mismatch_leaves_no_file_at_the_destination(tmp_path: Path) -> None:
    script = _tree(tmp_path)
    stub_dir = _curl_stub(tmp_path, payload=b"corrupted download\n")

    result = _run(script, stub_dir)

    assert result.returncode != 0
    assert "checksum" in (result.stdout + result.stderr).lower()
    for _url, _digest, destination in _checkpoint_table().values():
        assert not (tmp_path / destination).exists()
        assert not (tmp_path / f"{destination}.partial").exists()


@pytest.mark.skipif(BASH is None, reason="bash is required to execute the script")
def test_an_already_verified_checkpoint_is_not_downloaded_again(tmp_path: Path) -> None:
    script = _tree(tmp_path)
    table = _checkpoint_table()
    assert table, "the script must declare its pinned checkpoints"

    # Publish every checkpoint with content the script accepts, so a second run
    # has nothing left to fetch. A curl stub that fails loudly proves it.
    source = script.read_text(encoding="utf-8")
    for _name, (_url, digest, destination) in table.items():
        target = tmp_path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(CONTENT)
        source = source.replace(digest, DIGEST)
    script.write_text(source, encoding="utf-8")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    forbidden = stub_dir / "curl"
    forbidden.write_text('#!/bin/sh\necho "download attempted" >&2\nexit 3\n', encoding="utf-8")
    forbidden.chmod(0o755)

    result = _run(script, stub_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "download attempted" not in result.stderr


def test_the_pinned_checkpoints_match_the_repository_lock_files() -> None:
    """The script may not drift from the locks the pipeline verifies against.

    ``rfdetr.py`` and ``Beit3ImageEncoder`` re-verify these same hashes at load
    time, so a script that downloads a different revision fails the run rather
    than the setup -- on every machine at once, after the images are built.
    """
    table = _checkpoint_table()

    rfdetr = json.loads(
        (REPOSITORY_ROOT / "configs" / "rfdetr_model.lock.json").read_text(
            encoding="utf-8"
        )
    )["production_model"]["checkpoint"]
    assert table["rfdetr"][0] == rfdetr["source"]
    assert table["rfdetr"][1] == rfdetr["sha256"]

    providers = {
        provider["model_id"]: provider
        for provider in json.loads(
            (REPOSITORY_ROOT / "configs" / "embedding_providers.json").read_text(
                encoding="utf-8"
            )
        )["providers"]
    }
    assert table["beit3"][1] == providers["beit3"]["checkpoint_sha256"]
    assert table["beit3_tokenizer"][1] == providers["beit3"]["tokenizer_sha256"]

    perception_encoder = providers["perception_encoder"]
    assert table["perception_encoder"][1] == perception_encoder["checkpoint_sha256"]
    assert perception_encoder["model_revision"] in table["perception_encoder"][0]


def test_the_runtime_check_installs_the_torch_six_shim_before_unilm() -> None:
    """The check has to follow the loader it is checking.

    ``utils.py`` in the pinned unilm checkout still does ``from torch._six
    import inf``, a module torch removed in 2.x. ``Beit3ImageEncoder`` handles
    that by calling ``_install_torch_six_inf_compat()`` before it imports
    modeling_finetune. A check that imports it bare raises ModuleNotFoundError
    on a machine where BEiT-3 would in fact load, which is worse than having no
    check at all: it sends someone debugging a healthy install.
    """
    script = SCRIPT.read_text(encoding="utf-8")

    assert "_install_torch_six_inf_compat" in script, (
        "the runtime check must install the shim rather than reimplement or "
        "skip it"
    )
    assert script.index("_install_torch_six_inf_compat") < script.index(
        'import_module("modeling_finetune")'
    )

