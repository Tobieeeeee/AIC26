"""Optional dependency groups have to be installable together.

The enrichment runner builds a BEiT-3 encoder and a Perception Encoder inside
one process, so ``pip install -e '.[beit3,perception-encoder]'`` is the normal
way to prepare a machine. Two groups pinning one package to two exact versions
makes that unsatisfiable, and pip reports it only after minutes of backtracking
on a fresh machine -- long enough to look like a hang rather than a conflict.
"""
from __future__ import annotations

import re
import tomllib
from collections import defaultdict
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_EXACT_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s,;]+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _optional_dependencies() -> dict[str, list[str]]:
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return document["project"]["optional-dependencies"]


def _requirement_names(group: list[str]) -> set[str]:
    names = set()
    for requirement in group:
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match:
            names.add(_normalize(match.group(1)))
    return names


def test_no_two_groups_pin_one_package_to_different_versions() -> None:
    pins: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for group_name, requirements in _optional_dependencies().items():
        for requirement in requirements:
            match = _EXACT_PIN.match(requirement)
            if match:
                package, version = match.groups()
                pins[_normalize(package)][version].add(group_name)

    conflicts = {
        package: {version: sorted(groups) for version, groups in versions.items()}
        for package, versions in pins.items()
        if len(versions) > 1
    }

    assert conflicts == {}, (
        "optional dependency groups pin the same package to different exact "
        f"versions, so they cannot be installed together: {conflicts}"
    )


def test_the_torch_companion_libraries_follow_the_pinned_torch_version() -> None:
    """A compiled extension may not drift away from the torch it links against.

    torchaudio and torchvision ship native extensions built against one CUDA
    runtime. Declared as a range next to an exact ``torch`` pin, pip is free to
    resolve a build for a newer CUDA than that torch supplies, and nothing
    complains until an unrelated-looking import fails::

        import chunkformer -> torchaudio -> _torchaudio
        OSError: libcudart.so.13: cannot open shared object file
    """
    exact: dict[str, set[str]] = defaultdict(set)
    ranged: dict[str, set[str]] = defaultdict(set)
    for group_name, requirements in _optional_dependencies().items():
        for requirement in requirements:
            match = _EXACT_PIN.match(requirement)
            package = _normalize(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement).group(1))
            if package not in {"torch", "torchaudio", "torchvision"}:
                continue
            if match:
                exact[package].add(match.group(2))
            else:
                ranged[package].add(group_name)

    assert exact["torch"], "torch must carry one exact pin the companions can follow"
    torch_version = exact["torch"]
    assert len(torch_version) == 1, f"torch is pinned to several versions: {torch_version}"

    assert "torchaudio" not in ranged, (
        "torchaudio is declared as a range in "
        f"{sorted(ranged['torchaudio'])}, so pip may install a build for a "
        f"different CUDA runtime than torch {next(iter(torch_version))} provides"
    )


def test_the_beit3_group_covers_the_unilm_import_chain() -> None:
    """Everything ``import modeling_finetune`` reaches must be installable.

    ``Beit3ImageEncoder._load_official_runtime`` imports ``modeling_finetune``
    from the pinned microsoft/unilm checkout, which pulls in a chain the group
    never declared::

        modeling_finetune -> utils          -> torchmetrics, tensorboardX
        modeling_finetune -> modeling_utils -> torchscale.model.BEiT3
                                            -> torchscale.architecture.encoder
                                            -> fairscale.nn

    Each missing name turns an install-time problem into a runtime
    ``Beit3DependencyError`` raised after the run has already downloaded every
    keyframe. torchscale itself is deliberately absent: its own metadata pins
    ``timm==0.6.13``, which no resolver can reconcile with the 1.x that this
    project needs, so it is installed with ``--no-deps`` by the setup script
    while its real import-time requirements are declared here.
    """
    declared = _requirement_names(_optional_dependencies()["beit3"])

    assert {"torchmetrics", "tensorboardx", "fairscale"} <= declared
