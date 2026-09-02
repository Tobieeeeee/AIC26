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


# rfdetr 1.9.3, the version configs/rfdetr_model.lock.json pins the production
# detector to, declares "transformers>=5.1.0,<6.0.0".
RFDETR_NEEDS_TRANSFORMERS = (5, 1)


def _setup_script_groups() -> set[str]:
    """The extras scripts/operations/setup_vm.sh installs in one command."""
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "operations" / "setup_vm.sh"
    ).read_text(encoding="utf-8")
    match = re.search(r"pip install -e '\.\[([^\]]+)\]'", script)
    assert match, "the setup script must install the project with extras"
    return {group.strip() for group in match.group(1).split(",")}


def _upper_bounds(requirement: str) -> list[tuple[int, ...]]:
    return [
        tuple(int(part) for part in raw.split(".") if part.isdigit())
        for raw in re.findall(r"<\s*([0-9][0-9.]*)", requirement)
    ]


def test_the_setup_groups_leave_room_for_the_pinned_detector() -> None:
    """A ceiling in one group must not exclude what another group's package needs.

    The enrichment runner detects objects and embeds images in one process, so
    rfdetr and BEiT-3 share an environment. BEiT-3's own image path never
    imports transformers -- only Beit3TextEncoder does, for the online query
    side -- so a ``<5`` ceiling inherited from a 2023 README bought nothing and
    made the whole install unresolvable::

        ERROR: ResolutionImpossible
        mvr-competition[...] depends on transformers<5 and >=4.39
        rfdetr 1.9.3 depends on transformers<6.0.0 and >=5.1.0
    """
    groups = _optional_dependencies()
    offenders: dict[str, str] = {}
    for group_name in _setup_script_groups():
        for requirement in groups.get(group_name, []):
            if _normalize(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement).group(1)) != "transformers":
                continue
            if any(bound <= RFDETR_NEEDS_TRANSFORMERS for bound in _upper_bounds(requirement)):
                offenders[group_name] = requirement

    assert offenders == {}, (
        "these groups cap transformers below the "
        f"{'.'.join(str(part) for part in RFDETR_NEEDS_TRANSFORMERS)} that the "
        f"pinned rfdetr requires, so the setup install cannot resolve: {offenders}"
    )
