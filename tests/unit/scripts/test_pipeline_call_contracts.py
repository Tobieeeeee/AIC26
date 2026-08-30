"""Guard the seam between a pipeline script and the retrieval function it calls.

The two live in different files, so a signature change on one side merges
without a Git conflict and breaks only at runtime -- after the encoder and the
index have loaded, which in a competition is the worst possible moment to find
out. These tests read the call sites statically, in milliseconds, with no model.
"""
import ast
import inspect
from pathlib import Path

import pytest

from mvr.retrieval.qna import search_qna
from mvr.retrieval.trake import search_trake

REPO_ROOT = Path(__file__).resolve().parents[3]

CALL_SITES = [
    ("scripts/run_trake_pipeline.py", "search_trake", search_trake),
    ("scripts/run_qna_pipeline.py", "search_qna", search_qna),
]


def _keywords_passed(script_path: Path, function_name: str) -> set[str]:
    """Every keyword name the script hands to ``function_name``."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    keywords: set[str] = set()
    called = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != function_name:
            continue
        called = True
        for keyword in node.keywords:
            if keyword.arg is not None:  # ``**kwargs`` carries no name
                keywords.add(keyword.arg)
    if not called:
        raise AssertionError(f"{script_path.name} never calls {function_name}()")
    return keywords


@pytest.mark.parametrize(
    "relative_path,function_name,function",
    CALL_SITES,
    ids=[site[0] for site in CALL_SITES],
)
def test_pipeline_script_passes_only_keywords_the_function_accepts(
    relative_path: str, function_name: str, function
) -> None:
    passed = _keywords_passed(REPO_ROOT / relative_path, function_name)
    accepted = set(inspect.signature(function).parameters)

    unsupported = sorted(passed - accepted)
    assert not unsupported, (
        f"{relative_path} passes {unsupported} to {function_name}(), "
        f"which no longer accepts it; the script will raise TypeError at runtime"
    )
