from __future__ import annotations

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.contract

_ROOT = Path("kaggle_researcher")
_LOW_LEVEL_BUILDERS = {
    Path("kaggle_researcher/contracts/evidence.py"),
    Path("kaggle_researcher/contracts/evidence_manifest.py"),
    Path("kaggle_researcher/contracts/references.py"),
}
_FORBIDDEN_CALLS = {
    "generate_allowed_evidence_refs",
    "generate_semantic_evidence_refs",
    "build_evidence_registry",
    "namespace_for",
}


def test_evidence_reference_space_is_not_recomputed_downstream() -> None:
    violations: list[str] = []
    for path in _ROOT.rglob("*.py"):
        relative = Path(path.as_posix())
        if relative in _LOW_LEVEL_BUILDERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) in _FORBIDDEN_CALLS:
                violations.append(f"{relative}:{node.lineno}:{_call_name(node.func)}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "evidence"
                and isinstance(node.value, ast.Name)
                and node.value.id == "registries"
            ):
                violations.append(f"{relative}:{node.lineno}:direct registries.evidence traversal")
    assert not violations, (
        "Downstream code must consume the immutable EvidenceReferenceManifest. "
        "Allowed exceptions are the low-level publication/compatibility definitions only: "
        + "; ".join(violations)
    )


def _call_name(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None
