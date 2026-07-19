from __future__ import annotations

import re
from typing import Any


STATIC_ANALYZER_VERSION = "1.0"


def analyze_notebook_text(text: str) -> dict[str, Any]:
    """Inspect extracted notebook text without importing or executing downloaded code."""
    return _analyze_text(text, source_kind="notebook")


def analyze_repository_text(text: str) -> dict[str, Any]:
    """Inspect selected repository text without cloning or executing repository code."""
    return _analyze_text(text, source_kind="repository")


def _analyze_text(text: str, *, source_kind: str) -> dict[str, Any]:
    bounded = text[:100_000]
    imports = sorted(set(re.findall(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", bounded)))
    detectors = {
        "cross_validation": bool(re.search(r"\b(?:KFold|StratifiedKFold|GroupKFold|cross_val)\b", bounded, re.I)),
        "early_stopping": bool(re.search(r"\bearly[_ -]?stopping\b", bounded, re.I)),
        "feature_importance": bool(re.search(r"\b(?:feature_importance|shap)\b", bounded, re.I)),
        "target_encoding": bool(re.search(r"\btarget[_ -]?encod", bounded, re.I)),
    }
    return {
        "schema_version": "1.0",
        "analyzer_version": STATIC_ANALYZER_VERSION,
        "source_kind": source_kind,
        "imports": imports[:100],
        "detectors": detectors,
        "input_truncated": len(text) > len(bounded),
        "executed": False,
    }
