from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from kaggle_researcher.research_scout.schemas import (
    EdaTaskPlanDraft,
    ResearchScoutOutput,
    ScoutEdaTask,
    ScoutHypothesis,
    ScoutLimitation,
    ScoutStructuredFinding,
)


def _load_legacy_research_scout() -> Any:
    legacy_path = Path(__file__).resolve().parent.parent / "research_scout.py"
    spec = importlib.util.spec_from_file_location(
        "kaggle_researcher._legacy_research_scout",
        legacy_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy research_scout module: {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_research_scout()
for _name in dir(_legacy):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_legacy, _name)


__all__ = [
    "EdaTaskPlanDraft",
    "ResearchScoutOutput",
    "ScoutEdaTask",
    "ScoutHypothesis",
    "ScoutLimitation",
    "ScoutStructuredFinding",
    *[
        name
        for name in dir(_legacy)
        if not name.startswith("_")
    ],
]
