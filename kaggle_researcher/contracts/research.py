from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.ids import HypothesisId, SourceClaimId
from kaggle_researcher.contracts.versions import CURRENT_SCHEMA_VERSION


HypothesisCategory = Literal[
    "schema", "metric", "validation", "leakage", "relationship", "drift",
    "baseline", "feature", "notebook", "leaderboard", "data_quality",
]
Confidence = Literal["low", "medium", "high"]
Priority = Literal["P0", "P1", "P2", "P3"]
HypothesisStatus = Literal["needs_eda", "supported_by_source", "analogous_only", "not_testable"]
ALLOWED_HYPOTHESIS_CATEGORIES = tuple(HypothesisCategory.__args__)


class ResearchHypothesis(ContractModel):
    hypothesis_id: HypothesisId
    category: HypothesisCategory
    claim: str = Field(min_length=1)
    rationale: str | None = None
    expected_eda_checks: list[str] = Field(default_factory=list)
    priority: Priority = "P1"
    confidence_before_eda: Confidence
    source_refs: list[SourceClaimId] = Field(default_factory=list)
    status: HypothesisStatus = "needs_eda"
    limitations: list[str] = Field(default_factory=list)


class ResearchHypotheses(ContractModel):
    contract_family: Literal["research_hypotheses"] = "research_hypotheses"
    schema_version: Literal["1.0"] = CURRENT_SCHEMA_VERSION
    competition_id: str = Field(min_length=1)
    created_at: str | None = None
    hypotheses: list[ResearchHypothesis] = Field(default_factory=list)
    eda_tasks: list[dict[str, Any]] = Field(default_factory=list)
    structured_findings: list[dict[str, Any]] = Field(default_factory=list)
    scout_limitations: list[str] = Field(default_factory=list)
    models_used: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ALLOWED_HYPOTHESIS_CATEGORIES",
    "Confidence",
    "HypothesisCategory",
    "HypothesisStatus",
    "Priority",
    "ResearchHypothesis",
    "ResearchHypotheses",
]
