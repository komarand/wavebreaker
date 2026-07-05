from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


SourceType = Literal[
    "kaggle",
    "arxiv",
    "papers_with_code",
    "papers_with_code_legacy",
    "huggingface_papers",
    "github",
]
ConfidenceLevel = Literal["low", "medium", "high"]
PriorityLevel = Literal["P0", "P1", "P2", "P3"]


class SourceDocument(BaseModel):
    id: str
    competition_id: str
    source: SourceType
    title: str
    url: HttpUrl | None = None
    content: str
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    id: str
    competition_id: str
    source: SourceType
    title: str
    url: HttpUrl | None = None
    content: str
    score: float
    rrf_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanData(BaseModel):
    task_type: str
    metric: str
    domain: str
    kaggle_queries: list[str] = Field(default_factory=list)
    arxiv_queries: list[str] = Field(default_factory=list)
    github_queries: list[str] = Field(default_factory=list)
    key_techniques: list[str] = Field(default_factory=list)
    similar_competitions: list[str] = Field(default_factory=list)


class ResearchRunResult(BaseModel):
    competition_id: str
    report_path: str
    num_documents: int
    num_sources: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    duration_sec: float
    report_mode: str = "full"
    run_artifacts_path: str | None = None
    retrieved_evidence_count: int = 0


class ReasoningBaseResult(BaseModel):
    confidence: ConfidenceLevel
    evidence_ids: list[str] = Field(default_factory=list)


class ValidationResult(ReasoningBaseResult):
    recommended_cv: str
    validation_risk: ConfidenceLevel
    likely_split: str
    failure_modes: list[str] = Field(default_factory=list)
    reasoning: str
    primary_validation: dict[str, Any] = Field(default_factory=dict)
    secondary_validation: dict[str, Any] = Field(default_factory=dict)
    do_not_use: list[str] = Field(default_factory=list)
    policy_enforced: bool = False
    policy_notes: list[str] = Field(default_factory=list)


class LeakageRiskResult(ReasoningBaseResult):
    risk_level: ConfidenceLevel
    possible_issues: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)


class MetricResult(ReasoningBaseResult):
    metric_explanation: str
    needs_calibration: bool
    rank_averaging_useful: bool
    threshold_search_needed: bool
    surrogate_loss_suggestion: str


class LeaderboardAuditResult(ReasoningBaseResult):
    shake_up_risk: ConfidenceLevel
    submission_selection_rule: str
    public_lb_trust: str
    warnings: list[str] = Field(default_factory=list)


class ExperimentItem(BaseModel):
    priority: PriorityLevel
    experiment: str
    why: str
    cost: str
    expected_gain: str
    risk: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReviewResult(ReasoningBaseResult):
    unsupported_claims: list[str] = Field(default_factory=list)
    too_generic: list[str] = Field(default_factory=list)
    unnecessary_experiments: list[str] = Field(default_factory=list)
    revised_sections: dict[str, str] = Field(default_factory=dict)
