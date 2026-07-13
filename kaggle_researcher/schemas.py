from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictBool, model_validator

from kaggle_researcher.contracts.normalization import normalize_contract_payload


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
    report_path: str | None = None
    num_documents: int
    num_sources: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    duration_sec: float
    mode: str = "full"
    report_mode: str = "full"
    run_artifacts_path: str | None = None
    retrieved_evidence_count: int = 0
    research_hypotheses_path: str | None = None
    eda_task_plan_path: str | None = None
    summary_path: str | None = None
    num_hypotheses: int = 0
    num_eda_tasks: int = 0
    eda_evidence_pack_path: str | None = None
    eda_summary_path: str | None = None
    final_strategy_path: str | None = None
    final_strategy_summary_path: str | None = None


class ReasoningBaseResult(BaseModel):
    confidence: ConfidenceLevel
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: Any) -> Any:
        return normalize_contract_payload(value, cls.__name__)


class ValidationPolicy(BaseModel):
    """A concrete validation policy; secondary policies may be absent when unjustified."""

    model_config = ConfigDict(extra="allow")

    method: str = Field(min_length=1)


class ValidationResult(ReasoningBaseResult):
    recommended_cv: str
    validation_risk: ConfidenceLevel
    likely_split: str
    failure_modes: list[str] = Field(default_factory=list)
    reasoning: str
    primary_validation: ValidationPolicy
    secondary_validation: ValidationPolicy | None = None
    do_not_use: list[str] = Field(default_factory=list)
    policy_enforced: StrictBool = False
    policy_notes: list[str] = Field(default_factory=list)

class LeakageRiskResult(ReasoningBaseResult):
    risk_level: ConfidenceLevel
    possible_issues: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)


class MetricResult(ReasoningBaseResult):
    metric_explanation: str
    needs_calibration: StrictBool
    rank_averaging_useful: StrictBool
    threshold_search_needed: StrictBool
    surrogate_loss_suggestion: str


class LeaderboardAuditResult(ReasoningBaseResult):
    shake_up_risk: ConfidenceLevel
    submission_selection_rule: str
    public_lb_trust: str
    warnings: list[str] = Field(default_factory=list)


class ExperimentItem(BaseModel):
    experiment_id: str | None = None
    priority: PriorityLevel
    experiment: str
    why: str
    cost: str
    expected_gain: str
    risk: str
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: Any) -> Any:
        return normalize_contract_payload(value, cls.__name__)


class ReviewResult(ReasoningBaseResult):
    confidence: ConfidenceLevel = "medium"
    unsupported_claims: list[str] = Field(default_factory=list)
    too_generic: list[str] = Field(default_factory=list)
    unnecessary_experiments: list[str] = Field(default_factory=list)
    approved_experiment_ids: list[str] = Field(default_factory=list)
    rejected_experiment_ids: list[str] = Field(default_factory=list)
    revised_sections: dict[str, Any] = Field(default_factory=dict)
