from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator

from kaggle_researcher.contracts.experiments import ExperimentItem
from kaggle_researcher.contracts.review import ReviewResult
from kaggle_researcher.contracts.validation import (
    ConfidenceLevel,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ReasoningBaseResult,
    ValidationPolicy,
    ValidationResult,
)
from kaggle_researcher.workflow import FinalSynthesisStageStatus, WorkflowStatus


SourceType = Literal[
    "kaggle",
    "arxiv",
    "papers_with_code",
    "papers_with_code_legacy",
    "huggingface_papers",
    "github",
]


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
    workflow_status: WorkflowStatus = "success"
    degraded_stages: list[str] = Field(default_factory=list)
    report_path: str | None = None
    num_documents: int
    num_sources: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
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
    final_synthesis_diagnostics_path: str | None = None
    final_synthesis_status: Literal[
        "llm_success", "repaired_success", "degraded_fallback"
    ] | None = None
    final_synthesis_degraded: bool = False
    final_synthesis_stage_status: FinalSynthesisStageStatus | None = None
    source_cache_report_path: str | None = None
    num_new_sources: int = 0
    num_reused_sources: int = 0
    num_changed_sources: int = 0
    num_reused_embeddings: int = 0
    num_computed_embeddings: int = 0

    @model_validator(mode="after")
    def validate_workflow_synthesis_state(self) -> "ResearchRunResult":
        is_degraded = self.final_synthesis_status == "degraded_fallback"
        if self.final_synthesis_degraded != is_degraded:
            raise ValueError(
                "final_synthesis_degraded must match final_synthesis_status"
            )
        if is_degraded:
            if self.workflow_status == "success":
                raise ValueError(
                    "degraded final synthesis cannot have workflow_status='success'"
                )
            if "final_synthesis" not in self.degraded_stages:
                raise ValueError(
                    "degraded final synthesis must list final_synthesis in degraded_stages"
                )
            expected_stage_status = (
                "failed" if self.workflow_status == "failed" else "degraded_fallback"
            )
            if self.final_synthesis_stage_status != expected_stage_status:
                raise ValueError(
                    "final synthesis stage status contradicts workflow_status"
                )
        return self
