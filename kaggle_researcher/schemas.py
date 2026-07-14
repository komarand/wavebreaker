from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

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
