from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MIN_LEADERBOARD_MATCH_FRACTION = 0.60


class CodeObservation(BaseModel):
    name: str
    kwargs: dict[str, str]
    locator: str


class DeclaredCvObservation(BaseModel):
    value: float
    metric_name: str | None = None
    locator: str
    raw_text: str


class ScoreObservation(BaseModel):
    value: float
    value_raw: str
    metric_raw: str | None = None
    metric_canonical: str | None = None
    locator: str
    raw_text: str
    source: Literal["markdown", "code", "code_string", "title", "ref"]


class CompetitionMetadata(BaseModel):
    competition_id: str
    title: str | None = None
    metric_name: str | None = None
    evaluation_metric_raw: str | None = None
    metric_status: Literal["available", "placeholder", "unavailable"] = "unavailable"
    is_code_competition: bool | None = None
    submissions_per_day: int | None = None
    max_team_size: int | None = None
    deadline: datetime | None = None
    reward: str | None = None
    category: str | None = None
    num_teams: int | None = None
    unavailable_fields: list[str]


class FileInfo(BaseModel):
    name: str
    size_bytes: int | None = None
    role_hint: Literal["train", "test", "submission", "auxiliary"]


class FileManifest(BaseModel):
    files: list[FileInfo]
    train_test_size_ratio: float | None = None
    sample_submission_columns: list[str]
    sample_submission_source: Literal["api", "full_download", "unavailable"]
    limitations: list[str]


class NotebookFacts(BaseModel):
    ref: str
    title: str
    author: str | None = None
    votes: int = 0
    public_score: float | None = None
    last_run: datetime | None = None

    ast_fingerprint: str
    lineage_cluster_id: str

    splitters: list[CodeObservation]
    models: list[CodeObservation]
    metrics: list[CodeObservation]
    feature_ops: list[CodeObservation]
    declared_cv: list[str]
    declared_cv_observations: list[DeclaredCvObservation] = Field(default_factory=list)
    score_observations: list[ScoreObservation] = Field(default_factory=list)
    score_candidates_seen: int = 0
    score_candidates_excluded: int = 0

    parse_status: Literal["ok", "partial", "failed"]


class DiscussionFacts(BaseModel):
    topic_id: str
    title: str
    author: str | None = None
    author_is_host: bool
    votes: int
    created_at: datetime | None = None
    source_type: Literal["discussion", "winner_writeup"]
    competition_id: str
    text: str


class LeaderboardStability(BaseModel):
    competition_id: str
    status: Literal["computed", "not_computable"]
    public_private_spearman: float | None = None
    top10_retention: float | None = None
    median_rank_change: float | None = None
    matched_teams: int
    match_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Literal["meta_kaggle", "api_final_only", "unavailable"]
    not_computable_reason: str | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_contract(self) -> LeaderboardStability:
        derived_metrics = (
            self.public_private_spearman,
            self.top10_retention,
            self.median_rank_change,
        )
        if self.status == "computed":
            if self.not_computable_reason:
                raise ValueError(
                    "computed leaderboard stability cannot have a "
                    "not_computable_reason"
                )
            if self.match_fraction is None:
                raise ValueError(
                    "computed leaderboard stability requires match_fraction"
                )
            if self.match_fraction < MIN_LEADERBOARD_MATCH_FRACTION:
                raise ValueError(
                    "computed leaderboard stability requires match_fraction of at "
                    f"least {MIN_LEADERBOARD_MATCH_FRACTION:.1%}"
                )
            if self.public_private_spearman is None or self.median_rank_change is None:
                raise ValueError(
                    "computed leaderboard stability requires Spearman and median rank "
                    "change metrics"
                )
        else:
            if not self.not_computable_reason or not self.not_computable_reason.strip():
                raise ValueError(
                    "not_computable leaderboard stability requires a reason"
                )
            if any(metric is not None for metric in derived_metrics):
                raise ValueError(
                    "not_computable leaderboard stability cannot contain derived metrics"
                )
        return self


class CvLbPair(BaseModel):
    notebook_ref: str
    declared_cv: float
    public_score: float
    lineage_cluster_id: str
    metric_name: str | None = None


class CvLbDiagnostics(BaseModel):
    notebooks_total: int = 0
    notebooks_with_public_score: int = 0
    notebooks_with_declared_cv: int = 0
    notebooks_with_both: int = 0
    comparable_pairs: int = 0
    rejected_non_comparable_pairs: int = 0
    zero_pairs_reason: str | None = None


class ScoreDiagnostics(BaseModel):
    notebooks_total: int = 0
    notebooks_with_score_observations: int = 0
    observations_total: int = 0
    observations_with_raw_metric: int = 0
    observations_with_canonical_metric: int = 0
    observations_without_canonical_metric: int = 0
    title_or_ref_observations: int = 0
    candidates_seen: int = 0
    candidates_excluded: int = 0


class UserConstraints(BaseModel):
    vram_gb: float | None = None
    hours_per_week: float | None = None
    cloud_budget_usd: float | None = None
    objective: Literal["medal", "top_percent", "learn", "fast_baseline"] = "medal"


class CompetitionFacts(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    collected_at: datetime

    metadata: CompetitionMetadata
    files: FileManifest
    notebooks: list[NotebookFacts]
    discussions: list[DiscussionFacts]
    similar_competitions: list[LeaderboardStability]
    cv_lb_pairs: list[CvLbPair]
    cv_lb_diagnostics: CvLbDiagnostics = Field(default_factory=CvLbDiagnostics)
    score_diagnostics: ScoreDiagnostics = Field(default_factory=ScoreDiagnostics)

    discussion_collection_status: Literal[
        "collected", "empty", "forbidden", "unavailable", "failed"
    ] = "empty"
    discussion_collection_error: str | None = None
    discussion_auth_mode: Literal["legacy", "oauth", "unknown"] = "unknown"
    limitations: list[str] = Field(default_factory=list)

    user_constraints: UserConstraints
    collection_errors: list[str]
