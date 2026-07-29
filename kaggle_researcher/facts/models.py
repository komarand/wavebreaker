from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class CodeObservation(BaseModel):
    name: str
    kwargs: dict[str, str]
    locator: str


class CompetitionMetadata(BaseModel):
    competition_id: str
    title: str | None = None
    metric_name: str | None = None
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
    sample_submission_source: Literal["api", "header_download", "unavailable"]
    limitations: list[str]


class NotebookFacts(BaseModel):
    ref: str
    title: str
    author: str | None = None
    votes: int
    public_score: float | None = None
    last_run: datetime | None = None

    ast_fingerprint: str
    lineage_cluster_id: str

    splitters: list[CodeObservation]
    models: list[CodeObservation]
    metrics: list[CodeObservation]
    feature_ops: list[CodeObservation]
    declared_cv: list[str]

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
    source: Literal["meta_kaggle", "api_final_only", "unavailable"]
    not_computable_reason: str | None = None


class CvLbPair(BaseModel):
    notebook_ref: str
    declared_cv: float
    public_score: float
    lineage_cluster_id: str


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

    user_constraints: UserConstraints
    collection_errors: list[str]
