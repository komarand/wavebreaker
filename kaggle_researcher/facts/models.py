from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MIN_LEADERBOARD_MATCH_FRACTION = 0.60
ScoreSplit = Literal["cv", "lb", "unknown"]
OptimizationDirection = Literal["maximize", "minimize"]
MetricCanonicalSource = Literal["alias", "competition_hint", "none"]
CvLbOptimizationDirection = Literal[
    "maximize", "minimize", "higher_is_better", "lower_is_better"
]


class CodeObservation(BaseModel):
    name: str
    kwargs: dict[str, str]
    kwargs_resolved_from: dict[str, Literal["direct", "dict_literal"]] = Field(
        default_factory=dict
    )
    locator: str


class KwargDistribution(BaseModel):
    key: str
    cluster_count: int
    median: str
    minimum: str | None = None
    maximum: str | None = None
    distinct_values: int
    is_integer: bool


class CodeFamilyUsage(BaseModel):
    name: str
    cluster_count: int
    notebook_count: int
    cluster_share: float
    kwargs_distribution: list[KwargDistribution] = Field(default_factory=list)
    kwargs_basis: Literal["cluster_median_of_public_notebooks"] = (
        "cluster_median_of_public_notebooks"
    )
    best_public_score: float | None = None


class ModelCombination(BaseModel):
    names: list[str]
    cluster_count: int


class CodeAggregates(BaseModel):
    total_clusters: int
    total_notebooks: int
    models: list[CodeFamilyUsage]
    splitters: list[CodeFamilyUsage]
    metrics: list[CodeFamilyUsage]
    feature_ops: list[CodeFamilyUsage]
    model_combinations: list[ModelCombination]


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
    metric_canonical_source: MetricCanonicalSource = "none"
    locator: str
    raw_text: str
    source: Literal["markdown", "code", "code_string", "title", "ref"]
    split: ScoreSplit = "unknown"
    split_signals: list[str] = Field(default_factory=list)
    observation_id: str | None = None
    optimization_direction: OptimizationDirection | None = None
    source_kind: str | None = None
    context_text: str = ""
    context_signals: list[str] = Field(default_factory=list)
    plausible: bool = True
    implausible_reason: str | None = None


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
    sample_submission_status: Literal[
        "api",
        "full_download",
        "file_not_found",
        "size_unknown",
        "size_over_limit",
        "download_forbidden",
        "download_failed",
        "header_unreadable",
    ] = "file_not_found"
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
    style_markup_stripped_cells: int = 0
    style_markup_stripped_markdown_cells: int = 0
    style_markup_stripped_code_strings: int = 0
    dataset_paths: list[str] = Field(default_factory=list)

    parse_status: Literal["ok", "partial", "failed"]


class DiscussionLink(BaseModel):
    url: str
    text: str | None = None
    kind: Literal["kaggle", "external", "relative"]
    competition_slug: str | None = None


class DiscussionMessageFacts(BaseModel):
    evidence_id: str
    message_id: str
    topic_id: str
    author_name: str | None = None
    author_slug: str | None = None
    author_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    votes: int | None = None
    content_html: str | None = None
    content_text: str = ""
    content_sha256: str
    content_truncated: bool = False
    content_original_length: int = 0
    links: list[DiscussionLink] = Field(default_factory=list)


class DiscussionFacts(BaseModel):
    topic_id: str
    title: str
    author: str | None = None
    author_is_host: bool | None = None
    votes: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    source_type: Literal["discussion", "winner_writeup"]
    competition_id: str
    text: str = ""
    evidence_id: str | None = None
    url: str | None = None
    url_constructed: bool = False
    comment_count: int | None = None
    is_writeup_candidate: bool = False
    writeup_signals: list[str] = Field(default_factory=list)
    messages: list[DiscussionMessageFacts] = Field(default_factory=list)
    collection_status: Literal["collected", "empty", "rate_limited", "forbidden", "failed"] = (
        "collected"
    )
    collection_error: str | None = None


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
                    "computed leaderboard stability cannot have a " "not_computable_reason"
                )
            if self.match_fraction is None:
                raise ValueError("computed leaderboard stability requires match_fraction")
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
                raise ValueError("not_computable leaderboard stability requires a reason")
            if any(metric is not None for metric in derived_metrics):
                raise ValueError(
                    "not_computable leaderboard stability cannot contain derived metrics"
                )
        return self


class SimilarityEvidence(BaseModel):
    same_metric: bool | None = None
    same_slug_family: bool | None = None
    same_submission_shape: bool | None = None
    same_code_competition: bool | None = None
    same_category: bool | None = None
    matched_features: list[str] = Field(default_factory=list)
    metric_self: str | None = None
    metric_candidate: str | None = None


class SimilarCompetition(BaseModel):
    slug: str
    title: str | None = None
    discovered_by: Literal["manual", "discussion_mention"]
    confirmed: bool = False
    verification: Literal["verified", "rejected", "not_found", "unchecked"] = (
        "unchecked"
    )
    evidence: SimilarityEvidence = Field(default_factory=SimilarityEvidence)
    match_strength: Literal["family", "metric_and_family", "metric_only"] | None = None
    rejection_reason: str | None = None
    evidence_topic_ids: list[str] = Field(default_factory=list)
    mention_topic_count: int = 0
    mention_total: int = 0


class SimilarSearchDiagnostics(BaseModel):
    status: Literal["found_by_mention", "no_candidates", "all_rejected"]
    candidates_seen: int
    verified: int
    rejected: int
    not_found: int
    metadata_lookups: int


class LeaderboardEntry(BaseModel):
    team_name: str | None
    score: float | None
    rank: int | None


class LeaderboardShape(BaseModel):
    entry_count: int
    top_score: float | None
    score_at_rank: dict[int, float]
    median_adjacent_delta: float | None
    tied_adjacent_pairs: int = 0
    nonzero_adjacent_pairs: int = 0
    tied_ratio: float | None = None
    teams_within_median_delta_of_median: int | None
    plateau_ratio: float | None
    span_top_to_last: float | None
    direction: Literal[
        "higher_is_better",
        "lower_is_better",
        "unknown",
    ]


class PublicLeaderboard(BaseModel):
    status: Literal["collected", "unavailable"]
    entries: list[LeaderboardEntry]
    entry_count: int
    unavailable_reason: str | None
    shape: LeaderboardShape | None = None


class DatasetReference(BaseModel):
    slug: str
    raw_path: str
    notebook_refs: list[str]
    lineage_cluster_ids: list[str]
    reference_count: int
    cluster_count: int


class LeaderboardMatch(BaseModel):
    notebook_ref: str
    team_name: str
    score: float
    match_confidence: Literal["exact", "partial"]


class CvLbPair(BaseModel):
    notebook_ref: str
    declared_cv: float
    public_score: float
    lineage_cluster_id: str
    metric_name: str | None = None
    metric_raw: str | None = None
    metric_canonical: str | None = None
    metric_match: Literal["exact", "assumed"] = "exact"
    optimization_direction: CvLbOptimizationDirection | None = None
    cv_score: float | None = None
    lb_score: float | None = None
    cv_observation_ids: list[str] = Field(default_factory=list)
    lb_observation_ids: list[str] = Field(default_factory=list)
    cv_representative_observation_id: str | None = None
    lb_representative_observation_id: str | None = None
    cv_source: str = "declared_cv_legacy"
    lb_source: Literal["observation", "leaderboard_match"] = "observation"
    cv_aggregation: str = "legacy"
    lb_aggregation: str = "single"
    cv_selection_reason: str | None = None
    lb_selection_reason: str | None = None
    gap: float | None = None
    absolute_gap: float | None = None
    comparability_status: str = "comparable"
    comparability_reason: str | None = None


class CvLbDiagnostics(BaseModel):
    notebooks_total: int = 0
    notebooks_with_public_score: int = 0
    notebooks_with_declared_cv: int = 0
    notebooks_with_both: int = 0
    comparable_pairs: int = 0
    rejected_non_comparable_pairs: int = 0
    zero_pairs_reason: str | None = None
    notebooks_with_cv_scores: int = 0
    notebooks_with_lb_scores: int = 0
    notebooks_with_both_sides: int = 0
    pairs_created: int = 0
    pairs_created_from_api_lb: int = 0
    pairs_created_from_observation_lb: int = 0
    pairs_created_from_leaderboard_match: int = 0
    pairs_rejected_missing_cv: int = 0
    pairs_rejected_missing_lb: int = 0
    pairs_rejected_metric_mismatch: int = 0
    pairs_rejected_scale_mismatch: int = 0
    rejected_implausible_gap: int = 0
    pairs_rejected_implausible_gap: int = 0
    leaderboard_pairs_rejected_implausible_gap: int = 0
    pairs_rejected_ambiguous_metric: int = 0
    pairs_rejected_ambiguous_split: int = 0
    fold_series_aggregated: int = 0
    unknown_direction_fallbacks: int = 0


class ScoreDiagnostics(BaseModel):
    notebooks_total: int = 0
    notebooks_with_score_observations: int = 0
    observations_total: int = 0
    observations_with_raw_metric: int = 0
    observations_with_canonical_metric: int = 0
    observations_without_canonical_metric: int = 0
    canonical_by_alias: int = 0
    canonical_by_competition_hint: int = 0
    style_markup_stripped_cells: int = 0
    style_markup_stripped_markdown_cells: int = 0
    style_markup_stripped_code_strings: int = 0
    title_or_ref_observations: int = 0
    candidates_seen: int = 0
    candidates_excluded: int = 0
    split_cv: int = 0
    split_lb: int = 0
    split_unknown: int = 0
    notebooks_with_cv_scores: int = 0
    notebooks_with_lb_scores: int = 0
    notebooks_with_both_sides: int = 0
    implausible_observations: dict[str, int] = Field(default_factory=dict)
    implausible_top_labels: dict[str, int] = Field(default_factory=dict)
    notebooks_failed_by_status: dict[int, int] = Field(default_factory=dict)
    notebooks_failed_by_exception: dict[str, int] = Field(default_factory=dict)


class UserConstraints(BaseModel):
    vram_gb: float | None = None
    hours_per_week: float | None = None
    cloud_budget_usd: float | None = None
    objective: Literal["medal", "top_percent", "learn", "fast_baseline"] | None = None


class CompetitionFacts(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    collected_at: datetime

    metadata: CompetitionMetadata
    files: FileManifest
    notebooks: list[NotebookFacts]
    code_aggregates: CodeAggregates | None = None
    public_leaderboard: PublicLeaderboard = Field(
        default_factory=lambda: PublicLeaderboard(
            status="unavailable",
            entries=[],
            entry_count=0,
            unavailable_reason="Public leaderboard was not collected.",
        )
    )
    leaderboard_matches: list[LeaderboardMatch] = Field(default_factory=list)
    dataset_references: list[DatasetReference] = Field(default_factory=list)
    discussions: list[DiscussionFacts]
    similar_competitions: list[LeaderboardStability]
    similar_candidates: list[SimilarCompetition] = Field(default_factory=list)
    similar_diagnostics: SimilarSearchDiagnostics | None = None
    cv_lb_pairs: list[CvLbPair]
    implausible_gap_pairs: list[CvLbPair] = Field(default_factory=list)
    cv_lb_diagnostics: CvLbDiagnostics = Field(default_factory=CvLbDiagnostics)
    score_diagnostics: ScoreDiagnostics = Field(default_factory=ScoreDiagnostics)

    discussion_collection_status: Literal[
        "collected",
        "partial",
        "empty",
        "rate_limited",
        "forbidden",
        "unavailable",
        "failed",
    ] = "empty"
    discussion_collection_error: str | None = None
    discussion_auth_mode: Literal["legacy", "oauth", "unknown"] = "unknown"
    limitations: list[str] = Field(default_factory=list)

    user_constraints: UserConstraints
    collection_errors: list[str]
