from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

from kaggle_researcher.contracts.eda import (
    EdaEvidencePack,
    EdaHypothesisStatus,
    EdaRisk,
    HypothesisResult,
    RecommendedNextAction,
    RiskSeverity,
    RiskStatus,
    RiskType,
)
from kaggle_researcher.contracts.research_hypotheses import (
    ResearchHypothesis,
    ResearchHypotheses,
)
from kaggle_researcher.contracts.eda_task_plan import EdaTask, EdaTaskPlan
from kaggle_researcher.contracts.evidence_manifest import EvidenceConflictPolicy


Confidence = Literal["low", "medium", "high"]
Priority = Literal["P0", "P1", "P2", "P3"]
HypothesisStatus = EdaHypothesisStatus
LeakageCheckStatus = Literal[
    "passed",
    "failed",
    "warning",
    "not_testable",
    "skipped",
]
Severity = Literal["low", "medium", "high", "critical"]
HypothesisCategory = Literal[
    "schema",
    "metric",
    "validation",
    "leakage",
    "relationship",
    "drift",
    "baseline",
    "feature",
    "notebook",
    "leaderboard",
    "data_quality",
]
DatasetSource = Literal["local", "cache", "kaggle", "unknown"]
DatasetFileRole = Literal[
    "train",
    "test",
    "sample_submission",
    "metadata",
    "unknown",
]
DatasetTableHint = Literal[
    "base",
    "secondary",
    "depth_0",
    "depth_1",
    "depth_2",
    "submission",
    "unknown",
]
ColumnRoleName = Literal[
    "target",
    "primary_id",
    "group",
    "time",
    "date",
    "prediction",
    "numeric_feature",
    "categorical_feature",
    "unknown",
]
TableRole = Literal["train", "test", "submission", "metadata", "unknown"]
TableType = Literal["base", "secondary", "depth_0", "depth_1", "depth_2", "unknown"]
TieSensitivity = Literal["low", "medium", "high"]

EvidenceRef = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EdaRunConfig(BaseModel):
    competition_id: str
    competition_url: str | None = None

    hypotheses_path: Path
    task_plan_path: Path

    local_dataset_path: Path | None = None
    output_dir: Path | None = None

    download_dataset: bool = True
    force_download: bool = False

    modules: list[str] | None = None
    skip_modules: list[str] = Field(default_factory=list)

    profile_sample_rows: int = 200_000
    max_profile_rows_full_scan: int = 2_000_000
    max_adversarial_rows: int = 500_000
    max_baseline_rows: int = 1_000_000
    max_ablation_rows: int = 100_000
    max_ablations: int = 12
    max_ablation_runtime_sec: int | None = None
    ablation_n_folds: int = 5
    max_interaction_rows: int = 100_000
    max_interaction_numeric_columns: int = 30
    max_interaction_categorical_columns: int = 20
    max_interaction_pair_candidates: int = 300
    max_reported_interactions_per_type: int = 20
    interaction_min_group_rows: int = 20
    max_table_bytes: int = 512 * 1024 * 1024
    max_column_cardinality_scan_rows: int = 200_000
    module_timeout_sec: int = 900

    enable_p1_modules: bool = False
    enable_baseline: bool = False
    enable_baseline_ablations: bool = False
    enable_interaction_diagnostics: bool = False
    enable_source_claim_validation: bool = False
    enable_visual_diagnostics: bool = False
    enable_slice_diagnostics: bool = False
    enable_notebook_static_analysis: bool = False

    random_seed: int = 42
    fail_fast: bool = False
    evidence_conflict_policy: EvidenceConflictPolicy = EvidenceConflictPolicy.STRICT


class EdaRunResult(BaseModel):
    competition_id: str
    run_id: str
    output_dir: Path

    evidence_pack_path: Path
    evidence_manifest_path: Path | None = None
    published_bundle_path: Path | None = None
    summary_path: Path

    module_statuses: dict[str, str] = Field(default_factory=dict)
    hypothesis_results_count: int = 0

    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    duration_sec: float


class DatasetInfo(BaseModel):
    competition_id: str
    competition_url: str | None = None
    dataset_path: str | None = None
    source: DatasetSource = "unknown"
    download_requested: bool = False
    local_dataset_path: str | None = None
    cache_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class DatasetFile(BaseModel):
    path: str
    name: str
    extension: str
    size_bytes: int
    size_mb: float

    role_hint: DatasetFileRole = "unknown"
    table_hint: DatasetTableHint = "unknown"

    can_read: bool
    read_error: str | None = None


class FileInventoryResult(BaseModel):
    dataset_path: str
    files: list[DatasetFile] = Field(default_factory=list)

    detected_formats: dict[str, int] = Field(default_factory=dict)
    table_roles: dict[str, str] = Field(default_factory=dict)

    train_files: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    sample_submission_files: list[str] = Field(default_factory=list)
    metadata_files: list[str] = Field(default_factory=list)

    missing_train_test_pairs: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_format_pairs: list[dict[str, Any]] = Field(default_factory=list)
    suspicious_files: list[dict[str, Any]] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)


class ColumnRole(BaseModel):
    name: str
    role: ColumnRoleName
    confidence: Confidence
    reason: str


class TableSchema(BaseModel):
    table_name: str
    path: str

    role: TableRole = "unknown"
    table_type: TableType = "unknown"

    n_columns: int
    columns: list[dict[str, Any]] = Field(default_factory=list)
    column_roles: list[ColumnRole] = Field(default_factory=list)

    candidate_join_keys: list[str] = Field(default_factory=list)
    candidate_time_columns: list[str] = Field(default_factory=list)
    candidate_date_columns: list[str] = Field(default_factory=list)

    confidence: Confidence
    warnings: list[str] = Field(default_factory=list)


class InferredSchema(BaseModel):
    global_roles: dict[str, Any] = Field(default_factory=dict)

    tables: list[TableSchema] = Field(default_factory=list)

    target_column: str | None = None
    primary_id_column: str | None = None
    prediction_column: str | None = None

    train_base_table: str | None = None
    test_base_table: str | None = None
    sample_submission_table: str | None = None

    candidate_time_columns: list[str] = Field(default_factory=list)
    candidate_group_columns: list[str] = Field(default_factory=list)
    candidate_date_columns: list[str] = Field(default_factory=list)

    confidence: Confidence
    warnings: list[str] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    name: str
    dtype: str

    missing_count: int | None = None
    missing_pct: float | None = None

    n_unique: int | None = None
    unique_ratio: float | None = None

    mean: float | None = None
    std: float | None = None
    min: float | int | str | None = None
    max: float | int | str | None = None
    q01: float | None = None
    q05: float | None = None
    q50: float | None = None
    q95: float | None = None
    q99: float | None = None

    top_values: list[dict[str, Any]] = Field(default_factory=list)
    date_min: str | None = None
    date_max: str | None = None

    is_constant: bool = False
    is_mostly_missing: bool = False
    is_high_cardinality: bool = False


class TableProfile(BaseModel):
    table_name: str
    path: str

    n_rows: int | None = None
    n_cols: int

    sampled: bool = False
    sample_rows: int | None = None

    columns: list[ColumnProfile] = Field(default_factory=list)

    mostly_missing_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)


class MetricEvidence(BaseModel):
    metric_name: str
    normalized_metric_name: str | None = None
    task_type: str | None = None
    metric_family: str | None = None
    base_metric: str | None = None
    greater_is_better: bool | None = None

    requires_probabilities: bool | None = None
    requires_threshold: bool | None = None
    requires_calibration: bool | None = None
    requires_groups: bool | None = None
    requires_time: bool | None = None
    requires_query_groups: bool | None = None
    rank_based: bool | None = None
    requires_time_or_groups: bool | None = None
    local_metric_available: bool = False
    needs_custom_implementation: bool = False

    threshold_search_needed: bool | None = None
    prediction_output_type: str | None = None
    tie_sensitivity: TieSensitivity | None = None

    components: dict[str, Any] = Field(default_factory=dict)
    required_columns: dict[str, Any] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)


class ValidationEvidence(BaseModel):
    target_available: bool = False
    id_column_available: bool = False
    target_column: str | None = None
    id_column: str | None = None

    time_columns: list[dict[str, Any]] = Field(default_factory=list)
    group_columns: list[dict[str, Any]] = Field(default_factory=list)
    query_columns: list[dict[str, Any]] = Field(default_factory=list)

    class_balance: dict[str, Any] = Field(default_factory=dict)
    target_summary: dict[str, Any] = Field(default_factory=dict)
    target_by_period: list[dict[str, Any]] = Field(default_factory=list)
    target_by_group: list[dict[str, Any]] = Field(default_factory=list)

    test_time_relation: dict[str, Any] = Field(default_factory=dict)
    test_group_relation: dict[str, Any] = Field(default_factory=dict)
    oot_holdout: dict[str, Any] = Field(default_factory=dict)
    temporal_folds: dict[str, Any] = Field(default_factory=dict)

    primary_validation: dict[str, Any] = Field(default_factory=dict)
    diagnostic_validations: list[dict[str, Any]] = Field(default_factory=list)
    rejected_validations: list[dict[str, Any]] = Field(default_factory=list)
    recommended_validation: dict[str, Any] = Field(default_factory=dict)
    confidence: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None

    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class LeakageCheckResult(BaseModel):
    check_id: str
    status: LeakageCheckStatus
    severity: Severity

    finding: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    related_hypothesis_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def competition_ids_match(
    hypotheses: ResearchHypotheses,
    task_plan: EdaTaskPlan,
) -> bool:
    """Return whether Scout inputs target the same competition."""

    return hypotheses.competition_id == task_plan.competition_id


__all__ = [
    "ColumnProfile",
    "ColumnRole",
    "Confidence",
    "DatasetFile",
    "DatasetInfo",
    "EdaEvidencePack",
    "EdaRisk",
    "EdaRunConfig",
    "EdaRunResult",
    "EdaTask",
    "EdaTaskPlan",
    "EvidenceRef",
    "FileInventoryResult",
    "HypothesisResult",
    "HypothesisStatus",
    "LeakageCheckResult",
    "LeakageCheckStatus",
    "MetricEvidence",
    "Priority",
    "RecommendedNextAction",
    "ResearchHypotheses",
    "ResearchHypothesis",
    "Severity",
    "TableProfile",
    "TableSchema",
    "ValidationEvidence",
    "competition_ids_match",
]
