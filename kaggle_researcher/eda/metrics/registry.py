from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kaggle_researcher.eda.presets import CompetitionPreset


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    SURVIVAL = "survival"
    FORECASTING_TABULAR = "forecasting_tabular"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    UNKNOWN = "unknown"

    TIME_SERIES = "forecasting_tabular"


class MetricFamily(str, Enum):
    RANK_CLASSIFICATION = "rank_classification"
    PROBABILISTIC_CLASSIFICATION = "probabilistic_classification"
    THRESHOLD_CLASSIFICATION = "threshold_classification"
    REGRESSION_ERROR = "regression_error"
    ORDINAL_CLASSIFICATION = "ordinal_classification"
    RANKING = "ranking"
    SURVIVAL = "survival"
    TEMPORAL_STABILITY = "temporal_stability"
    CUSTOM = "custom"
    UNKNOWN = "unknown"

    CLASSIFICATION = "threshold_classification"
    REGRESSION = "regression_error"
    ORDINAL = "ordinal_classification"
    RECOMMENDER_RANKING = "ranking"


@dataclass(slots=True)
class MetricSpec:
    name: str
    family: MetricFamily
    aliases: list[str] = field(default_factory=list)
    task_types: list[TaskType] = field(default_factory=lambda: [TaskType.UNKNOWN])
    greater_is_better: bool | None = None
    requires_probabilities: bool = False
    requires_threshold: bool = False
    requires_calibration: bool = False
    requires_groups: bool = False
    requires_time: bool = False
    requires_query_groups: bool = False
    supports_local_eval: bool = True
    needs_custom_implementation: bool = False
    notes: list[str] = field(default_factory=list)

    rank_based: bool | None = None
    threshold_search_needed: bool | None = None
    local_metric_available: bool | None = None
    requires_groups_or_time: bool | None = None

    def __post_init__(self) -> None:
        self.family = _coerce_metric_family(self.family)
        self.task_types = [_coerce_task_type(task_type) for task_type in self.task_types]
        self.aliases = [str(alias) for alias in self.aliases]
        if isinstance(self.notes, str):
            self.notes = [self.notes]
        self.notes = [str(note) for note in self.notes]

        if self.rank_based is None:
            self.rank_based = self.family in {
                MetricFamily.RANK_CLASSIFICATION,
                MetricFamily.RANKING,
                MetricFamily.SURVIVAL,
                MetricFamily.TEMPORAL_STABILITY,
            }
        if self.threshold_search_needed is None:
            self.threshold_search_needed = self.requires_threshold
        if self.local_metric_available is None:
            self.local_metric_available = self.supports_local_eval
        else:
            self.supports_local_eval = self.local_metric_available
        if self.requires_groups_or_time is None:
            self.requires_groups_or_time = self.requires_groups or self.requires_time


class MetricRegistry:
    def __init__(self, specs: list[MetricSpec] | tuple[MetricSpec, ...] | None = None) -> None:
        self._specs_by_name: dict[str, MetricSpec] = {}
        for spec in specs or _DEFAULT_METRIC_SPECS:
            self.register(spec)

    def register(self, spec: MetricSpec) -> None:
        for name in (spec.name, *spec.aliases):
            self._specs_by_name[_normalize_metric_name(name)] = spec

    def get(self, metric_name: str | None) -> MetricSpec | None:
        if metric_name is None:
            return None
        return self._specs_by_name.get(_normalize_metric_name(metric_name))


def infer_metric_spec(
    metric_name: str | None,
    task_type: TaskType | str | None = None,
    preset: CompetitionPreset | None = None,
) -> MetricSpec:
    canonical_metric_name = (
        preset.canonical_metric_name(metric_name)
        if preset is not None
        else metric_name
    )
    normalized_metric_name = _normalize_metric_name(canonical_metric_name or "unknown")
    if normalized_metric_name in {"custom", "custom_metric"}:
        return _custom_metric_spec(canonical_metric_name or "custom", task_type)
    if normalized_metric_name == "unknown":
        return _unknown_metric_spec(canonical_metric_name or "unknown", task_type)

    spec = MetricRegistry().get(canonical_metric_name)
    if spec is not None:
        return spec

    return _unknown_metric_spec(canonical_metric_name or "unknown", task_type)


def normalize_metric_name(metric_name: str | None) -> str:
    return _normalize_metric_name(metric_name or "unknown")


def _custom_metric_spec(metric_name: str, task_type: TaskType | str | None) -> MetricSpec:
    return MetricSpec(
        name=metric_name,
        family=MetricFamily.CUSTOM,
        task_types=[_coerce_task_type(task_type)],
        greater_is_better=None,
        supports_local_eval=False,
        needs_custom_implementation=True,
        notes=["Custom metric requires competition-specific implementation."],
    )


def _unknown_metric_spec(metric_name: str, task_type: TaskType | str | None) -> MetricSpec:
    return MetricSpec(
        name=metric_name,
        family=MetricFamily.UNKNOWN,
        task_types=[_coerce_task_type(task_type)],
        greater_is_better=None,
        supports_local_eval=False,
        needs_custom_implementation=True,
        notes=["Unknown metric is not available in the local metric registry."],
    )


def _coerce_task_type(task_type: TaskType | str | None) -> TaskType:
    if isinstance(task_type, TaskType):
        return task_type
    if task_type is None:
        return TaskType.UNKNOWN
    normalized = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "time_series":
        normalized = "forecasting_tabular"
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.UNKNOWN


def _coerce_metric_family(metric_family: MetricFamily | str) -> MetricFamily:
    if isinstance(metric_family, MetricFamily):
        return metric_family
    normalized = str(metric_family).strip().lower().replace("-", "_").replace(" ", "_")
    legacy_aliases = {
        "classification": "threshold_classification",
        "regression": "regression_error",
        "ordinal": "ordinal_classification",
        "recommender_ranking": "ranking",
    }
    normalized = legacy_aliases.get(normalized, normalized)
    try:
        return MetricFamily(normalized)
    except ValueError:
        return MetricFamily.UNKNOWN


def _normalize_metric_name(metric_name: str) -> str:
    normalized = str(metric_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith("ndcg@"):
        return "ndcg"
    if normalized.startswith("map@"):
        return "map@k"
    if normalized.startswith("recall@"):
        return "recall@k"
    aliases = {
        "mean_absolute_percentage_error": "mape",
        "symmetric_mean_absolute_percentage_error": "smape",
        "normalized_gini_coefficient": "normalized_gini",
    }
    return aliases.get(normalized, normalized)


_CLASSIFICATION_TASKS = [
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
]
_REGRESSION_TASKS = [TaskType.REGRESSION, TaskType.FORECASTING_TABULAR]


_DEFAULT_METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="auc",
        aliases=["roc_auc"],
        family=MetricFamily.RANK_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        requires_probabilities=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="gini",
        aliases=["normalized_gini", "normalized_gini_coefficient"],
        family=MetricFamily.RANK_CLASSIFICATION,
        task_types=[TaskType.BINARY_CLASSIFICATION],
        greater_is_better=True,
        requires_probabilities=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="logloss",
        aliases=["log_loss", "cross_entropy"],
        family=MetricFamily.PROBABILISTIC_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=False,
        requires_probabilities=True,
        requires_calibration=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="accuracy",
        family=MetricFamily.THRESHOLD_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        requires_threshold=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="f1",
        aliases=["macro_f1", "f1_macro"],
        family=MetricFamily.THRESHOLD_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        requires_threshold=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="precision",
        family=MetricFamily.THRESHOLD_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        requires_threshold=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="recall",
        family=MetricFamily.THRESHOLD_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        requires_threshold=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="quadratic_weighted_kappa",
        aliases=["qwk", "cohen_kappa", "cohen_kappa_quadratic"],
        family=MetricFamily.ORDINAL_CLASSIFICATION,
        task_types=[TaskType.MULTICLASS_CLASSIFICATION],
        greater_is_better=True,
        requires_threshold=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="rmse",
        aliases=["root_mean_squared_error"],
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="mse",
        aliases=["mean_squared_error"],
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="mae",
        aliases=["mean_absolute_error"],
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="rmsle",
        aliases=["root_mean_squared_log_error"],
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="mape",
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="smape",
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="r2",
        aliases=["r_squared", "coefficient_of_determination"],
        family=MetricFamily.REGRESSION_ERROR,
        task_types=_REGRESSION_TASKS,
        greater_is_better=True,
        supports_local_eval=True,
    ),
    MetricSpec(
        name="map@k",
        aliases=["mapk", "mean_average_precision_at_k"],
        family=MetricFamily.RANKING,
        task_types=[TaskType.RANKING],
        greater_is_better=True,
        requires_query_groups=True,
        supports_local_eval=False,
    ),
    MetricSpec(
        name="ndcg",
        aliases=["ndcg@k", "normalized_discounted_cumulative_gain"],
        family=MetricFamily.RANKING,
        task_types=[TaskType.RANKING],
        greater_is_better=True,
        requires_query_groups=True,
        supports_local_eval=False,
    ),
    MetricSpec(
        name="recall@k",
        aliases=["recallk"],
        family=MetricFamily.RANKING,
        task_types=[TaskType.RANKING],
        greater_is_better=True,
        requires_query_groups=True,
        supports_local_eval=False,
    ),
    MetricSpec(
        name="concordance_index",
        aliases=["c_index", "concordance"],
        family=MetricFamily.SURVIVAL,
        task_types=[TaskType.SURVIVAL],
        greater_is_better=True,
        supports_local_eval=False,
    ),
    MetricSpec(
        name="gini_stability",
        family=MetricFamily.TEMPORAL_STABILITY,
        task_types=[TaskType.BINARY_CLASSIFICATION],
        greater_is_better=True,
        requires_probabilities=True,
        requires_time=True,
        supports_local_eval=True,
        notes=["Stability variant of normalized Gini; requires period evidence."],
    ),
)


__all__ = [
    "MetricFamily",
    "MetricRegistry",
    "MetricSpec",
    "TaskType",
    "infer_metric_spec",
    "normalize_metric_name",
]
