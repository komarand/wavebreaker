from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kaggle_researcher.eda.presets import CompetitionPreset


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    SURVIVAL = "survival"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"


class MetricFamily(str, Enum):
    RANKING = "ranking"
    PROBABILISTIC_CLASSIFICATION = "probabilistic_classification"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ORDINAL = "ordinal"
    RECOMMENDER_RANKING = "recommender_ranking"
    SURVIVAL = "survival"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    family: MetricFamily
    aliases: tuple[str, ...] = field(default_factory=tuple)
    task_types: tuple[TaskType, ...] = field(default_factory=lambda: (TaskType.UNKNOWN,))
    greater_is_better: bool | None = None
    requires_probabilities: bool = False
    requires_groups_or_time: bool = False
    rank_based: bool = False
    threshold_search_needed: bool = False
    local_metric_available: bool = True
    notes: str | None = None


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
    canonical_metric_name = preset.canonical_metric_name(metric_name) if preset is not None else metric_name
    normalized_metric_name = _normalize_metric_name(canonical_metric_name or "unknown")
    if normalized_metric_name in {"custom", "custom_metric"}:
        return _custom_metric_spec(canonical_metric_name or "custom", task_type)

    spec = MetricRegistry().get(canonical_metric_name)
    if spec is not None:
        return spec

    return _unknown_metric_spec(canonical_metric_name or "unknown", task_type)


def _custom_metric_spec(metric_name: str, task_type: TaskType | str | None) -> MetricSpec:
    return MetricSpec(
        name=metric_name,
        family=MetricFamily.CUSTOM,
        task_types=(_coerce_task_type(task_type),),
        greater_is_better=None,
        local_metric_available=False,
        notes="Custom metric requires competition-specific implementation.",
    )


def _unknown_metric_spec(metric_name: str, task_type: TaskType | str | None) -> MetricSpec:
    return MetricSpec(
        name=metric_name,
        family=MetricFamily.UNKNOWN,
        task_types=(_coerce_task_type(task_type),),
        greater_is_better=None,
        local_metric_available=False,
        notes="Unknown metric is not available in the local metric registry.",
    )


def _coerce_task_type(task_type: TaskType | str | None) -> TaskType:
    if isinstance(task_type, TaskType):
        return task_type
    if task_type is None:
        return TaskType.UNKNOWN
    normalized = str(task_type).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.UNKNOWN


def _normalize_metric_name(metric_name: str) -> str:
    return (
        str(metric_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("mean_absolute_percentage_error", "mape")
        .replace("symmetric_mean_absolute_percentage_error", "smape")
    )


_CLASSIFICATION_TASKS = (
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
)
_REGRESSION_TASKS = (TaskType.REGRESSION, TaskType.TIME_SERIES)


_DEFAULT_METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="auc",
        aliases=("roc_auc", "auroc"),
        family=MetricFamily.RANKING,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        requires_probabilities=True,
        rank_based=True,
    ),
    MetricSpec(
        name="gini",
        aliases=("normalized_gini", "normalized_gini_coefficient"),
        family=MetricFamily.RANKING,
        task_types=(TaskType.BINARY_CLASSIFICATION,),
        greater_is_better=True,
        requires_probabilities=True,
        rank_based=True,
    ),
    MetricSpec(
        name="gini_stability",
        family=MetricFamily.RANKING,
        task_types=(TaskType.BINARY_CLASSIFICATION,),
        greater_is_better=True,
        requires_probabilities=True,
        requires_groups_or_time=True,
        rank_based=True,
        notes="Stability variant of normalized Gini; requires period/group evidence.",
    ),
    MetricSpec(
        name="logloss",
        aliases=("log_loss", "cross_entropy"),
        family=MetricFamily.PROBABILISTIC_CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=False,
        requires_probabilities=True,
    ),
    MetricSpec(
        name="accuracy",
        family=MetricFamily.CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
    ),
    MetricSpec(
        name="f1",
        aliases=("macro_f1", "f1_macro"),
        family=MetricFamily.CLASSIFICATION,
        task_types=_CLASSIFICATION_TASKS,
        greater_is_better=True,
        threshold_search_needed=True,
    ),
    MetricSpec(
        name="quadratic_weighted_kappa",
        aliases=("qwk", "cohen_kappa_quadratic"),
        family=MetricFamily.ORDINAL,
        task_types=(TaskType.MULTICLASS_CLASSIFICATION,),
        greater_is_better=True,
    ),
    MetricSpec(
        name="rmse",
        aliases=("root_mean_squared_error",),
        family=MetricFamily.REGRESSION,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
    ),
    MetricSpec(
        name="rmsle",
        aliases=("root_mean_squared_log_error",),
        family=MetricFamily.REGRESSION,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
    ),
    MetricSpec(
        name="mae",
        aliases=("mean_absolute_error",),
        family=MetricFamily.REGRESSION,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
    ),
    MetricSpec(
        name="mape",
        family=MetricFamily.REGRESSION,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
    ),
    MetricSpec(
        name="smape",
        family=MetricFamily.REGRESSION,
        task_types=_REGRESSION_TASKS,
        greater_is_better=False,
    ),
    MetricSpec(
        name="r2",
        aliases=("r_squared", "coefficient_of_determination"),
        family=MetricFamily.REGRESSION,
        task_types=_REGRESSION_TASKS,
        greater_is_better=True,
    ),
    MetricSpec(
        name="map@k",
        aliases=("mapk", "mean_average_precision_at_k"),
        family=MetricFamily.RECOMMENDER_RANKING,
        task_types=(TaskType.RANKING,),
        greater_is_better=True,
        rank_based=True,
    ),
    MetricSpec(
        name="ndcg",
        aliases=("ndcg@k", "normalized_discounted_cumulative_gain"),
        family=MetricFamily.RECOMMENDER_RANKING,
        task_types=(TaskType.RANKING,),
        greater_is_better=True,
        rank_based=True,
    ),
    MetricSpec(
        name="concordance_index",
        aliases=("c_index", "concordance"),
        family=MetricFamily.SURVIVAL,
        task_types=(TaskType.SURVIVAL,),
        greater_is_better=True,
        rank_based=True,
    ),
)


__all__ = [
    "MetricFamily",
    "MetricRegistry",
    "MetricSpec",
    "TaskType",
    "infer_metric_spec",
]
