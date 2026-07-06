from __future__ import annotations

import pytest

from kaggle_researcher.eda.metrics import (
    MetricFamily,
    MetricRegistry,
    TaskType,
    infer_metric_spec,
)
from kaggle_researcher.eda.presets import HOME_CREDIT_CRMS_PRESET


@pytest.mark.parametrize(
    ("metric_name", "expected_family"),
    [
        ("auc", MetricFamily.RANK_CLASSIFICATION),
        ("roc_auc", MetricFamily.RANK_CLASSIFICATION),
        ("gini", MetricFamily.RANK_CLASSIFICATION),
        ("normalized_gini", MetricFamily.RANK_CLASSIFICATION),
        ("logloss", MetricFamily.PROBABILISTIC_CLASSIFICATION),
        ("log_loss", MetricFamily.PROBABILISTIC_CLASSIFICATION),
        ("cross_entropy", MetricFamily.PROBABILISTIC_CLASSIFICATION),
        ("accuracy", MetricFamily.THRESHOLD_CLASSIFICATION),
        ("f1", MetricFamily.THRESHOLD_CLASSIFICATION),
        ("macro_f1", MetricFamily.THRESHOLD_CLASSIFICATION),
        ("precision", MetricFamily.THRESHOLD_CLASSIFICATION),
        ("recall", MetricFamily.THRESHOLD_CLASSIFICATION),
        ("quadratic_weighted_kappa", MetricFamily.ORDINAL_CLASSIFICATION),
        ("qwk", MetricFamily.ORDINAL_CLASSIFICATION),
        ("cohen_kappa", MetricFamily.ORDINAL_CLASSIFICATION),
        ("rmse", MetricFamily.REGRESSION_ERROR),
        ("mse", MetricFamily.REGRESSION_ERROR),
        ("mae", MetricFamily.REGRESSION_ERROR),
        ("rmsle", MetricFamily.REGRESSION_ERROR),
        ("mape", MetricFamily.REGRESSION_ERROR),
        ("smape", MetricFamily.REGRESSION_ERROR),
        ("r2", MetricFamily.REGRESSION_ERROR),
        ("map@k", MetricFamily.RANKING),
        ("mapk", MetricFamily.RANKING),
        ("ndcg", MetricFamily.RANKING),
        ("ndcg@k", MetricFamily.RANKING),
        ("recall@k", MetricFamily.RANKING),
        ("concordance_index", MetricFamily.SURVIVAL),
        ("c_index", MetricFamily.SURVIVAL),
        ("gini_stability", MetricFamily.TEMPORAL_STABILITY),
    ],
)
def test_supported_metric_aliases_resolve_to_expected_family(
    metric_name: str,
    expected_family: MetricFamily,
) -> None:
    spec = infer_metric_spec(metric_name, TaskType.BINARY_CLASSIFICATION)

    assert spec.family == expected_family


def test_metric_specific_flags_are_generic_not_home_credit_defaults() -> None:
    roc_auc = infer_metric_spec("roc_auc", "binary_classification")
    logloss = infer_metric_spec("logloss", "binary_classification")
    f1 = infer_metric_spec("f1", "binary_classification")
    rmse = infer_metric_spec("rmse", "regression")
    ndcg = infer_metric_spec("ndcg", "ranking")

    assert roc_auc.requires_probabilities is True
    assert roc_auc.requires_time is False
    assert roc_auc.family == MetricFamily.RANK_CLASSIFICATION
    assert logloss.requires_probabilities is True
    assert logloss.requires_calibration is True
    assert f1.requires_threshold is True
    assert f1.threshold_search_needed is True
    assert rmse.family == MetricFamily.REGRESSION_ERROR
    assert rmse.greater_is_better is False
    assert ndcg.requires_query_groups is True
    assert ndcg.supports_local_eval is False


def test_gini_stability_is_specific_temporal_registry_entry() -> None:
    known = infer_metric_spec("gini_stability", "binary_classification")
    unknown = infer_metric_spec("not_gini_stability", "binary_classification")

    assert known.family == MetricFamily.TEMPORAL_STABILITY
    assert known.requires_time is True
    assert known.requires_probabilities is True
    assert known.requires_groups_or_time is True
    assert unknown.family == MetricFamily.UNKNOWN
    assert unknown.supports_local_eval is False
    assert unknown.needs_custom_implementation is True


def test_unknown_and_custom_metrics_require_manual_implementation() -> None:
    unknown = infer_metric_spec("mystery_metric", "binary_classification")
    custom = infer_metric_spec("custom", "regression")

    assert unknown.family == MetricFamily.UNKNOWN
    assert unknown.task_types == [TaskType.BINARY_CLASSIFICATION]
    assert unknown.supports_local_eval is False
    assert unknown.needs_custom_implementation is True
    assert custom.family == MetricFamily.CUSTOM
    assert custom.task_types == [TaskType.REGRESSION]
    assert custom.supports_local_eval is False
    assert custom.needs_custom_implementation is True


def test_registry_alias_lookup_returns_canonical_spec() -> None:
    registry = MetricRegistry()

    assert registry.get("roc_auc").name == "auc"
    assert registry.get("normalized_gini").name == "gini"
    assert registry.get("macro_f1").name == "f1"
    assert registry.get("qwk").name == "quadratic_weighted_kappa"
    assert registry.get("mapk").name == "map@k"
    assert registry.get("unknown_metric") is None


def test_preset_aliases_are_applied_without_global_home_credit_default() -> None:
    preset_spec = infer_metric_spec(
        "stability_gini",
        "binary_classification",
        preset=HOME_CREDIT_CRMS_PRESET,
    )
    generic_spec = infer_metric_spec("stability_gini", "binary_classification")

    assert preset_spec.name == "gini_stability"
    assert preset_spec.family == MetricFamily.TEMPORAL_STABILITY
    assert generic_spec.name == "stability_gini"
    assert generic_spec.family == MetricFamily.UNKNOWN
