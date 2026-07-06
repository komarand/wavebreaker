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
        ("auc", MetricFamily.RANKING),
        ("roc_auc", MetricFamily.RANKING),
        ("gini", MetricFamily.RANKING),
        ("normalized_gini", MetricFamily.RANKING),
        ("gini_stability", MetricFamily.RANKING),
        ("logloss", MetricFamily.PROBABILISTIC_CLASSIFICATION),
        ("accuracy", MetricFamily.CLASSIFICATION),
        ("f1", MetricFamily.CLASSIFICATION),
        ("macro_f1", MetricFamily.CLASSIFICATION),
        ("quadratic_weighted_kappa", MetricFamily.ORDINAL),
        ("rmse", MetricFamily.REGRESSION),
        ("rmsle", MetricFamily.REGRESSION),
        ("mae", MetricFamily.REGRESSION),
        ("mape", MetricFamily.REGRESSION),
        ("smape", MetricFamily.REGRESSION),
        ("r2", MetricFamily.REGRESSION),
        ("map@k", MetricFamily.RECOMMENDER_RANKING),
        ("ndcg", MetricFamily.RECOMMENDER_RANKING),
        ("concordance_index", MetricFamily.SURVIVAL),
    ],
)
def test_each_supported_metric_maps_to_expected_family(
    metric_name: str,
    expected_family: MetricFamily,
) -> None:
    spec = infer_metric_spec(metric_name, TaskType.BINARY_CLASSIFICATION)

    assert spec.family == expected_family
    assert spec.local_metric_available is True


def test_unknown_metric_returns_unavailable_unknown_spec() -> None:
    spec = infer_metric_spec("mystery_metric", "binary_classification")

    assert spec.name == "mystery_metric"
    assert spec.family == MetricFamily.UNKNOWN
    assert spec.task_types == (TaskType.BINARY_CLASSIFICATION,)
    assert spec.local_metric_available is False


def test_custom_metric_returns_unavailable_custom_spec() -> None:
    spec = infer_metric_spec("custom", "regression")

    assert spec.family == MetricFamily.CUSTOM
    assert spec.task_types == (TaskType.REGRESSION,)
    assert spec.local_metric_available is False


def test_gini_stability_is_one_metric_spec_not_global_default() -> None:
    known = infer_metric_spec("gini_stability", "binary_classification")
    unknown = infer_metric_spec("not_gini_stability", "binary_classification")

    assert known.name == "gini_stability"
    assert known.requires_groups_or_time is True
    assert known.requires_probabilities is True
    assert known.rank_based is True
    assert unknown.name == "not_gini_stability"
    assert unknown.family == MetricFamily.UNKNOWN
    assert unknown.local_metric_available is False
    assert unknown.requires_groups_or_time is False


def test_registry_alias_lookup_returns_canonical_spec() -> None:
    registry = MetricRegistry()

    assert registry.get("roc_auc").name == "auc"
    assert registry.get("normalized_gini").name == "gini"
    assert registry.get("macro_f1").name == "f1"
    assert registry.get("qwk").name == "quadratic_weighted_kappa"
    assert registry.get("unknown_metric") is None


def test_metric_spec_flags_are_metric_specific() -> None:
    logloss = infer_metric_spec("logloss", "binary_classification")
    accuracy = infer_metric_spec("accuracy", "binary_classification")
    rmse = infer_metric_spec("rmse", "regression")
    r2 = infer_metric_spec("r2", "regression")

    assert logloss.requires_probabilities is True
    assert logloss.greater_is_better is False
    assert accuracy.requires_probabilities is False
    assert accuracy.greater_is_better is True
    assert rmse.greater_is_better is False
    assert r2.greater_is_better is True


def test_metric_inference_uses_preset_aliases_without_global_default() -> None:
    preset_spec = infer_metric_spec(
        "stability_gini",
        "binary_classification",
        preset=HOME_CREDIT_CRMS_PRESET,
    )
    generic_spec = infer_metric_spec("stability_gini", "binary_classification")

    assert preset_spec.name == "gini_stability"
    assert preset_spec.family == MetricFamily.RANKING
    assert generic_spec.name == "stability_gini"
    assert generic_spec.family == MetricFamily.UNKNOWN
    assert generic_spec.local_metric_available is False
