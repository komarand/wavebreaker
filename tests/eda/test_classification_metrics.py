from __future__ import annotations

import pytest

from kaggle_researcher.eda.metrics.classification import (
    accuracy,
    binary_roc_auc,
    f1,
    logloss,
)


def test_binary_roc_auc_matches_perfect_and_reversed_rankings() -> None:
    y_true = [0, 0, 1, 1]

    assert binary_roc_auc(y_true, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert binary_roc_auc(y_true, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_logloss_is_lower_for_better_probabilities() -> None:
    y_true = [0, 1, 1, 0]

    good = logloss(y_true, [0.1, 0.9, 0.8, 0.2])
    bad = logloss(y_true, [0.9, 0.1, 0.2, 0.8])

    assert good < bad


def test_threshold_classification_metrics_compute_values() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    assert accuracy(y_true, y_pred) == pytest.approx(0.75)
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)
