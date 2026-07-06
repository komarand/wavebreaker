from __future__ import annotations

import pytest

from kaggle_researcher.eda.metrics.gini import gini_from_predictions, normalized_gini
from kaggle_researcher.eda.metrics.gini_stability import (
    simple_gini_stability_score,
    weekly_gini,
)


def test_normalized_gini_is_one_for_perfect_ranking() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]

    assert normalized_gini(y_true, y_pred) == pytest.approx(1.0)
    assert gini_from_predictions(y_true, y_pred) == pytest.approx(1.0)


def test_normalized_gini_is_negative_for_reversed_ranking() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0.9, 0.8, 0.2, 0.1]

    assert normalized_gini(y_true, y_pred) == pytest.approx(-1.0)


def test_normalized_gini_handles_tied_predictions() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.5]

    assert normalized_gini(y_true, y_pred) == pytest.approx(0.0)


def test_weekly_gini_returns_per_week_components() -> None:
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]
    week = [1, 1, 2, 2, 3, 3]

    components = weekly_gini(y_true, y_pred, week)

    assert [component["week"] for component in components] == [1, 2, 3]
    assert [component["n_rows"] for component in components] == [2, 2, 2]
    assert components[0]["gini"] == pytest.approx(1.0)
    assert components[2]["gini"] == pytest.approx(-1.0)


def test_simple_gini_stability_score_includes_penalty_components() -> None:
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]
    week = [1, 1, 2, 2, 3, 3]

    result = simple_gini_stability_score(y_true, y_pred, week)

    assert len(result["weekly_gini"]) == 3
    assert "trend_penalty" in result
    assert "residual_std_penalty" in result
    assert "score" in result
