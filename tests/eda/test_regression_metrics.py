from __future__ import annotations

import pytest

from kaggle_researcher.eda.metrics.regression import mae, mape, mse, r2, rmse, rmsle, smape


def test_regression_error_metrics_compute_expected_values() -> None:
    y_true = [3.0, 5.0, 7.0]
    y_pred = [2.0, 5.0, 8.0]

    assert mse(y_true, y_pred) == pytest.approx(2 / 3)
    assert rmse(y_true, y_pred) == pytest.approx((2 / 3) ** 0.5)
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert r2(y_true, y_pred) == pytest.approx(0.75)


def test_log_scaled_and_percentage_regression_metrics_compute_values() -> None:
    y_true = [1.0, 3.0]
    y_pred = [1.0, 5.0]

    assert rmsle(y_true, y_pred) >= 0
    assert mape(y_true, y_pred) == pytest.approx(1 / 3)
    assert smape(y_true, y_pred) == pytest.approx(0.25)


def test_rmsle_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rmsle([1.0, -1.0], [1.0, 2.0])
