from __future__ import annotations

import json
from pathlib import Path

from kaggle_researcher.eda.modules.notebook_static_analyzer import analyze_notebooks_static
from kaggle_researcher.schemas import RetrievedDocument, SourceDocument


def test_static_fixture_detects_model_and_cv_patterns() -> None:
    source = SourceDocument(
        id="nb-1",
        competition_id="comp-1",
        source="kaggle",
        title="LightGBM folds",
        url="https://example.com/nb",
        content="""
from sklearn.model_selection import StratifiedKFold, GroupKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
cv = StratifiedKFold(n_splits=5)
model = LGBMClassifier()
        """,
    )

    result = analyze_notebooks_static([source])

    assert result["status"] == "completed"
    assert _patterns(result["cv_strategy"]) >= {"stratified_kfold", "group_kfold"}
    assert _patterns(result["model_families"]) >= {"lightgbm", "catboost"}
    assert result["notebook_scores_are_observations_not_truth"] is True


def test_static_analysis_does_not_execute_code(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    source = {
        "id": "dangerous",
        "title": "Dangerous text",
        "source": "kaggle",
        "content": (
            "import pathlib\n"
            f"pathlib.Path({str(marker)!r}).write_text('executed')\n"
            "raise RuntimeError('this would fail if executed')\n"
            "KFold(n_splits=5)\n"
        ),
    }

    result = analyze_notebooks_static([source], output_dir=tmp_path)

    assert not marker.exists()
    assert _patterns(result["cv_strategy"]) == {"kfold"}
    assert (tmp_path / "notebook_static_analysis.json").is_file()


def test_metric_specific_patterns_are_extracted() -> None:
    source = RetrievedDocument(
        id="nb-2",
        competition_id="comp-1",
        source="kaggle",
        title="Metric notebook",
        url="https://example.com/metric",
        content="""
from sklearn.metrics import roc_auc_score, log_loss, cohen_kappa_score
score = roc_auc_score(y, p)
loss = log_loss(y, p.clip(1e-5, 1 - 1e-5))
pred = np.expm1(model.predict(X_valid))
rmsle = mean_squared_log_error(y_valid, pred) ** 0.5
# QWK threshold optimization
best = cohen_kappa_score(y, rounded, weights='quadratic')
        """,
        score=0.9,
        rrf_score=0.2,
    )

    result = analyze_notebooks_static([source])

    assert _patterns(result["metric_code"]) >= {
        "roc_auc",
        "logloss_calibration",
        "rmsle_target_transform",
        "qwk_threshold_optimization",
    }
    assert "clipping" in _patterns(result["postprocessing"])
    assert any("Logloss" in warning for warning in result["warnings"])


def test_ipynb_json_source_is_parsed_statically() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "from sklearn.model_selection import TimeSeriesSplit\n",
                    "import xgboost as xgb\n",
                    "pred = rankdata(pred) / len(pred)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "source": ["Public LB shake-up warning; avoid leaderboard probing."],
            },
        ]
    }

    result = analyze_notebooks_static(
        [{"id": "ipynb", "title": "Notebook JSON", "content": json.dumps(notebook)}]
    )

    assert "time_series_split" in _patterns(result["cv_strategy"])
    assert "xgboost" in _patterns(result["model_families"])
    assert "rank_averaging" in _patterns(result["postprocessing"])
    assert "public_lb_tuning" in _patterns(result["suspicious_leaderboard_overfit_patterns"])


def _patterns(items: list[dict]) -> set[str]:
    return {item["pattern"] for item in items}
