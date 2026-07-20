from __future__ import annotations

from kaggle_researcher.eda.presets.base import CompetitionPreset


HOME_CREDIT_CRMS_PRESET = CompetitionPreset(
    preset_id="home_credit_crms",
    competition_ids=(
        "home-credit-credit-risk-model-stability",
        "home_credit_credit_risk_model_stability",
        "home_credit_tiny",
        "fixture_competition",
    ),
    preferred_id_columns=("case_id",),
    preferred_target_columns=("target",),
    preferred_time_columns=("WEEK_NUM",),
    preferred_prediction_columns=("score",),
    table_name_patterns={
        "train": ("train", "train_"),
        "test": ("test", "test_"),
        "sample_submission": ("sample_submission", "submission"),
        "base": ("base",),
        "depth_0": ("_0", "depth_0"),
        "depth_1": ("_1", "depth_1"),
        "depth_2": ("_2", "depth_2"),
    },
    metric_aliases={
        "gini_stability": "gini_stability",
        "gini_stability_score": "gini_stability",
        "stability_gini": "gini_stability",
        "gini_stability_metric": "gini_stability",
    },
    known_validation_hints=(
        "WEEK_NUM is a known period column for this competition.",
        "The stability metric requires period evidence before temporal validation is primary.",
        "date_decision is a known decision-date field.",
    ),
)


__all__ = ["HOME_CREDIT_CRMS_PRESET"]
