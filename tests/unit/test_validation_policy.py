from __future__ import annotations

from kaggle_researcher.reasoning.validation_architect import enforce_temporal_validation_policy


def test_temporal_policy_promotes_strict_temporal_validation() -> None:
    result = enforce_temporal_validation_policy(
        {
            "recommended_cv": "StratifiedGroupKFold by WEEK_NUM",
            "validation_risk": "medium",
            "likely_split": "groups",
            "failure_modes": [],
            "reasoning": "Use groups.",
            "confidence": "medium",
            "evidence_ids": [],
        },
        competition_desc="Predict default. Metric: gini stability.",
        plan_data={"metric": "gini stability"},
        retrieved_documents=[{"title": "Notebook", "content": "Use WEEK_NUM and month features."}],
    )

    assert result["primary_validation"]["method"] == "out_of_time_holdout_and_rolling_cv"
    assert "StratifiedGroupKFold" in result["secondary_validation"]["method"]
    assert result["validation_risk"] == "high"
    assert result["policy_enforced"] is True
    assert any("demoted" in note for note in result["policy_notes"])


def test_temporal_policy_flags_random_and_plain_stratified_kfold() -> None:
    result = enforce_temporal_validation_policy(
        {
            "recommended_cv": "RandomKFold or plain StratifiedKFold",
            "validation_risk": "low",
            "likely_split": "random",
            "failure_modes": [],
            "reasoning": "stability metric",
            "confidence": "low",
            "evidence_ids": [],
        },
        competition_desc="Out-of-time evaluation",
        plan_data={"metric": "stability"},
        retrieved_documents=[],
    )

    assert "RandomKFold" in result["do_not_use"]
    assert "plain StratifiedKFold" in result["do_not_use"]


def test_non_temporal_competition_is_not_rewritten() -> None:
    original = {
        "recommended_cv": "StratifiedKFold",
        "validation_risk": "medium",
        "likely_split": "iid",
        "failure_modes": [],
        "reasoning": "No temporal signal.",
        "confidence": "medium",
        "evidence_ids": [],
    }

    result = enforce_temporal_validation_policy(
        original.copy(),
        competition_desc="Classify images.",
        plan_data={"metric": "accuracy"},
        retrieved_documents=[{"title": "CNN notebook", "content": "image classification"}],
    )

    assert result["recommended_cv"] == "StratifiedKFold"
    assert result["policy_enforced"] is False
