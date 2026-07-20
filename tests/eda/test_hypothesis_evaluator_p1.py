from __future__ import annotations

from kaggle_researcher.eda.modules.hypothesis_evaluator import evaluate_hypotheses
from kaggle_researcher.eda.schemas import ResearchHypothesis


def test_relationship_hypothesis_confirmed_when_join_key_and_coverage_exist() -> None:
    result = evaluate_hypotheses(
        [_hypothesis("rel_001", "relationship", "Secondary tables join by customer_id.")],
        {
            "relationship_evidence": {
                "relationships": [
                    {
                        "table": "train_orders.csv",
                        "selected_join_key": "customer_id",
                        "relationship_type": "one_to_many",
                        "coverage_left_to_right": 0.95,
                    }
                ]
            }
        },
    )[0]

    assert result.status == "confirmed"
    assert result.evidence_refs == ["relationship_evidence.relationships"]


def test_relationship_hypothesis_partially_confirmed_for_weak_candidate_keys() -> None:
    result = evaluate_hypotheses(
        [_hypothesis("rel_weak_001", "relationship", "Secondary tables may have join keys.")],
        {
            "relationship_evidence": {
                "relationships": [
                    {
                        "table": "train_extra.csv",
                        "candidate_join_keys": ["customer_id"],
                        "selected_join_key": None,
                        "relationship_type": "unknown",
                    }
                ]
            }
        },
    )[0]

    assert result.status == "partially_confirmed"
    assert result.limitations


def test_drift_hypothesis_confirmed_for_medium_or_high_drift_and_rejected_for_low() -> None:
    high_result = evaluate_hypotheses(
        [_hypothesis("drift_001", "drift", "Train/test drift is likely.")],
        {"drift_evidence": {"status": "completed", "severity": "high", "shared_columns": ["x"]}},
    )[0]
    low_result = evaluate_hypotheses(
        [_hypothesis("drift_low_001", "drift", "Train/test drift is likely.")],
        {"drift_evidence": {"status": "completed", "severity": "low", "shared_columns": ["x"]}},
    )[0]

    assert high_result.status == "confirmed"
    assert high_result.evidence_refs == ["drift_evidence"]
    assert low_result.status == "rejected"


def test_baseline_disabled_is_skipped_not_failure() -> None:
    result = evaluate_hypotheses(
        [_hypothesis("base_001", "baseline", "Honest baseline should run.")],
        {
            "baseline_evidence": {
                "status": "skipped",
                "reason": "Baseline runner requires enable_baseline=true.",
            }
        },
    )[0]

    assert result.status == "skipped"
    assert result.limitations == ["Baseline runner requires enable_baseline=true."]


def test_feature_hypothesis_uses_feature_probe_potential_statuses() -> None:
    result = evaluate_hypotheses(
        [_hypothesis("feat_001", "feature", "Feature families may help.")],
        {
            "feature_probe_evidence": [
                {"feature_family": "date_features", "status": "medium_potential"},
                {"feature_family": "target_encoding_or_woe", "status": "unsafe"},
            ]
        },
    )[0]

    assert result.status == "partially_confirmed"
    assert result.evidence_refs == ["feature_probe_evidence"]
    assert result.limitations


def test_notebook_hypothesis_confirms_only_pattern_observed() -> None:
    result = evaluate_hypotheses(
        [_hypothesis("nb_001", "notebook", "Top notebooks contain useful patterns.")],
        {
            "notebook_static_analysis": {
                "status": "completed",
                "cv_strategy": [{"pattern": "stratified_kfold"}],
                "feature_families": [],
                "model_families": [{"pattern": "lightgbm"}],
                "metric_code": [],
                "postprocessing": [],
                "suspicious_leaderboard_overfit_patterns": [],
            }
        },
    )[0]

    assert result.status == "confirmed"
    assert "not as factual performance proof" in result.impact_on_strategy
    assert result.limitations == ["Notebook code was not executed and scores are not proof."]


def _hypothesis(hypothesis_id: str, category: str, claim: str) -> ResearchHypothesis:
    return ResearchHypothesis(
        hypothesis_id=hypothesis_id,
        category=category,
        claim=claim,
        priority="P1",
        confidence_before_eda="medium",
    )
