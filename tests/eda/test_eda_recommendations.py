from __future__ import annotations

from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.schemas import HypothesisResult


def test_binary_iid_fixture_produces_stratified_kfold_action() -> None:
    actions = build_recommended_next_actions(
        {
            "validation_evidence": {
                "primary_validation": {"method": "stratified_kfold"}
            }
        },
        [_confirmed_result()],
    )

    assert any("StratifiedKFold" in action.action for action in actions)


def test_home_credit_like_fixture_produces_temporal_validation_action() -> None:
    actions = build_recommended_next_actions(
        {
            "validation_evidence": {
                "primary_validation": {
                    "method": "temporal_holdout",
                    "split_column": "WEEK_NUM",
                }
            }
        },
        [_confirmed_result()],
    )

    assert any("temporal validation" in action.action for action in actions)


def test_f1_metric_produces_threshold_tuning_action() -> None:
    actions = build_recommended_next_actions(
        {"metric_evidence": {"metric_name": "f1", "requires_threshold": True}},
        [_confirmed_result()],
    )

    assert any("threshold" in action.action.lower() for action in actions)


def test_logloss_metric_produces_calibration_action() -> None:
    actions = build_recommended_next_actions(
        {
            "metric_evidence": {
                "metric_name": "logloss",
                "requires_probabilities": True,
                "requires_calibration": True,
            }
        },
        [_confirmed_result()],
    )

    assert any("calibration" in action.action.lower() for action in actions)
    assert any("probabilities" in action.action.lower() for action in actions)


def test_rmse_metric_produces_regression_loss_action() -> None:
    actions = build_recommended_next_actions(
        {"metric_evidence": {"metric_name": "rmse", "metric_family": "regression_error"}},
        [_confirmed_result()],
    )

    assert any("regression loss" in action.action for action in actions)


def test_leakage_warning_or_failure_produces_p0_action() -> None:
    actions = build_recommended_next_actions(
        {
            "leakage_evidence": [
                {"check_id": "target_in_test", "status": "failed"},
                {"check_id": "group_overlap", "status": "warning"},
            ]
        },
        [_confirmed_result()],
    )

    assert actions[0].priority == "P0"
    assert any("leakage" in action.action.lower() for action in actions)


def test_secondary_tables_without_relationship_evidence_produce_p1_action() -> None:
    actions = build_recommended_next_actions(
        {
            "table_profiles": [
                {"path": "train_base.csv"},
                {"path": "train_static_0.csv"},
            ],
            "relationship_evidence": {},
        },
        [_confirmed_result()],
    )

    assert any(action.priority == "P1" and "relationship" in action.action for action in actions)


def test_every_action_has_evidence_refs_and_actions_are_priority_sorted() -> None:
    actions = build_recommended_next_actions(
        {
            "validation_evidence": {"primary_validation": {"method": "kfold"}},
            "metric_evidence": {
                "requires_calibration": True,
                "metric_family": "regression_error",
            },
        },
        [_confirmed_result()],
    )

    assert actions
    assert all(action.evidence_refs for action in actions)
    priorities = [action.priority for action in actions]
    assert priorities == sorted(priorities)


def test_no_actionable_hypotheses_produces_no_actions() -> None:
    actions = build_recommended_next_actions(
        {"validation_evidence": {"primary_validation": {"method": "kfold"}}},
        [
            HypothesisResult(
                hypothesis_id="h_not_testable",
                category="validation",
                status="not_testable",
                confidence_after_eda="low",
                finding="Missing evidence.",
                impact_on_strategy="Collect missing evidence.",
                limitations=["missing"],
            )
        ],
    )

    assert actions == []


def _confirmed_result() -> HypothesisResult:
    return HypothesisResult(
        hypothesis_id="h_confirmed",
        category="validation",
        status="confirmed",
        confidence_after_eda="high",
        finding="Evidence supports the hypothesis.",
        evidence_refs=["validation_evidence.primary_validation"],
        impact_on_strategy="Use selected validation.",
    )
