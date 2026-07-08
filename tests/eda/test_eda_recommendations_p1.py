from __future__ import annotations

from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.schemas import HypothesisResult


def test_relationship_recommendation_cites_relationship_evidence() -> None:
    actions = build_recommended_next_actions(
        {
            "relationship_evidence": {
                "relationships": [
                    {
                        "table": "train_orders.csv",
                        "relationship_type": "one_to_many",
                        "requires_aggregation": True,
                    }
                ]
            }
        },
        [_confirmed_result("relationship")],
    )

    relationship_actions = [action for action in actions if "Aggregate" in action.action]
    assert relationship_actions
    assert relationship_actions[0].evidence_refs == ["relationship_evidence.relationships"]


def test_high_drift_recommendation_cites_drift_evidence() -> None:
    actions = build_recommended_next_actions(
        {"drift_evidence": {"status": "completed", "severity": "high"}},
        [_confirmed_result("drift")],
    )

    assert any(action.evidence_refs == ["drift_evidence"] for action in actions)
    assert any("drift" in action.action.lower() for action in actions)


def test_baseline_complete_recommends_sanity_floor() -> None:
    actions = build_recommended_next_actions(
        {"baseline_evidence": {"status": "completed", "metric_value": 0.72}},
        [_confirmed_result("baseline")],
    )

    assert any("sanity floor" in action.action.lower() for action in actions)
    assert any(action.evidence_refs == ["baseline_evidence"] for action in actions)


def test_feature_probe_recommends_high_potential_and_risky_families() -> None:
    actions = build_recommended_next_actions(
        {
            "feature_probe_evidence": [
                {"feature_family": "secondary_table_aggregations", "status": "high_potential"},
                {"feature_family": "target_encoding_or_woe", "status": "unsafe"},
                {
                    "feature_family": "regression_target_transform",
                    "status": "medium_potential",
                },
            ]
        },
        [_confirmed_result("feature")],
    )

    assert any("high-potential" in action.action for action in actions)
    assert any("risky feature" in action.action.lower() for action in actions)
    assert any("target transform" in action.action.lower() for action in actions)
    assert all(action.evidence_refs for action in actions)


def test_notebook_risky_pattern_recommendation_cites_notebook_evidence() -> None:
    actions = build_recommended_next_actions(
        {
            "notebook_static_analysis": {
                "suspicious_leaderboard_overfit_patterns": [
                    {"pattern": "public_lb_tuning"}
                ],
                "feature_families": [{"pattern": "target_encoding"}],
            }
        },
        [_confirmed_result("notebook")],
    )

    assert any("notebook" in action.action.lower() for action in actions)
    assert any(
        action.evidence_refs
        == ["notebook_static_analysis.suspicious_leaderboard_overfit_patterns"]
        for action in actions
    )


def test_metric_specific_recommendations_still_cite_metric_evidence() -> None:
    actions = build_recommended_next_actions(
        {
            "metric_evidence": {
                "requires_threshold": True,
                "requires_calibration": True,
                "metric_family": "regression_error",
            }
        },
        [_confirmed_result("metric")],
    )

    metric_refs = [
        ref
        for action in actions
        for ref in action.evidence_refs
        if ref.startswith("metric_evidence")
    ]
    assert "metric_evidence.requires_threshold" in metric_refs
    assert "metric_evidence.requires_calibration" in metric_refs
    assert "metric_evidence.metric_family" in metric_refs


def _confirmed_result(category: str) -> HypothesisResult:
    return HypothesisResult(
        hypothesis_id=f"{category}_001",
        category=category,
        status="confirmed",
        confidence_after_eda="high",
        finding="Evidence supports the hypothesis.",
        evidence_refs=[f"{category}_evidence"],
        impact_on_strategy="Use evidence.",
    )
