from __future__ import annotations

import random

from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.schemas import EdaEvidencePack, HypothesisResult
from kaggle_researcher.eda.summary import build_eda_summary


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


def test_baseline_complete_recommends_completed_baseline_comparison() -> None:
    actions = build_recommended_next_actions(
        {
            "baseline_evidence": {
                "status": "completed",
                "metric_value": 0.72,
                "preprocessing_policy": {"fit_scope": "inside_cv_folds"},
            }
        },
        [_confirmed_result("baseline")],
    )

    assert any("completed eda baseline" in action.action.lower() for action in actions)
    assert any(
        action.evidence_refs
        == ["baseline_evidence.metric_value", "baseline_evidence.preprocessing_policy"]
        for action in actions
    )


def test_baseline_strategy_hints_reference_policy_and_skipped_fold_safe_action() -> None:
    completed = build_eda_strategy_hints(
        {
            "baseline_evidence": {
                "status": "completed",
                "metric_value": 0.72,
                "preprocessing_policy": {"fit_scope": "inside_cv_folds"},
            }
        }
    )
    skipped = build_eda_strategy_hints(
        {
            "baseline_evidence": {"status": "skipped", "reason": "disabled"},
            "validation_evidence": {"primary_validation": {"method": "kfold"}},
            "feature_diagnostics": {"status": "completed"},
        }
    )

    assert completed["baseline"][0]["action"] == "Compare future experiments against the completed EDA baseline."
    assert completed["baseline"][0]["evidence_refs"] == [
        "baseline_evidence.metric_value",
        "baseline_evidence.preprocessing_policy",
    ]
    assert completed["baseline"][0]["applies_to"] == ["baseline", "model_selection"]
    assert skipped["baseline"][0]["action"] == "Run a simple fold-safe baseline before advanced modeling."
    assert skipped["baseline"][0]["priority"] == "P0"
    assert skipped["baseline"][0]["risk"] == "medium"
    assert skipped["baseline"][0]["evidence_refs"] == [
        "validation_evidence.primary_validation",
        "feature_diagnostics",
    ]


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


def test_strategy_hints_do_not_duplicate_primary_action_intents() -> None:
    actions = build_recommended_next_actions(
        {
            "validation_evidence": {
                "primary_validation": {"method": "stratified_kfold"}
            },
            "metric_evidence": {"requires_threshold": True},
            "baseline_evidence": {"status": "completed", "metric_value": 0.72},
            "eda_strategy_hints": {
                "validation": [
                    {
                        "priority": "P0",
                        "action": "Use StratifiedKFold for model validation.",
                        "why": "Selected by EDA.",
                        "evidence_refs": ["validation_evidence.primary_validation"],
                    },
                    {
                        "priority": "P0",
                        "action": "Tune thresholds only inside validation folds.",
                        "why": "Required by the metric.",
                        "evidence_refs": ["metric_evidence.requires_threshold"],
                    },
                ],
                "baseline": [
                    {
                        "priority": "P1",
                        "action": "Use the EDA baseline as a reproducible sanity floor.",
                        "why": "Baseline completed.",
                        "evidence_refs": ["baseline_evidence"],
                    }
                ],
                "first_experiments": [
                    {
                        "priority": "P0",
                        "action": "Build a baseline with safe features.",
                        "why": "Start simply.",
                        "evidence_refs": ["validation_evidence.primary_validation"],
                    },
                    {
                        "priority": "P1",
                        "action": "Run validation-only threshold tuning.",
                        "why": "Metric requires it.",
                        "evidence_refs": ["metric_evidence.requires_threshold"],
                    },
                ],
            },
        },
        [_confirmed_result("validation")],
    )

    assert sum("threshold" in action.action.lower() for action in actions) == 1
    assert sum("sanity floor" in action.action.lower() for action in actions) == 1
    assert sum(action.action.lower().startswith("build a baseline") for action in actions) == 1
    assert sum(
        "validation_evidence.primary_validation" in action.evidence_refs
        and any(token in action.action.lower() for token in ("kfold", "validation"))
        for action in actions
    ) == 1


def test_validation_action_dedup_merges_stratified_kfold_wording() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P0",
                "action": "Use StratifiedKFold for model validation.",
                "evidence_refs": ["validation_evidence.primary_validation"],
            },
            {
                "priority": "P0",
                "action": "Use stratified_kfold for model comparison.",
                "evidence_refs": ["validation_evidence.primary_validation"],
            },
        ]
    )

    assert len(actions) == 1
    assert actions[0].priority == "P0"
    assert actions[0].evidence_refs == ["validation_evidence.primary_validation"]


def test_threshold_action_dedup_keeps_validation_boundary() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P0",
                "action": "Tune classification thresholds on validation data only.",
                "evidence_refs": ["metric_evidence.requires_threshold"],
            },
            {
                "priority": "P1",
                "action": "Run validation-only threshold tuning.",
                "evidence_refs": ["metric_evidence.requires_threshold"],
            },
            {
                "priority": "P0",
                "action": "Tune thresholds only inside validation folds.",
                "evidence_refs": ["metric_evidence.requires_threshold"],
            },
        ]
    )

    threshold_actions = [action for action in actions if "threshold" in action.action.lower()]
    assert len(threshold_actions) == 1
    assert threshold_actions[0].priority == "P0"
    assert "validation folds" in threshold_actions[0].action.lower() or "validation-only" in threshold_actions[0].action.lower()
    assert threshold_actions[0].evidence_refs == ["metric_evidence.requires_threshold"]


def test_baseline_action_dedup_merges_sanity_floor_wording() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P1",
                "action": "Use the EDA baseline as a reproducible sanity floor.",
                "evidence_refs": ["baseline_evidence"],
            },
            {
                "priority": "P1",
                "action": "Use the honest baseline as a sanity floor.",
                "evidence_refs": ["baseline_evidence"],
            },
        ]
    )

    assert sum("sanity floor" in action.action.lower() for action in actions) == 1


def test_feature_actions_are_not_over_deduplicated() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P1",
                "action": "Add fold-fitted categorical encoders.",
                "evidence_refs": ["feature_diagnostics.categorical_feature_diagnostics"],
            },
            {
                "priority": "P1",
                "action": "Evaluate missingness indicators.",
                "evidence_refs": ["feature_diagnostics.missingness_diagnostics"],
            },
        ]
    )

    assert len(actions) == 2
    assert any("categorical encoders" in action.action for action in actions)
    assert any("missingness indicators" in action.action for action in actions)


def test_duplicate_actions_merge_evidence_refs_and_priority() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P1",
                "action": "Tune thresholds only inside validation folds.",
                "evidence_refs": ["validation_evidence.primary_validation"],
            },
            {
                "priority": "P0",
                "action": "Tune classification thresholds on validation data only.",
                "evidence_refs": ["metric_evidence.requires_threshold"],
            },
        ]
    )

    assert len(actions) == 1
    assert actions[0].priority == "P0"
    assert actions[0].evidence_refs == [
        "validation_evidence.primary_validation",
        "metric_evidence.requires_threshold",
    ]


def test_strategy_hints_are_primary_over_equivalent_legacy_actions() -> None:
    actions = build_recommended_next_actions(
        eda_strategy_hints={
            "validation": [
                {
                    "priority": "P0",
                    "action": "Tune thresholds only inside validation folds.",
                    "why": "Metric evidence requires thresholded outputs.",
                    "evidence_refs": ["metric_evidence.requires_threshold"],
                    "risk": "medium",
                    "applies_to": ["validation", "metric"],
                }
            ]
        },
        legacy_actions=[
            {
                "priority": "P0",
                "action": "Tune classification thresholds on validation data only.",
                "why": "Metric evidence requires thresholded predictions.",
                "evidence_refs": ["metric_evidence.requires_threshold"],
            }
        ],
    )

    assert len(actions) == 1
    assert actions[0].action == "Tune thresholds only inside validation folds."
    assert actions[0].risk == "medium"
    assert actions[0].source_categories == ["validation"]


def test_summary_renders_deduplicated_recommended_actions() -> None:
    pack = EdaEvidencePack(
        competition_id="dedupe_summary",
        created_at="2026-07-11T12:00:00+03:00",
        run_id="dedupe_summary_run",
        recommended_next_actions=build_recommended_next_actions(
            legacy_actions=[
                {
                    "priority": "P0",
                    "action": "Tune classification thresholds on validation data only.",
                    "evidence_refs": ["metric_evidence.requires_threshold"],
                },
                {
                    "priority": "P0",
                    "action": "Tune thresholds only inside validation folds.",
                    "evidence_refs": ["metric_evidence.requires_threshold"],
                },
                {
                    "priority": "P0",
                    "action": "Use StratifiedKFold for model validation.",
                    "evidence_refs": ["validation_evidence.primary_validation"],
                },
                {
                    "priority": "P0",
                    "action": "Use stratified_kfold for model comparison.",
                    "evidence_refs": ["validation_evidence.primary_validation"],
                },
                {
                    "priority": "P1",
                    "action": "Use the EDA baseline as a reproducible sanity floor.",
                    "evidence_refs": ["baseline_evidence"],
                },
                {
                    "priority": "P1",
                    "action": "Use the honest baseline as a sanity floor.",
                    "evidence_refs": ["baseline_evidence"],
                },
            ]
        ),
    )

    summary = build_eda_summary(pack)
    assert "## Testable follow-up hypotheses" in summary
    assert "Tune classification thresholds" not in summary
    assert "Use StratifiedKFold" not in summary
    assert "sanity floor" not in summary


def test_recommendation_aggregation_is_deterministic_for_shuffled_inputs() -> None:
    legacy_actions = [
        {
            "priority": "P0",
            "action": "Tune classification thresholds on validation data only.",
            "evidence_refs": ["metric_evidence.requires_threshold"],
        },
        {
            "priority": "P1",
            "action": "Run validation-only threshold tuning.",
            "evidence_refs": ["metric_evidence.requires_threshold"],
        },
        {
            "priority": "P0",
            "action": "Use stratified_kfold for model comparison.",
            "evidence_refs": ["validation_evidence.primary_validation"],
        },
        {
            "priority": "P0",
            "action": "Use StratifiedKFold for model validation.",
            "evidence_refs": ["validation_evidence.primary_validation"],
        },
        {
            "priority": "P1",
            "action": "Evaluate missingness indicators.",
            "evidence_refs": ["feature_diagnostics.missingness_diagnostics"],
        },
    ]
    expected = [
        action.model_dump(mode="json")
        for action in build_recommended_next_actions(legacy_actions=legacy_actions)
    ]

    rng = random.Random(42)
    for _ in range(5):
        shuffled = list(legacy_actions)
        rng.shuffle(shuffled)
        actual = [
            action.model_dump(mode="json")
            for action in build_recommended_next_actions(legacy_actions=shuffled)
        ]
        assert actual == expected


def test_malformed_empty_action_is_removed() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P1",
                "action": ".",
                "why": "Malformed action text.",
                "evidence_refs": ["feature_diagnostics.text_feature_diagnostics"],
            },
            {
                "priority": "P1",
                "action": "Extract simple text/code features before heavier NLP.",
                "why": "Text diagnostics found free-text or code-like columns.",
                "evidence_refs": ["feature_diagnostics.text_feature_diagnostics"],
            },
        ]
    )

    assert len(actions) == 1
    assert actions[0].action != "."
    assert actions[0].action.strip()


def test_completed_baseline_does_not_absorb_feature_diagnostic_refs() -> None:
    actions = build_recommended_next_actions(
        legacy_actions=[
            {
                "priority": "P1",
                "action": "Use the EDA baseline as a reproducible sanity floor.",
                "why": "Baseline completed.",
                "evidence_refs": ["baseline_evidence"],
            },
            {
                "priority": "P0",
                "action": "Build a baseline with safe numeric and categorical features.",
                "why": "Validation and feature diagnostics define safe feature roles.",
                "evidence_refs": ["validation_evidence.primary_validation", "feature_diagnostics"],
            },
        ]
    )

    sanity = next(action for action in actions if "sanity floor" in action.action.lower())
    experiment = next(action for action in actions if action.action.lower().startswith("build a baseline"))
    assert sanity.evidence_refs == ["baseline_evidence"]
    assert "feature_diagnostics" not in sanity.evidence_refs
    assert experiment.evidence_refs == [
        "validation_evidence.primary_validation",
        "feature_diagnostics",
    ]


def test_build_baseline_experiment_remains_separate_from_sanity_floor() -> None:
    actions = build_recommended_next_actions(
        {
            "baseline_evidence": {"status": "completed"},
            "eda_strategy_hints": {
                "baseline": [
                    {
                        "priority": "P1",
                        "action": "Use the EDA baseline as a reproducible sanity floor.",
                        "why": "Baseline runner completed.",
                        "evidence_refs": ["baseline_evidence"],
                    }
                ],
                "first_experiments": [
                    {
                        "priority": "P0",
                        "action": "Build a baseline with safe numeric and categorical features.",
                        "why": "Validation and feature diagnostics define safe feature roles.",
                        "evidence_refs": ["validation_evidence.primary_validation", "feature_diagnostics"],
                    }
                ],
            },
        }
    )

    assert sum("sanity floor" in action.action.lower() for action in actions) == 1
    assert sum(action.action.lower().startswith("build a baseline") for action in actions) == 1


def test_summary_has_no_malformed_recommended_action_bullet() -> None:
    pack = EdaEvidencePack(
        competition_id="bad_action_summary",
        created_at="2026-07-11T12:00:00+03:00",
        run_id="bad_action_summary_run",
        recommended_next_actions=build_recommended_next_actions(
            legacy_actions=[
                {
                    "priority": "P1",
                    "action": ".",
                    "why": "Malformed action text.",
                    "evidence_refs": ["feature_diagnostics.text_feature_diagnostics"],
                },
                {
                    "priority": "P1",
                    "action": "Extract simple text/code features before heavier NLP.",
                    "why": "Text diagnostics found free-text or code-like columns.",
                    "evidence_refs": ["feature_diagnostics.text_feature_diagnostics"],
                },
            ]
        ),
    )

    summary = build_eda_summary(pack)
    assert "## Testable follow-up hypotheses" in summary
    assert "Malformed action text" not in summary
    assert "`. [refs:" not in summary


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
