from __future__ import annotations

from kaggle_researcher.eda.modules.risk_register import (
    build_eda_risk_register,
    deduplicate_eda_risks,
    risk_summary,
    validate_eda_risk_register,
)
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.schemas import EdaEvidencePack, RecommendedNextAction
from kaggle_researcher.eda.summary import build_eda_summary


def test_primary_id_risk_is_high_and_mitigated_without_column_name() -> None:
    risks = _risks(inferred_schema={"primary_id_column": "row_id", "confidence": "high"})

    risk = _risk(risks, "Primary ID must not be used as a model feature")
    assert risk["risk_type"] == "leakage"
    assert risk["severity"] == "high"
    assert risk["status"] == "mitigated_by_policy"
    assert "Exclude primary IDs" in str(risk["mitigation"])
    assert "row_id" not in risk["title"]
    assert "row_id" not in risk["finding"]


def test_threshold_metric_risk_mentions_validation_folds() -> None:
    risks = _risks(metric_evidence={"requires_threshold": True})

    risk = _risk(risks, "Threshold choice affects metric score")
    assert risk["risk_type"] == "metric"
    assert risk["severity"] == "medium"
    assert "validation folds" in str(risk["mitigation"])


def test_severe_imbalance_adds_target_and_validation_risks() -> None:
    risks = _risks(
        target_diagnostics={
            "status": "completed",
            "distribution": {"target_type": "binary"},
            "imbalance": {"severity": "severe"},
        }
    )

    assert _risk(risks, "Rare classes may be missing from folds")["severity"] == "high"
    target_risk = _risk(risks, "Target imbalance can hide minority errors")
    assert target_risk["severity"] == "high"
    assert "minority-class" in str(target_risk["mitigation"])


def test_safe_feature_drift_and_id_artifact_drift_are_separate() -> None:
    high = _risks(drift_evidence={"status": "completed", "feature_drift_severity": "high"})
    artifact_only = _risks(
        drift_evidence={
            "status": "completed",
            "feature_drift_severity": "low",
            "id_artifact_drift": {"severity": "high", "columns": [{"column": "id"}]},
        }
    )

    drift_risk = _risk(high, "Train/test safe-feature drift may affect leaderboard reliability")
    assert drift_risk["severity"] == "high"
    assert "drift_evidence.feature_drift_severity" in drift_risk["evidence_refs"]
    assert not any(
        risk["title"] == "Train/test safe-feature drift may affect leaderboard reliability"
        and risk["severity"] == "high"
        for risk in artifact_only
    )
    artifact_risk = _risk(artifact_only, "ID/index drift artifact excluded from feature drift")
    assert artifact_risk["severity"] == "low"
    assert artifact_risk["status"] == "mitigated_by_policy"


def test_high_cardinality_and_missingness_risks() -> None:
    risks = _risks(
        feature_diagnostics={
            "categorical_feature_diagnostics": {
                "high_cardinality_candidates": [{"column": "cat"}],
                "target_association_cautions": [{"column": "cat"}],
            },
            "missingness_diagnostics": {
                "target_associated_missingness": [{"column": "x"}],
                "train_test_missingness_shift": [{"column": "x"}],
            },
        }
    )

    high_cardinality = _risk(risks, "High-cardinality categorical features need robust encoding")
    assert high_cardinality["risk_type"] == "high_cardinality"
    assert "rare handling" in str(high_cardinality["mitigation"])
    assert "fold-fitted encoders" in str(high_cardinality["mitigation"])
    assert _risk(risks, "High-cardinality target association may be unreliable")
    assert "fold-safe imputation" in str(_risk(risks, "Missingness appears informative")["mitigation"])
    assert _risk(risks, "Missingness pattern differs between train and test")


def test_baseline_skipped_and_completed_policy_risks() -> None:
    skipped = _risks(baseline_evidence={"status": "skipped", "reason": "disabled"})
    completed = _risks(
        baseline_evidence={
            "status": "completed",
            "metric_value": 0.7,
            "preprocessing_policy": {
                "safety_checks": {"fits_preprocessing_inside_folds": True}
            },
        }
    )

    skipped_risk = _risk(skipped, "No completed baseline evidence is available")
    assert skipped_risk["status"] == "skipped"
    assert "fold-safe baseline" in str(skipped_risk["mitigation"])
    assert _risk(completed, "Completed baseline can serve as sanity floor")["severity"] == "info"
    assert not any("not auditable" in risk["title"] for risk in completed)


def test_not_testable_leakage_checks_do_not_create_high_risks() -> None:
    risks = _risks(
        leakage_evidence=[
            {"check_id": "future_time_risk", "status": "not_testable", "finding": "No time column."},
            {"check_id": "group_overlap", "status": "not_testable", "finding": "No group column."},
            {"check_id": "ranking_query_overlap", "status": "not_testable", "finding": "Non-ranking task."},
        ]
    )

    assert not [risk for risk in risks if risk["risk_type"] == "leakage" and risk["severity"] in {"high", "critical"}]


def test_risk_dedup_merges_equivalent_high_cardinality_risks() -> None:
    risks = _risks(
        feature_diagnostics={
            "categorical_feature_diagnostics": {
                "high_cardinality_candidates": [{"column": "cat"}]
            },
            "baseline_preprocessing_policy": {"high_cardinality": {"columns": ["cat"]}},
        },
    )

    matching = [
        risk for risk in risks
        if risk["title"] == "High-cardinality categorical features need robust encoding"
    ]
    assert len(matching) == 1
    assert matching[0]["risk_intent"] == "high_cardinality_encoding"
    assert matching[0]["severity"] == "medium"
    assert "feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates" in matching[0]["evidence_refs"]
    assert "feature_diagnostics.baseline_preprocessing_policy.high_cardinality" in matching[0]["evidence_refs"]
    assert validate_eda_risk_register(risks) == []


def test_exact_duplicate_high_cardinality_risks_merge_refs_and_applies_to() -> None:
    risks = deduplicate_eda_risks(
        [
            _risk_payload(
                "risk_a",
                "high_cardinality",
                "medium",
                "confirmed",
                "High-cardinality categorical features need robust encoding",
                ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"],
                ["feature_engineering"],
            ),
            _risk_payload(
                "risk_b",
                "high_cardinality",
                "medium",
                "confirmed",
                "High-cardinality categorical features need robust encoding",
                ["baseline_evidence.preprocessing_policy.high_cardinality"],
                ["validation"],
            ),
        ]
    )

    assert len(risks) == 1
    assert risks[0]["risk_intent"] == "high_cardinality_encoding"
    assert risks[0]["evidence_refs"] == [
        "feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates",
        "baseline_evidence.preprocessing_policy.high_cardinality",
    ]
    assert risks[0]["applies_to"] == ["feature_engineering", "validation"]


def test_near_duplicate_high_cardinality_encoding_titles_merge() -> None:
    risks = deduplicate_eda_risks(
        [
            _risk_payload(
                "risk_a",
                "high_cardinality",
                "medium",
                "confirmed",
                "High-cardinality categorical features need robust encoding",
                ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"],
                ["feature_engineering"],
            ),
            _risk_payload(
                "risk_b",
                "high_cardinality",
                "medium",
                "suspected",
                "High-cardinality columns require rare handling",
                ["feature_probe_evidence"],
                ["validation"],
            ),
            _risk_payload(
                "risk_c",
                "high_cardinality",
                "low",
                "suspected",
                "High-cardinality categorical encoding requires caution",
                ["baseline_evidence.preprocessing_policy.high_cardinality"],
                ["modeling"],
            ),
        ]
    )

    assert len(risks) == 1
    assert risks[0]["risk_intent"] == "high_cardinality_encoding"
    assert risks[0]["severity"] == "medium"
    assert risks[0]["status"] == "confirmed"
    assert risks[0]["confidence"] == "high"


def test_high_cardinality_encoding_and_target_association_remain_separate() -> None:
    risks = deduplicate_eda_risks(
        [
            _risk_payload(
                "risk_a",
                "high_cardinality",
                "medium",
                "confirmed",
                "High-cardinality categorical features need robust encoding",
                ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"],
                ["feature_engineering"],
            ),
            _risk_payload(
                "risk_b",
                "feature_engineering",
                "medium",
                "suspected",
                "High-cardinality target association may be unreliable",
                ["feature_diagnostics.categorical_feature_diagnostics.target_association_cautions"],
                ["validation"],
            ),
        ]
    )

    assert len(risks) == 2
    assert {risk["risk_intent"] for risk in risks} == {
        "high_cardinality_encoding",
        "high_cardinality_target_association_reliability",
    }


def test_duplicate_risk_merge_keeps_strongest_fields() -> None:
    risks = deduplicate_eda_risks(
        [
            {
                **_risk_payload(
                    "risk_a",
                    "high_cardinality",
                    "medium",
                    "suspected",
                    "High-cardinality columns require rare handling",
                    ["feature_probe_evidence"],
                    ["validation"],
                ),
                "confidence": "low",
                "mitigation": "Consider encoding.",
            },
            {
                **_risk_payload(
                    "risk_b",
                    "high_cardinality",
                    "high",
                    "confirmed",
                    "High-cardinality categorical features need robust encoding",
                    ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"],
                    ["feature_engineering"],
                ),
                "confidence": "high",
                "mitigation": "Use rare handling, frequency/hash encoding, or fold-fitted encoders; validate impact.",
            },
        ]
    )

    assert len(risks) == 1
    assert risks[0]["severity"] == "high"
    assert risks[0]["status"] == "confirmed"
    assert risks[0]["confidence"] == "high"
    assert "fold-fitted encoders" in str(risks[0]["mitigation"])


def test_risk_register_validation_flags_duplicate_intents() -> None:
    warnings = validate_eda_risk_register(
        [
            _risk_payload(
                "risk_a",
                "high_cardinality",
                "medium",
                "confirmed",
                "High-cardinality categorical features need robust encoding",
                ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"],
                ["feature_engineering"],
            ),
            _risk_payload(
                "risk_b",
                "high_cardinality",
                "medium",
                "confirmed",
                "High-cardinality columns require rare handling",
                ["feature_probe_evidence"],
                ["validation"],
            ),
        ]
    )

    assert any("duplicates risk_intent" in warning for warning in warnings)


def test_risk_register_summary_rendering_is_concise() -> None:
    risks = _risks(
        inferred_schema={"primary_id_column": "row_id", "confidence": "high"},
        metric_evidence={"requires_threshold": True},
    )
    summary = build_eda_summary(
        EdaEvidencePack(
            competition_id="risk_summary",
            created_at="2026-07-11T12:00:00+03:00",
            run_id="risk_summary_run",
            eda_risk_register=risks,
            risk_summary=risk_summary(risks),
        )
    )

    assert "## Risk register" in summary
    assert "- Risks:" in summary
    assert "High risks:" in summary
    assert "Medium risks:" in summary
    assert '"risk_id"' not in summary


def test_risk_register_summary_does_not_repeat_high_cardinality_risks() -> None:
    risks = _risks(
        feature_diagnostics={
            "categorical_feature_diagnostics": {
                "high_cardinality_candidates": [{"column": "cat"}]
            }
        },
        baseline_evidence={
            "status": "completed",
            "metric_value": 0.7,
            "preprocessing_policy": {
                "safety_checks": {"fits_preprocessing_inside_folds": True},
                "high_cardinality": {"columns": ["cat"]},
            },
        },
    )
    summary = build_eda_summary(
        EdaEvidencePack(
            competition_id="risk_summary",
            created_at="2026-07-11T12:00:00+03:00",
            run_id="risk_summary_run",
            eda_risk_register=risks,
        )
    )

    assert validate_eda_risk_register(risks) == []
    assert summary.count("High-cardinality categorical features need robust encoding") == 1


def test_risk_register_strategy_hints_reference_high_risk_ids() -> None:
    risks = _risks(
        leakage_evidence=[
            {"check_id": "target_in_test", "status": "failed", "finding": "Target present."}
        ],
        drift_evidence={"status": "completed", "feature_drift_severity": "high"},
        baseline_evidence={
            "status": "completed",
            "metric_value": 0.7,
            "preprocessing_policy": {"safety_checks": {"fits_preprocessing_inside_folds": True}},
        },
    )
    hints = build_eda_strategy_hints({"eda_risk_register": risks})
    actions = [
        RecommendedNextAction(**item)
        for category in hints.values()
        for item in category
    ]

    assert any("leakage-prone" in action.action for action in actions)
    assert any("CV/LB gap" in action.action for action in actions)
    assert all("Completed baseline can serve as sanity floor" not in action.action for action in actions)
    assert any(ref.startswith("eda_risk_register.risk_leakage_") for action in actions for ref in action.evidence_refs)


def _risks(**overrides):
    defaults = {
        "inferred_schema": {},
        "metric_evidence": {},
        "validation_evidence": {},
        "target_diagnostics": {},
        "leakage_evidence": [],
        "drift_evidence": {},
        "relationship_evidence": {},
        "feature_probe_evidence": [],
        "feature_diagnostics": {},
        "baseline_evidence": {"status": "completed", "preprocessing_policy": {"safety_checks": {"fits_preprocessing_inside_folds": True}}},
        "notebook_static_analysis": {},
    }
    defaults.update(overrides)
    return build_eda_risk_register(**defaults)


def _risk(risks, title):
    return next(risk for risk in risks if risk["title"] == title)


def _risk_payload(
    risk_id,
    risk_type,
    severity,
    status,
    title,
    evidence_refs,
    applies_to,
):
    return {
        "risk_id": risk_id,
        "risk_type": risk_type,
        "severity": severity,
        "status": status,
        "confidence": "high",
        "title": title,
        "finding": f"{title}.",
        "impact": "This can make EDA conclusions less reliable.",
        "mitigation": "Use robust, fold-safe mitigation and validate impact.",
        "applies_to": applies_to,
        "evidence_refs": evidence_refs,
        "related_actions": ["Validate the risk mitigation."],
        "limitations": [],
    }
