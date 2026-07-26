from __future__ import annotations

import pytest

from kaggle_researcher.contracts.action_evidence_resolution import (
    classify_action,
    resolve_action_evidence_refs,
    resolve_final_strategy_action_evidence,
)
from kaggle_researcher.contracts.action_support import (
    FinalStrategyCompilationContext,
    UnsupportedFinalStrategyActionError,
    compile_final_strategy_action_support,
)
from kaggle_researcher.contracts.eda import EdaEvidencePack
from tests.contracts.factories import build_final_strategy_reference_catalog
from kaggle_researcher.contracts.research import ResearchHypotheses


def _hypotheses(*, temporal_claim: bool = False) -> ResearchHypotheses:
    return ResearchHypotheses.model_validate({
        "competition_id": "demo",
        "hypotheses": [
            {
                "hypothesis_id": "val_001",
                "category": "validation",
                "claim": (
                    "Temporal validation must be primary."
                    if temporal_claim
                    else "Stratified validation should be primary."
                ),
                "priority": "P0",
                "confidence_before_eda": "medium",
            },
            {
                "hypothesis_id": "metric_001",
                "category": "metric",
                "claim": "The metric needs probability predictions.",
                "priority": "P0",
                "confidence_before_eda": "medium",
            },
            {
                "hypothesis_id": "leak_001",
                "category": "leakage",
                "claim": "Target-in-test leakage must be blocked.",
                "priority": "P0",
                "confidence_before_eda": "medium",
            },
            {
                "hypothesis_id": "baseline_001",
                "category": "baseline",
                "claim": "Measure an honest baseline as a diagnostic sanity floor.",
                "priority": "P1",
                "confidence_before_eda": "low",
            },
        ],
    })


def _pack(
    *,
    validation_status: str = "confirmed",
    recommendations: list[dict[str, object]] | None = None,
    metric: dict[str, object] | None = None,
    leakage: list[dict[str, object]] | None = None,
    baseline: bool = False,
) -> EdaEvidencePack:
    return EdaEvidencePack.model_validate({
        "competition_id": "demo",
        "created_at": "2026-07-18T00:00:00Z",
        "run_id": "run",
        "validation_evidence": {
            "primary_validation": {"method": "stratified_kfold", "n_splits": 5},
        },
        "metric_evidence": metric or {},
        "leakage_evidence": leakage or [],
        "baseline_evidence": (
            {
                "metric_value": 0.83,
                "preprocessing_policy": "fold_safe",
                "model_type": "logistic_regression",
            }
            if baseline
            else {}
        ),
        "hypothesis_results": [
            {
                "hypothesis_id": "val_001",
                "category": "validation",
                "status": validation_status,
                "confidence_after_eda": "high",
                "finding": "EDA selected stratified_kfold as primary validation.",
                "evidence_refs": ["validation_evidence.primary_validation"],
                "impact_on_strategy": "Use the selected primary validation.",
            },
            *([{
                "hypothesis_id": "metric_001",
                "category": "metric",
                "status": "confirmed",
                "confidence_after_eda": "high",
                "finding": "The metric requires probabilities.",
                "evidence_refs": ["metric_evidence.requires_probabilities"],
                "impact_on_strategy": "Generate probabilities.",
            }] if metric and metric.get("requires_probabilities") else []),
            *([{
                "hypothesis_id": "leak_001",
                "category": "leakage",
                "status": "confirmed",
                "confidence_after_eda": "high",
                "finding": "A critical leakage check failed.",
                "evidence_refs": ["leakage_evidence.target_in_test"],
                "impact_on_strategy": "Block the unsafe source.",
            }] if leakage else []),
            *([{
                "hypothesis_id": "baseline_001",
                "category": "baseline",
                "status": "confirmed",
                "confidence_after_eda": "medium",
                "finding": "Honest baseline completed for the supported task type.",
                "evidence_refs": ["baseline_evidence"],
                "impact_on_strategy": "Use the baseline as a sanity floor.",
            }] if baseline else []),
        ],
        "recommended_next_actions": recommendations or [],
    })


def _catalog(pack: EdaEvidencePack, hypotheses: ResearchHypotheses):
    return build_final_strategy_reference_catalog(
        pack, research_hypotheses=hypotheses
    )


def test_missing_p0_validation_refs_are_resolved_without_downgrade() -> None:
    hypotheses = _hypotheses()
    pack = _pack()
    action = {
        "action_id": "action_val_001",
        "priority": "P0",
        "action": "Use the selected primary validation.",
        "reason": "Follow EDA.",
        "evidence_refs": [],
        "related_hypothesis_ids": ["val_001"],
    }

    report = resolve_action_evidence_refs(
        action,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )

    assert report.support_status == "supported"
    assert report.original_refs == ()
    assert report.added_refs == ("validation_evidence.primary_validation",)
    assert action["priority"] == "P0"
    assert action["evidence_refs"] == ["validation_evidence.primary_validation"]


def test_rejected_temporal_hypothesis_is_contradicted_then_replaced() -> None:
    hypotheses = _hypotheses(temporal_claim=True)
    pack = _pack(validation_status="rejected")
    action = {
        "action_id": "action_val_001",
        "priority": "P0",
        "action": "Use temporal CV as primary validation.",
        "reason": "The Scout proposed it.",
        "evidence_refs": [],
        "related_hypothesis_ids": ["val_001"],
        "validation_strategy": "temporal_cv",
    }

    direct = resolve_action_evidence_refs(
        dict(action),
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )
    assert direct.support_status == "contradicted"

    resolved, report = resolve_final_strategy_action_evidence(
        {"actions": [action], "sections": []},
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )
    repaired = resolved["actions"][0]
    action_report = report.for_action("action_val_001")
    assert repaired["validation_strategy"] == "stratified_kfold"
    assert "temporal" not in repaired["action"].lower()
    assert repaired["evidence_refs"] == ["validation_evidence.primary_validation"]
    assert action_report is not None
    assert action_report.resolution_method == "deterministic_fallback"
    assert action_report.contradictory_refs == ("hypothesis_results.val_001",)


def test_broken_explicit_ref_is_reported_and_replaced() -> None:
    hypotheses = _hypotheses()
    pack = _pack()
    action = {
        "action_id": "action_val_001",
        "priority": "P0",
        "action": "Use the selected primary validation.",
        "evidence_refs": ["validation_evidence.does_not_exist"],
        "related_hypothesis_ids": ["val_001"],
    }

    report = resolve_action_evidence_refs(
        action,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )

    assert report.unresolved_refs == ("validation_evidence.does_not_exist",)
    assert report.resolved_refs == ("validation_evidence.primary_validation",)
    assert report.support_status == "partially_supported"


def test_recommended_action_inherits_factual_refs_and_hypothesis_link() -> None:
    hypotheses = _hypotheses()
    pack = _pack(recommendations=[{
        "priority": "P0",
        "action": "Apply stratified CV for model comparison.",
        "why": "EDA selected it.",
        "evidence_refs": ["validation_evidence.primary_validation"],
        "source_categories": ["validation"],
    }])
    action = {
        "action_id": "freeze_validation",
        "priority": "P0",
        "action": "Freeze StratifiedKFold splits before training.",
        "evidence_refs": [],
        "related_hypothesis_ids": [],
    }

    report = resolve_action_evidence_refs(
        action,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )

    assert report.support_status == "supported"
    assert report.matching_recommended_action_ids == ("recommended_next_actions[0]",)
    assert action["evidence_refs"] == ["validation_evidence.primary_validation"]
    assert action["related_hypothesis_ids"] == ["val_001"]


def test_metric_and_critical_leakage_actions_get_exact_refs() -> None:
    hypotheses = _hypotheses()
    leakage = [{
        "check_id": "target_in_test",
        "status": "failed",
        "severity": "critical",
        "finding": "Target is present in a test-side source.",
        "evidence": {"column": "target"},
    }]
    pack = _pack(metric={"requires_probabilities": True}, leakage=leakage)
    catalog = _catalog(pack, hypotheses)
    payload, report = resolve_final_strategy_action_evidence(
        {"actions": [], "sections": []},
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=catalog,
    )
    by_id = {action["action_id"]: action for action in payload["actions"]}

    assert by_id["eda_metric_prediction_type"]["evidence_refs"] == [
        "metric_evidence.requires_probabilities"
    ]
    assert by_id["eda_leakage_block_target_in_test"]["evidence_refs"] == [
        "leakage_evidence.target_in_test"
    ]
    assert set(report.fallback_action_ids) >= {
        "eda_validation_primary",
        "eda_metric_prediction_type",
        "eda_leakage_block_target_in_test",
    }


def test_arbitrary_unsupported_p0_still_fails_strict_gate() -> None:
    hypotheses = _hypotheses()
    pack = _pack()
    catalog = _catalog(pack, hypotheses)
    original = {
        "actions": [{
            "action_id": "arbitrary_p0",
            "priority": "P0",
            "action": "Deploy an unsupported exotic ensemble.",
            "reason": "LLM preference.",
            "evidence_refs": [],
            "related_hypothesis_ids": [],
        }],
        "sections": [],
    }
    resolved, report = resolve_final_strategy_action_evidence(
        original,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=catalog,
    )

    with pytest.raises(UnsupportedFinalStrategyActionError):
        compile_final_strategy_action_support(
            resolved,
            original_payload=original,
            context=FinalStrategyCompilationContext(
                reference_catalog=catalog,
                action_evidence_resolutions=report.actions,
            ),
        )


def test_leak_free_baseline_wording_does_not_capture_critical_leakage_intent() -> None:
    category, intent = classify_action({
        "action": (
            "Run exp_baseline_validation with logistic regression on core safe features "
            "to establish a minimal leak-free accuracy floor."
        ),
        "related_hypothesis_ids": ["baseline_001"],
    })

    assert category == "baseline"
    assert intent == "baseline.run_sanity_baseline"


def test_explicit_critical_intent_is_reclassified_for_baseline_hypothesis() -> None:
    hypotheses = _hypotheses()
    leakage = [{
        "check_id": "target_in_test",
        "status": "passed",
        "severity": "low",
        "finding": "Target is absent from test.",
        "evidence": {"column": "target"},
    }]
    pack = _pack(leakage=leakage, baseline=True)
    original = {
        "actions": [{
            "action_id": "action_regression",
            "priority": "P0",
            "mandatory": True,
            "intent": "leakage.block_critical_issue",
            "action": "Run a leakage-safe baseline to establish an accuracy floor.",
            "reason": "Measure a reproducible benchmark.",
            "evidence_refs": [],
            "related_hypothesis_ids": ["baseline_001", "val_001", "leak_001"],
        }],
        "sections": [],
    }

    resolved, report = resolve_final_strategy_action_evidence(
        original,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )
    action_report = report.for_action("action_regression")

    assert action_report is not None
    assert action_report.original_intent == "leakage.block_critical_issue"
    assert action_report.intent == "baseline.run_sanity_baseline"
    assert action_report.alignment_status == "invalid"
    assert action_report.related_hypothesis_categories == ("baseline", "validation")
    assert action_report.matching_failed_critical_leakage_refs == ()
    assert action_report.original_mandatory_status is True
    assert action_report.normalized_mandatory_status is False
    assert action_report.resolution_method == "deterministic_fallback"
    assert set(action_report.policy_violation_codes) == {
        "intent_hypothesis_category_mismatch",
        "critical_leakage_action_without_failed_check",
    }
    assert resolved["actions"][0]["evidence_refs"] == [
        "baseline_evidence", "validation_evidence.primary_validation",
    ]
    assert resolved["actions"][0]["related_hypothesis_ids"] == [
        "baseline_001", "val_001",
    ]


def test_matching_failed_critical_check_preserves_strict_leakage_action() -> None:
    hypotheses = _hypotheses()
    leakage = [{
        "check_id": "target_in_test",
        "status": "failed",
        "severity": "critical",
        "finding": "Target column appears in the test table.",
        "evidence": {"column": "target"},
    }]
    pack = _pack(leakage=leakage)
    action = {
        "action_id": "critical_target_leak",
        "priority": "P0",
        "intent": "leakage.block_critical_issue",
        "action": "Remove target from the model contract after leakage check target_in_test.",
        "evidence_refs": [],
        "related_hypothesis_ids": ["leak_001"],
    }

    report = resolve_action_evidence_refs(
        action,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )

    assert report.intent == "leakage.block_critical_issue"
    assert report.matching_failed_critical_leakage_refs == (
        "leakage_evidence.target_in_test",
    )
    assert report.failed_critical_leakage_count == 1
    assert report.normalized_mandatory_status is True
    assert action["evidence_refs"] == ["leakage_evidence.target_in_test"]


def test_warning_only_leakage_reclassifies_to_diagnostic_audit() -> None:
    hypotheses = _hypotheses()
    leakage = [{
        "check_id": "group_overlap",
        "status": "warning",
        "severity": "high",
        "finding": "Potential household overlap requires an audit.",
        "evidence": {"column": "household_id"},
    }]
    pack = _pack(leakage=leakage)
    action = {
        "action_id": "warning_overlap",
        "priority": "P0",
        "intent": "leakage.block_critical_issue",
        "action": "Audit leakage warning from check group_overlap before training.",
        "evidence_refs": [],
        "related_hypothesis_ids": ["leak_001"],
    }

    report = resolve_action_evidence_refs(
        action,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )

    assert report.intent == "leakage.audit_warning"
    assert report.normalized_mandatory_status is False
    assert report.resolved_refs == ("leakage_evidence.group_overlap",)
    assert "critical_leakage_action_without_failed_check" in report.policy_violation_codes


def test_passed_checks_cannot_support_critical_remediation() -> None:
    hypotheses = _hypotheses()
    leakage = [{
        "check_id": "target_in_test",
        "status": "passed",
        "severity": "low",
        "finding": "Target is absent from test.",
        "evidence": {"column": "target"},
    }]
    pack = _pack(leakage=leakage)
    action = {
        "action_id": "false_positive_leakage",
        "priority": "P0",
        "intent": "leakage.block_critical_issue",
        "action": "Remove the target after leakage check target_in_test.",
        "evidence_refs": ["leakage_evidence.target_in_test"],
        "related_hypothesis_ids": ["leak_001"],
    }

    report = resolve_action_evidence_refs(
        action,
        eda_evidence_pack=pack,
        research_hypotheses=hypotheses,
        allowed_reference_index=_catalog(pack, hypotheses),
    )

    assert report.intent == "leakage.run_required_checks"
    assert report.matching_failed_critical_leakage_refs == ()
    assert report.normalized_mandatory_status is False
    assert report.support_status == "unsupported"
