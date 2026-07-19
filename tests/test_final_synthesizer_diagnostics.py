from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.final_strategy import (
    FinalStrategyResult,
    upgrade_final_strategy_v1_to_v2,
)
from kaggle_researcher.contracts.final_strategy_draft import (
    FinalStrategyDraftReferenceError,
    compile_final_strategy_draft,
)
from kaggle_researcher.contracts.reference_catalog import (
    build_final_strategy_reference_catalog,
)
from kaggle_researcher.eda.schemas import (
    EdaEvidencePack,
    ResearchHypotheses,
    ResearchHypothesis,
)
from kaggle_researcher.reasoning.model_registry import (
    is_valid_model_comparison,
    resolve_model_identity,
    supported_models,
)
from kaggle_researcher.reasoning.strategy_compaction import compact_final_strategy
from tests.test_deterministic_fallback_strategy import _fallback, _titanic_evidence


FIXTURES = Path(__file__).parent / "fixtures" / "reasoning"


def test_recorded_failure_was_schema_not_json_parsing() -> None:
    diagnostic = json.loads(
        (FIXTURES / "final_synthesis_contract_failure.json").read_text(encoding="utf-8")
    )

    assert diagnostic["initial"]["json_parse_succeeded"] is True
    assert diagnostic["initial"]["issue_counts"]["extra_forbidden"] == 46
    assert diagnostic["initial"]["issue_counts"]["literal_error"] == 28
    assert diagnostic["repair"]["preserved_incompatible_fields"] is True
    assert diagnostic["fallback_reason"].startswith("Deterministic repair failed")


def test_recorded_draft_fields_and_unicode_origin_compile_to_canonical_contract() -> None:
    pack, hypotheses = _minimal_contract_inputs()
    catalog = build_final_strategy_reference_catalog(
        pack, research_hypotheses=hypotheses
    )
    raw = _recorded_shape_draft()

    compiled = compile_final_strategy_draft(raw, catalog)

    assert compiled["schema_version"] == "2.0"
    assert "executive_summary" not in compiled
    assert "warnings" not in compiled
    assert "narrative" not in compiled["sections"][0]
    action = compiled["sections"][1]["actions"][0]
    assert "support_refs" not in action
    assert action["evidence_origin"] == "EDA-confirmed"
    assert action["evidence_refs"] == ["validation_evidence.primary_validation"]


def test_unknown_draft_evidence_remains_strictly_rejected() -> None:
    pack, hypotheses = _minimal_contract_inputs()
    catalog = build_final_strategy_reference_catalog(
        pack, research_hypotheses=hypotheses
    )
    raw = _recorded_shape_draft()
    raw["sections"][1]["actions"][0]["support_refs"] = [
        {"namespace": "evidence", "ref_id": "invented_evidence.path"}
    ]

    with pytest.raises(FinalStrategyDraftReferenceError):
        compile_final_strategy_draft(raw, catalog)


def test_model_aliases_resolve_and_self_comparison_is_rejected() -> None:
    left = resolve_model_identity("HistGradientBoostingClassifier")
    right = resolve_model_identity("sklearn_hist_gradient_boosting_classifier")
    candidate = resolve_model_identity("LogisticRegression")

    assert left == right
    assert not is_valid_model_comparison(left, right)
    assert is_valid_model_comparison(left, candidate)
    assert all(item.available for item in supported_models("binary_classification"))


def test_titanic_compaction_groups_families_and_enforces_budget_and_order() -> None:
    evidence = _titanic_evidence()
    evidence["baseline_evidence"]["metric_value"] = 0.832729
    evidence["inferred_schema"].update({
        "prediction_column": "Survived",
        "sample_submission_table": "gender_submission.csv",
    })
    compacted = compact_final_strategy(_fallback(evidence), evidence_pack=evidence)
    family_ids = {item.family_id for item in compacted.feature_experiment_families}
    core_ids = [item.experiment_id for item in compacted.core_experiments]
    backlog_ids = [item.experiment_id for item in compacted.experiment_backlog]

    assert {
        "missingness_age",
        "relationship_family",
        "cabin_representation",
        "name_representation",
        "ticket_representation",
        "fare_representation",
    } <= family_ids
    assert len(core_ids) <= 8
    assert backlog_ids
    assert "baseline" in core_ids[0]
    assert "threshold" in core_ids[-2]
    assert "submission" in core_ids[-1]
    threshold = compacted.core_experiments[-2]
    assert threshold.fit_scope == "oof_only"
    assert {
        "baseline_reproduced", "folds_locked", "provisional_model_selected",
        "oof_predictions_available",
    } <= set(threshold.dependencies)
    assert "0.832729" in next(
        section.summary for section in compacted.sections
        if section.section_id == "executive_summary"
    )
    assert compacted.quality_metrics.root_refs_for_specific_claims == 0
    assert compacted.quality_metrics.max_evidence_refs_per_action <= 4
    assert not any(
        action.model_dump(mode="json").get("evidence_bindings")
        for action in compacted.actions
    )
    serialized = compacted.model_dump(mode="json")
    assert all("evidence_bindings" not in action for action in serialized["actions"])


def test_generic_and_regression_compaction_stay_task_appropriate() -> None:
    generic = _titanic_evidence()
    generic["inferred_schema"] = {
        "target_column": "target",
        "tables": [{"path": "train.csv", "columns": [
            {"name": "target", "dtype": "Int64"},
            {"name": "x1", "dtype": "Float64"},
        ]}],
    }
    generic["feature_diagnostics"] = {"safe_feature_columns": ["x1"]}
    generic["feature_probe_evidence"] = []
    generic["baseline_ablation_evidence"] = {}
    generic_result = compact_final_strategy(_fallback(generic), evidence_pack=generic)
    assert generic_result.feature_experiment_families == []

    regression = _titanic_evidence()
    regression["metric_evidence"] = {
        "metric_name": "rmse",
        "task_type": "regression",
        "greater_is_better": False,
        "requires_threshold": False,
    }
    regression["baseline_evidence"]["model_type"] = "HistGradientBoostingRegressor"
    regression_result = compact_final_strategy(_fallback(regression), evidence_pack=regression)
    assert all("threshold" not in item.experiment_id for item in regression_result.experiments)
    assert all(
        not item.baseline_canonical_family_id
        or "regression" in item.baseline_canonical_family_id
        for item in regression_result.experiments
    )


def test_v1_migration_deduplicates_legacy_evidence_previews() -> None:
    evidence = _titanic_evidence()
    legacy = _fallback(evidence).model_dump(mode="json")
    legacy["schema_version"] = "1.0"
    for action in legacy["actions"]:
        action["evidence_bindings"] = [
            {
                "ref": action["evidence_refs"][0],
                "resolved_value_preview": "repeated-preview",
                "role": "primary",
            }
        ]

    migrated = upgrade_final_strategy_v1_to_v2(
        legacy, evidence_pack=evidence
    )

    assert migrated["schema_version"] == "2.0"
    assert migrated["evidence_catalog"]
    assert all("evidence_bindings" not in action for action in migrated["actions"])


def _minimal_contract_inputs() -> tuple[EdaEvidencePack, ResearchHypotheses]:
    hypotheses = ResearchHypotheses(
        competition_id="generic",
        hypotheses=[ResearchHypothesis(
            hypothesis_id="val_001",
            category="validation",
            claim="Use stratified validation.",
            priority="P0",
            confidence_before_eda="high",
        )],
    )
    pack = EdaEvidencePack(
        competition_id="generic",
        created_at="2026-07-19T00:00:00Z",
        run_id="generic_run",
        validation_evidence={
            "primary_validation": {"method": "stratified_kfold"}
        },
        hypothesis_results=[{
            "hypothesis_id": "val_001",
            "category": "validation",
            "status": "confirmed",
            "confidence_after_eda": "high",
            "finding": "Stratified validation is appropriate.",
            "evidence_refs": ["validation_evidence.primary_validation"],
            "impact_on_strategy": "Use the selected folds.",
        }],
    )
    return pack, hypotheses


def _recorded_shape_draft() -> dict[str, object]:
    section_ids = [
        "executive_summary", "metric_and_validation", "dataset_facts_from_eda",
        "leakage_and_data_quality", "drift_and_leaderboard_risk", "baseline_findings",
        "feature_priorities", "modeling_plan", "experiments_queue", "what_not_to_do",
        "first_48_hours",
    ]
    sections = []
    for section_id in section_ids:
        section: dict[str, object] = {
            "section_id": section_id,
            "title": section_id.replace("_", " ").title(),
            "narrative": "Evidence-backed guidance.",
            "actions": [],
            "evidence_summary_refs": [{
                "namespace": "evidence",
                "ref_id": "validation_evidence.primary_validation",
            }],
            "limitations": [],
        }
        if section_id == "metric_and_validation":
            section["actions"] = [{
                "action_id": "validation_action",
                "priority": "P0",
                "action": "Use stratified validation.",
                "reason": "Validated EDA policy.",
                "support_refs": [{
                    "namespace": "evidence",
                    "ref_id": "validation_evidence.primary_validation",
                }],
                "related_hypothesis_ids": ["val_001"],
                "experiment_ids": [],
                "evidence_origin": "EDA‑confirmed",
                "confidence": "high",
                "limitations": [],
            }]
        sections.append(section)
    return {
        "schema_version": "1.0",
        "competition_id": "generic",
        "executive_summary": "Use the validated policy.",
        "sections": sections,
        "warnings": ["Bounded warning."],
        "limitations": [],
    }
