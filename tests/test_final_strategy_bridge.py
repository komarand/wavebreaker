from __future__ import annotations

from types import SimpleNamespace

from kaggle_researcher.contracts.final_strategy import REQUIRED_SECTION_IDS
from kaggle_researcher.contracts.final_strategy_protocol import (
    PromptFingerprint,
    StrategySelectionDraft,
)
import pytest

from kaggle_researcher.reasoning.final_strategy_bridge import (
    StrategyBridgeError,
    freeze_strategy_selection,
    validate_skeleton_integrity,
)
from kaggle_researcher.reasoning.final_strategy_context import FinalStrategySelectionContext
from tests.test_final_synthesizer import _eda_pack, _plan, _research_hypotheses


def _context(*, baseline: bool = False, threshold: bool = False, source: bool = False):
    return FinalStrategySelectionContext(
        competition_id="generic-binary", task_type="binary_classification",
        metric_contract={"metric_name": "roc_auc", "requires_threshold": threshold},
        validation_contract={"primary_validation": {"method": "stratified_kfold"}},
        schema_summary={"tables": [{"columns": [{"name": "Age"}, {"name": "x1"}]}]},
        baseline_summary={"status": "completed"} if baseline else {},
        ablation_summary={}, drift_summary={}, leakage_summary=[], feature_diagnostic_summary={},
        hypothesis_catalog=[{
            "hypothesis_id": "val_001", "category": "validation", "statement": "Use stratified folds.",
            "priority": "P0", "source_refs": ["retrieved-1"] if source else [], "testability": "tested",
            "eda_result_status": "confirmed", "relevant_evidence_refs": ["validation_evidence.primary_validation"],
        }],
        source_catalog=([{"source_ref": "retrieved-1", "source_type": "kaggle", "title": "Validated source", "claim_summary": "Use stratified folds."}] if source else []), evidence_catalog=[{
            "evidence_ref": "validation_evidence.primary_validation",
            "value_preview": {"method": "stratified_kfold"}, "value_type": "dict",
            "specificity": "object", "semantic_tags": ["validation"], "required": True,
        }],
        model_catalog=[
            {"canonical_family_id": "sklearn_hist_gradient_boosting", "implementation_id": "sklearn.ensemble.HistGradientBoostingClassifier", "display_name": "HistGradientBoostingClassifier", "available": True, "capabilities": {}},
            {"canonical_family_id": "sklearn_logistic_regression", "implementation_id": "sklearn.linear_model.LogisticRegression", "display_name": "LogisticRegression", "available": True, "capabilities": {}},
        ],
        safety_constraint_catalog=[], validation_requirement_catalog=[],
        required_section_ids=list(REQUIRED_SECTION_IDS),
        strategy_limits={"max_actions": 15, "max_core_experiments": 8, "max_backlog_experiments": 12, "max_first_24h_experiments": 4, "max_first_48h_experiments": 8},
        context_character_budget=60000,
    )


def _synthesis_context():
    return SimpleNamespace(
        plan_data=_plan(), eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        research_hypotheses=_research_hypotheses(),
    )


def _fingerprint():
    return PromptFingerprint(
        prompt_name="selection", prompt_version="2.0", system_prompt_hash="a" * 64,
        user_template_hash="b" * 64, output_schema_version="2.0",
        context_policy_version="2.0", fingerprint="c" * 64,
    )


def _action():
    return {
        "client_action_key": "validation", "action_kind": "validation_setup",
        "action": "Lock stratified validation.", "priority": "P0", "confidence": "high",
        "reason": "The validated contract requires it.",
        "primary_evidence_refs": ["validation_evidence.primary_validation"],
        "supporting_evidence_refs": [], "limitation_evidence_refs": [], "source_refs": [],
        "motivating_hypothesis_ids": [], "safety_hypothesis_ids": [],
        "validation_context_ids": ["val_001"], "rejected_hypothesis_ids": [],
        "safety_constraint_ids": [], "validation_requirement_ids": [],
        "feature_metadata": None, "dependencies": [], "limitations": [],
    }


def _experiment(key: str, kind: str, name: str, *, dependencies=None, same_model=False):
    return {
        "client_experiment_key": key, "experiment_kind": kind, "name": name,
        "priority": "P1", "confidence": "medium", "hypothesis": f"Test {name}.",
        "exact_change": f"Change only {name}.", "family_key": None,
        "model_family_id": "HistGradientBoostingClassifier",
        "comparison_model_family_id": (
            "sklearn_hist_gradient_boosting_classifier" if same_model else None
        ),
        "validation_strategy": "stratified_kfold", "metric_name": "roc_auc",
        "acceptance_rule": "Keep only a stable paired OOF result.", "estimated_cost": "low",
        "primary_evidence_refs": ["validation_evidence.primary_validation"],
        "supporting_evidence_refs": [], "source_refs": [],
        "motivating_hypothesis_ids": ["val_001"], "safety_hypothesis_ids": [],
        "validation_context_ids": [], "dependencies": dependencies or [],
        "risks": ["Fold variance."], "limitations": [],
    }


def _sections():
    return [
        {"section_id": section, "selected_action_keys": ["validation"] if section in {"executive_summary", "metric_and_validation", "first_48_hours"} else [],
         "selected_family_keys": [], "selected_experiment_keys": [], "summary_intent": "Render the frozen plan."}
        for section in REQUIRED_SECTION_IDS
    ]


def test_bridge_normalizes_models_removes_self_comparison_and_budgets() -> None:
    experiments = [
        _experiment(f"exp_{index}", "feature_family", f"Feature experiment {index}", same_model=index == 9)
        for index in range(10)
    ]
    draft = StrategySelectionDraft.model_validate({
        "schema_version": "2.0", "selected_actions": [_action()],
        "feature_experiment_families": [], "candidate_experiments": experiments,
        "proposed_core_experiment_ids": [f"exp_{index}" for index in range(8)],
        "proposed_backlog_experiment_ids": ["exp_8", "exp_9"],
        "section_plan": _sections(), "limitations": [],
    })
    skeleton, diagnostic = freeze_strategy_selection(
        draft, synthesis_context=_synthesis_context(), selection_context=_context(),
        selection_status="llm_success", selection_prompt_fingerprint=_fingerprint(),
    )
    assert len(skeleton.core_experiments) == 8
    assert len(skeleton.experiment_backlog) == 1
    assert diagnostic["self_model_comparisons_removed"] == 1
    assert diagnostic["first_48h_experiment_count"] <= 8
    assert len(skeleton.skeleton_hash) == 64


def test_bridge_keeps_baseline_first_and_threshold_oof_downstream() -> None:
    experiments = [
        _experiment("baseline", "baseline_reproduction", "Reproduce recorded baseline"),
        _experiment("feature", "feature_family", "Feature family comparison", dependencies=["baseline"]),
        _experiment("oof", "oof_generation", "Generate OOF predictions", dependencies=["feature"]),
        _experiment("threshold", "threshold_postprocessing", "OOF threshold selection", dependencies=["oof", "provisional_model_selected", "oof_predictions_available"]),
    ]
    draft = StrategySelectionDraft.model_validate({
        "schema_version": "2.0", "selected_actions": [_action()],
        "feature_experiment_families": [], "candidate_experiments": experiments,
        "proposed_core_experiment_ids": [item["client_experiment_key"] for item in experiments],
        "proposed_backlog_experiment_ids": [], "section_plan": _sections(), "limitations": [],
    })
    skeleton, _ = freeze_strategy_selection(
        draft, synthesis_context=_synthesis_context(),
        selection_context=_context(baseline=True, threshold=True),
        selection_status="llm_success", selection_prompt_fingerprint=_fingerprint(),
    )
    names = [item["name"] for item in skeleton.core_experiments]
    assert names[0] == "Reproduce recorded baseline"
    assert names[-1] == "OOF threshold selection"
    threshold = skeleton.core_experiments[-1]
    assert threshold["fit_scope"] == "oof_only"
    assert "oof_predictions_available" in threshold["dependencies"]


def test_bridge_preserves_source_hypothesis_eda_action_provenance() -> None:
    action = _action()
    action["source_refs"] = ["retrieved-1"]
    action["motivating_hypothesis_ids"] = ["val_001"]
    action["validation_context_ids"] = []
    draft = StrategySelectionDraft.model_validate({
        "schema_version": "2.0", "selected_actions": [action],
        "feature_experiment_families": [], "candidate_experiments": [],
        "proposed_core_experiment_ids": [], "proposed_backlog_experiment_ids": [],
        "section_plan": _sections(), "limitations": [],
    })
    skeleton, _ = freeze_strategy_selection(
        draft, synthesis_context=_synthesis_context(), selection_context=_context(source=True),
        selection_status="llm_success", selection_prompt_fingerprint=_fingerprint(),
    )
    assert skeleton.actions[0]["source_refs"] == ["retrieved-1"]
    assert skeleton.source_to_hypothesis_links[0]["source_ref"] == "retrieved-1"
    assert skeleton.hypothesis_to_eda_links[0]["hypothesis_id"] == "val_001"
    assert skeleton.action_provenance[0]["eda_result_refs"] == ["validation_evidence.primary_validation"]


def test_bridge_places_remaining_p2_ideas_in_backlog() -> None:
    baseline = _experiment("baseline", "baseline_reproduction", "Reproduce recorded baseline")
    p2 = _experiment("idea", "model_comparison", "Optional distinct model comparison")
    p2["priority"] = "P2"
    draft = StrategySelectionDraft.model_validate({
        "contract_family": "strategy_selection_draft",
        "schema_version": "2.0",
        "selected_actions": [_action()],
        "feature_experiment_families": [],
        "candidate_experiments": [baseline, p2],
        "proposed_core_experiment_ids": ["baseline", "idea"],
        "proposed_backlog_experiment_ids": [],
        "section_plan": _sections(),
        "limitations": [],
    })
    skeleton, _ = freeze_strategy_selection(
        draft,
        synthesis_context=_synthesis_context(),
        selection_context=_context(baseline=True),
        selection_status="llm_success",
        selection_prompt_fingerprint=_fingerprint(),
    )
    assert [item["name"] for item in skeleton.core_experiments] == [
        "Reproduce recorded baseline"
    ]
    assert [item["name"] for item in skeleton.experiment_backlog] == [
        "Optional distinct model comparison"
    ]


def test_bridge_rejects_unknown_section_client_key() -> None:
    sections = _sections()
    sections[0]["selected_action_keys"] = ["invented"]
    draft = StrategySelectionDraft.model_validate({
        "contract_family": "strategy_selection_draft",
        "schema_version": "2.0",
        "selected_actions": [_action()],
        "feature_experiment_families": [],
        "candidate_experiments": [],
        "proposed_core_experiment_ids": [],
        "proposed_backlog_experiment_ids": [],
        "section_plan": sections,
        "limitations": [],
    })
    with pytest.raises(StrategyBridgeError, match="unknown selection client-key"):
        freeze_strategy_selection(
            draft,
            synthesis_context=_synthesis_context(),
            selection_context=_context(),
            selection_status="llm_success",
            selection_prompt_fingerprint=_fingerprint(),
        )


def test_skeleton_hash_detects_nested_mutation() -> None:
    draft = StrategySelectionDraft.model_validate({
        "contract_family": "strategy_selection_draft",
        "schema_version": "2.0",
        "selected_actions": [_action()],
        "feature_experiment_families": [],
        "candidate_experiments": [],
        "proposed_core_experiment_ids": [],
        "proposed_backlog_experiment_ids": [],
        "section_plan": _sections(),
        "limitations": [],
    })
    skeleton, _ = freeze_strategy_selection(
        draft,
        synthesis_context=_synthesis_context(),
        selection_context=_context(),
        selection_status="llm_success",
        selection_prompt_fingerprint=_fingerprint(),
    )
    validate_skeleton_integrity(skeleton)
    skeleton.actions[0]["priority"] = "P2"
    with pytest.raises(StrategyBridgeError, match="no longer matches"):
        validate_skeleton_integrity(skeleton)
