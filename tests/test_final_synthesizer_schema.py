from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyAction,
    FinalStrategyResult,
    FinalStrategySection,
)
from kaggle_researcher.contracts.final_strategy import (
    upgrade_legacy_final_strategy_synthesis_status,
)


def test_final_strategy_result_validates_linked_strategy_contract() -> None:
    result = FinalStrategyResult(
        competition_id="generic-binary",
        synthesis_status="llm_success",
        llm_output_valid=True,
        repair_attempted=False,
        repair_succeeded=False,
        fallback_used=False,
        synthesis_diagnostics_path=None,
        task_type="binary_classification",
        metric={"name": "roc_auc"},
        recommended_validation="stratified_kfold",
        sections=[
            FinalStrategySection(
                section_id="validation",
                title="Validation",
                summary="Use generic iid classification validation evidence.",
                action_ids=["action_validation"],
                evidence_refs=["validation_evidence.primary_validation"],
                related_hypothesis_ids=["val_001"],
            )
        ],
        actions=[
            _action(
                action_id="action_validation",
                validation_strategy="stratified_kfold",
            )
        ],
        source_to_hypothesis_links=[
            {
                "source_ref": "retrieved-1",
                "source_claim": "Sources describe iid binary classification.",
                "hypothesis_id": "val_001",
            }
        ],
        hypothesis_to_eda_links=[
            {
                "hypothesis_id": "val_001",
                "eda_result_ref": "validation_evidence.primary_validation",
            }
        ],
    )

    assert result.actions[0].evidence_refs == [
        "validation_evidence.primary_validation"
    ]
    assert result.actions[0].related_hypothesis_ids == ["val_001"]
    assert result.actions[0].eda_result_refs == [
        "validation_evidence.primary_validation"
    ]


@pytest.mark.parametrize(
    "validation_strategy",
    [
        "stratified_kfold",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
        "temporal_holdout",
        "temporal_cv",
        "ranking_group_cv",
        "custom_required",
    ],
)
def test_final_strategy_action_supports_generic_validation_outcomes(
    validation_strategy: str,
) -> None:
    action = _action(validation_strategy=validation_strategy)

    assert action.validation_strategy == validation_strategy


@pytest.mark.parametrize(
    "updates",
    [
        {"synthesis_status": "llm_success", "fallback_used": True},
        {"synthesis_status": "repaired_success", "repair_succeeded": False},
        {
            "synthesis_status": "degraded_fallback",
            "llm_output_valid": True,
            "limitations": ["A deterministic fallback was used."],
        },
        {
            "synthesis_status": "degraded_fallback",
            "llm_output_valid": False,
            "repair_attempted": False,
            "repair_succeeded": False,
            "fallback_used": True,
            "limitations": [],
        },
    ],
)
def test_contradictory_synthesis_states_fail_validation(
    updates: dict[str, object],
) -> None:
    payload = _result_payload()
    payload.update(updates)

    with pytest.raises(ValidationError):
        FinalStrategyResult.model_validate(payload)


def test_legacy_strategy_requires_explicit_upgrade_before_validation() -> None:
    legacy = _result_payload()
    for field in (
        "synthesis_status",
        "llm_output_valid",
        "repair_attempted",
        "repair_succeeded",
        "fallback_used",
        "synthesis_diagnostics_path",
    ):
        legacy.pop(field)

    with pytest.raises(ValidationError):
        FinalStrategyResult.model_validate(legacy)

    upgraded = upgrade_legacy_final_strategy_synthesis_status(legacy)
    result = FinalStrategyResult.model_validate(upgraded)
    assert result.synthesis_status == "degraded_fallback"
    assert result.fallback_used is True


def test_missing_evidence_refs_on_action_fails_validation() -> None:
    with pytest.raises(ValidationError):
        FinalStrategyAction(
            action_id="action_validation",
            priority="P0",
            action="Use the selected validation policy.",
            reason="The EDA evidence selected this policy.",
            evidence_refs=[],
            related_hypothesis_ids=["val_001"],
            eda_result_refs=["validation_evidence.primary_validation"],
        )


def test_action_accepts_required_contract_fields_only() -> None:
    action = FinalStrategyAction(
        priority="P0",
        action="Use the selected validation policy.",
        reason="The EDA evidence selected this policy.",
        evidence_refs=["validation_evidence.primary_validation"],
        related_hypothesis_ids=["val_001"],
    )

    assert action.action_id is None
    assert action.eda_result_refs == []


def _action(
    *,
    action_id: str = "action_validation",
    validation_strategy: str = "stratified_kfold",
) -> FinalStrategyAction:
    return FinalStrategyAction(
        action_id=action_id,
        priority="P0",
        action="Use the selected validation policy for model comparison.",
        reason="Source claims, Scout hypothesis, and EDA evidence point to this validation policy.",
        evidence_refs=["validation_evidence.primary_validation"],
        related_hypothesis_ids=["val_001"],
        source_claim="Sources describe a generic tabular supervised learning setup.",
        source_refs=["retrieved-1"],
        eda_result_refs=["validation_evidence.primary_validation"],
        validation_strategy=validation_strategy,
        confidence="high",
    )


def _result_payload() -> dict[str, object]:
    return {
        "competition_id": "generic-binary",
        "synthesis_status": "llm_success",
        "llm_output_valid": True,
        "repair_attempted": False,
        "repair_succeeded": False,
        "fallback_used": False,
        "synthesis_diagnostics_path": None,
        "sections": [{
            "section_id": "validation",
            "title": "Validation",
            "summary": "Use validation evidence.",
            "action_ids": ["action_validation"],
        }],
        "actions": [_action().model_dump(mode="json")],
    }
