from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyAction,
    FinalStrategyResult,
    FinalStrategySection,
)


def test_final_strategy_result_validates_linked_strategy_contract() -> None:
    result = FinalStrategyResult(
        competition_id="generic-binary",
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
