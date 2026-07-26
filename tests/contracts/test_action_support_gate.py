from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.action_support import (
    ACTION_SUPPORT_PHASE,
    ActionReferenceResolutionDiagnostics,
    FinalStrategyCompilationContext,
    UnsupportedFinalStrategyActionError,
    compile_final_strategy_action_support,
    enforce_action_evidence_support,
)
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.final_strategy import FinalStrategyAction, FinalStrategyResult
from tests.contracts.factories import build_final_strategy_reference_catalog


EVIDENCE_REF = "validation_evidence.primary_validation"
HYPOTHESIS_ID = "eda_hypothesis_005"


def _pack() -> EdaEvidencePack:
    return EdaEvidencePack(
        competition_id="demo",
        created_at="2026-07-14T00:00:00Z",
        run_id="run",
        validation_evidence={"primary_validation": {"method": "group_kfold"}},
        hypothesis_results=[{
            "hypothesis_id": HYPOTHESIS_ID,
            "category": "feature",
            "status": "not_testable",
            "confidence_after_eda": "low",
            "finding": "The temporal feature family was not testable.",
            "evidence_refs": [],
            "impact_on_strategy": "Keep it as an investigation only.",
            "limitations": ["Required timestamps were unavailable."],
        }],
        eda_risks=[{
            "risk_id": "risk_without_evidence",
            "risk_type": "leakage",
            "severity": "medium",
            "status": "suspected",
            "confidence": "low",
            "title": "Unresolved leakage",
            "finding": "Leakage was not measured.",
            "impact": "Impact is unknown.",
            "evidence_refs": [],
        }],
    )


def _context() -> FinalStrategyCompilationContext:
    return FinalStrategyCompilationContext(
        reference_catalog=build_final_strategy_reference_catalog(_pack())
    )


def _diagnostics(*refs: str) -> ActionReferenceResolutionDiagnostics:
    return ActionReferenceResolutionDiagnostics(
        original_refs=tuple(refs),
        unresolved_refs=tuple(refs),
    )


def test_action_with_valid_factual_evidence_is_kept() -> None:
    decision = enforce_action_evidence_support(
        {
            "action_id": "keep-action", "priority": "P0",
            "action": "Use grouped validation.", "evidence_refs": [EVIDENCE_REF],
        },
        _diagnostics(),
        _context(),
    )

    assert decision.decision == "keep"
    assert decision.resulting_evidence_refs == (EVIDENCE_REF,)
    assert "catalog-validated factual evidence" in decision.reason


def test_research_action_with_unresolved_hypothesis_is_downgraded_safely() -> None:
    decision = enforce_action_evidence_support(
        {
            "action_id": "investigate-temporal", "priority": "P1",
            "action": "Test the temporal feature family.", "evidence_refs": [],
            "related_hypothesis_ids": [HYPOTHESIS_ID],
        },
        _diagnostics(HYPOTHESIS_ID),
        _context(),
    )

    assert decision.decision == "downgrade"
    assert decision.resulting_evidence_refs == (
        f"hypothesis_results.{HYPOTHESIS_ID}",
    )
    assert decision.limitation is not None


def test_unsupported_optional_action_is_dropped_without_placeholder() -> None:
    decision = enforce_action_evidence_support(
        {
            "action_id": "optional-model", "priority": "P2",
            "action": "Deploy an unsupported model.", "evidence_refs": [],
        },
        _diagnostics("risk_without_evidence"),
        _context(),
    )

    assert decision.decision == "drop"
    assert decision.resulting_evidence_refs == ()
    assert "no factual evidence" in decision.reason
    assert not any(
        placeholder in decision.resulting_evidence_refs
        for placeholder in ("unknown", "unresolved", "missing_evidence", "generated_by_llm")
    )


def test_unsupported_p0_action_raises_early_typed_error() -> None:
    payload = {
        "actions": [{
            "action_id": "mandatory-action", "priority": "P0",
            "action": "Use the unsupported strategy.", "reason": "LLM suggestion.",
            "evidence_refs": [], "related_hypothesis_ids": [HYPOTHESIS_ID],
        }],
        "sections": [],
    }
    original = {
        **payload,
        "actions": [{**payload["actions"][0], "evidence_refs": ["risk_without_evidence"]}],
    }

    with pytest.raises(UnsupportedFinalStrategyActionError) as raised:
        compile_final_strategy_action_support(
            payload,
            original_payload=original,
            context=_context(),
        )

    error = raised.value
    assert error.action_id == "mandatory-action"
    assert error.priority == "P0"
    assert error.original_refs == ("risk_without_evidence",)
    assert error.unresolved_refs == ("risk_without_evidence",)
    assert error.phase == ACTION_SUPPORT_PHASE
    assert error.compilation_report.failed_actions[0].reason == error.decision_reason


def test_compilation_drops_optional_copies_and_never_returns_empty_action() -> None:
    supported = {
        "action_id": "supported", "priority": "P0",
        "action": "Use grouped validation.", "reason": "EDA measured it.",
        "evidence_refs": [EVIDENCE_REF],
        "related_hypothesis_ids": [HYPOTHESIS_ID],
    }
    unsupported = {
        "action_id": "unsupported", "priority": "P2",
        "action": "Deploy an unsupported model.", "reason": "LLM suggestion.",
        "evidence_refs": [], "related_hypothesis_ids": [HYPOTHESIS_ID],
    }
    payload = {
        "competition_id": "demo",
        "synthesis_status": "llm_success",
        "llm_output_valid": True,
        "repair_attempted": False,
        "repair_succeeded": False,
        "fallback_used": False,
        "synthesis_diagnostics_path": None,
        "actions": [supported, unsupported],
        "sections": [{
            "section_id": "optional", "title": "Optional", "summary": "Optional ideas.",
            "actions": [dict(unsupported)],
        }],
    }
    original = {
        **payload,
        "actions": [supported, {**unsupported, "evidence_refs": ["risk_without_evidence"]}],
        "sections": [{
            **payload["sections"][0],
            "actions": [{**unsupported, "evidence_refs": ["risk_without_evidence"]}],
        }],
    }

    compiled, report = compile_final_strategy_action_support(
        payload,
        original_payload=original,
        context=_context(),
    )

    assert [action["action_id"] for action in compiled["actions"]] == ["supported"]
    assert compiled["sections"] == []
    assert all(action["evidence_refs"] for action in compiled["actions"])
    assert report.kept_actions[0].action_id == "supported"
    assert report.dropped_actions[0].action_id == "unsupported"
    assert report.dropped_actions[0].reason
    FinalStrategyResult.model_validate(compiled)


def test_compilation_applies_downgrade_fields_without_rewriting_action() -> None:
    action = {
        "action_id": "research", "priority": "P1",
        "action": "Investigate the temporal feature family.",
        "reason": "The hypothesis remains unresolved.",
        "evidence_refs": [], "related_hypothesis_ids": [HYPOTHESIS_ID],
    }
    original = {"actions": [{**action, "evidence_refs": [HYPOTHESIS_ID]}], "sections": []}

    compiled, report = compile_final_strategy_action_support(
        {"actions": [action], "sections": []},
        original_payload=original,
        context=_context(),
    )
    compiled_action = compiled["actions"][0]

    assert compiled_action["action"] == action["action"]
    assert compiled_action["evidence_refs"] == [f"hypothesis_results.{HYPOTHESIS_ID}"]
    assert compiled_action["confidence"] == "low"
    assert compiled_action["evidence_origin"] == "Hypothesis-to-test"
    assert compiled_action["limitations"]
    assert report.downgraded_actions[0].reason


def test_strict_action_validator_still_rejects_empty_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence_refs must not be empty"):
        FinalStrategyAction(
            action_id="strict", priority="P1", action="Test something.",
            reason="No evidence.", evidence_refs=[],
            related_hypothesis_ids=[HYPOTHESIS_ID],
        )
