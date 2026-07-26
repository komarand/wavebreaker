from __future__ import annotations

import pytest

from kaggle_researcher.contracts.composite_reference_resolution import (
    resolve_composite_action_references,
    resolve_final_strategy_composite_references,
)
from kaggle_researcher.contracts.eda import EdaEvidencePack
from tests.contracts.factories import build_final_strategy_reference_catalog


TRACEBACK_REFS = (
    "risk_leakage_001",
    "validation_requirements.validation_requirement_003",
    "safety_constraints.safety_003",
)
EVIDENCE_REF = "validation_evidence.primary_validation"


def _pack(
    *,
    risk_refs: list[str] | None = None,
    requirement_refs: list[str] | None = None,
    constraint_refs: list[str] | None = None,
    constraint_origin: str = "dataset_measurement",
) -> EdaEvidencePack:
    return EdaEvidencePack(
        competition_id="demo",
        created_at="2026-07-14T00:00:00Z",
        run_id="run",
        validation_evidence={"primary_validation": {"method": "group_kfold"}},
        eda_risks=[{
            "risk_id": "risk_leakage_001",
            "risk_type": "leakage",
            "severity": "high",
            "status": "confirmed",
            "confidence": "high",
            "title": "Group leakage",
            "finding": "Groups cross folds.",
            "impact": "Scores become optimistic.",
            "evidence_refs": [EVIDENCE_REF] if risk_refs is None else risk_refs,
        }],
        validation_requirements=[{
            "validation_requirement_id": "validation_requirement_003",
            "rule": "Use grouped validation.",
            "reason": "Groups repeat.",
            "evidence_refs": [EVIDENCE_REF] if requirement_refs is None else requirement_refs,
        }],
        safety_constraints=[{
            "safety_constraint_id": "safety_003",
            "scope": "validation",
            "rule": "Do not split groups.",
            "reason": "Prevent leakage.",
            "evidence_refs": [EVIDENCE_REF] if constraint_refs is None else constraint_refs,
            "evidence_origin": constraint_origin,
        }],
    )


def _catalog(**kwargs):
    return build_final_strategy_reference_catalog(_pack(**kwargs))


@pytest.mark.parametrize("reference", TRACEBACK_REFS)
def test_each_traceback_composite_ref_resolves_to_backing_evidence(reference: str) -> None:
    migrated, diagnostics = resolve_composite_action_references(
        {"evidence_refs": [reference]},
        _catalog(),
    )

    assert migrated["evidence_refs"] == [EVIDENCE_REF]
    assert reference not in migrated["evidence_refs"]
    assert diagnostics.resolved_composite_refs == (reference,)
    assert diagnostics.inherited_backing_evidence_refs == (EVIDENCE_REF,)


def test_existing_direct_evidence_is_preserved_and_deduplicated_in_order() -> None:
    action = {
        "evidence_refs": [
            EVIDENCE_REF,
            "risk_leakage_001",
            "validation_requirements.validation_requirement_003",
            EVIDENCE_REF,
            "safety_constraints.safety_003",
        ],
    }

    migrated, diagnostics = resolve_composite_action_references(action, _catalog())

    assert migrated["evidence_refs"] == [EVIDENCE_REF]
    assert diagnostics.resolved_composite_refs == TRACEBACK_REFS
    assert diagnostics.inherited_backing_evidence_refs == (EVIDENCE_REF,)


def test_broken_backing_ref_is_not_added() -> None:
    broken = "validation_evidence.missing_policy"
    migrated, diagnostics = resolve_composite_action_references(
        {"evidence_refs": ["risk_leakage_001"]},
        _catalog(risk_refs=[broken]),
    )

    assert migrated["evidence_refs"] == []
    assert diagnostics.broken_backing_evidence_refs == (broken,)
    assert diagnostics.composite_refs_without_evidence == ("risk_leakage_001",)


def test_composite_without_evidence_has_diagnostic_and_no_fabricated_ref() -> None:
    reference = "validation_requirements.validation_requirement_003"
    migrated, diagnostics = resolve_composite_action_references(
        {"evidence_refs": [reference]},
        _catalog(requirement_refs=[]),
    )

    assert migrated["evidence_refs"] == []
    assert diagnostics.composite_refs_without_evidence == (reference,)


def test_policy_only_safety_constraint_is_not_factual_evidence() -> None:
    reference = "safety_constraints.safety_003"
    migrated, diagnostics = resolve_composite_action_references(
        {"evidence_refs": [reference]},
        _catalog(constraint_origin="system_policy"),
    )

    assert migrated["evidence_refs"] == []
    assert diagnostics.policy_only_refs == (reference,)
    assert diagnostics.inherited_backing_evidence_refs == ()


def test_resolution_is_idempotent() -> None:
    action = {"evidence_refs": list(TRACEBACK_REFS)}

    once, first = resolve_composite_action_references(action, _catalog())
    twice, second = resolve_composite_action_references(once, _catalog())

    assert twice == once
    assert first.changed
    assert not second.changed


def test_unknown_composite_ref_is_removed_with_diagnostic() -> None:
    migrated, diagnostics = resolve_composite_action_references(
        {"evidence_refs": [EVIDENCE_REF, "risk_unknown_999"]},
        _catalog(),
    )

    assert migrated["evidence_refs"] == [EVIDENCE_REF]
    assert diagnostics.unknown_composite_refs == ("risk_unknown_999",)


def test_top_level_and_section_action_copies_resolve_identically() -> None:
    payload = {
        "actions": [{
            "action_id": "shared",
            "evidence_refs": ["risk_leakage_001"],
        }],
        "sections": [{
            "actions": [{
                "action_id": "shared",
                "evidence_refs": ["safety_constraints.safety_003"],
            }],
        }],
    }

    migrated, _ = resolve_final_strategy_composite_references(payload, _catalog())

    assert migrated["actions"][0]["evidence_refs"] == [EVIDENCE_REF]
    assert migrated["sections"][0]["actions"][0]["evidence_refs"] == [EVIDENCE_REF]


def test_risk_is_not_rewritten_to_risk_register_path() -> None:
    migrated, _ = resolve_composite_action_references(
        {"evidence_refs": ["risk_leakage_001"]},
        _catalog(),
    )

    assert "eda_risk_register.risk_leakage_001" not in migrated["evidence_refs"]
    assert "eda_risks.risk_leakage_001" not in migrated["evidence_refs"]
