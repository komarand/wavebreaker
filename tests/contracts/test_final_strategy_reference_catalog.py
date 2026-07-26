from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.reference_catalog import (
    REFERENCE_NAMESPACES,
)
from tests.contracts.factories import build_final_strategy_reference_catalog


pytestmark = pytest.mark.contract


def _pack(
    *,
    hypothesis_id: str = "eda_hypothesis_005",
    risk_id: str = "risk_leakage_001",
    risk_evidence_refs: list[str] | None = None,
) -> EdaEvidencePack:
    evidence_ref = "validation_evidence.primary_validation"
    return EdaEvidencePack(
        competition_id="demo",
        created_at="2026-07-14T00:00:00Z",
        run_id="run",
        validation_evidence={
            "primary_validation": {"method": "stratified_group_kfold"},
        },
        hypothesis_results=[{
            "hypothesis_id": hypothesis_id,
            "category": "validation",
            "status": "confirmed",
            "confidence_after_eda": "high",
            "finding": "Grouped validation is required.",
            "evidence_refs": [evidence_ref],
            "impact_on_strategy": "Use group-safe folds.",
        }],
        eda_risks=[{
            "risk_id": risk_id,
            "risk_type": "leakage",
            "severity": "critical",
            "status": "confirmed",
            "confidence": "high",
            "title": "Cross-fold leakage",
            "finding": "Groups can leak across folds.",
            "impact": "Validation becomes optimistic.",
            "evidence_refs": risk_evidence_refs if risk_evidence_refs is not None else [evidence_ref],
        }],
        validation_requirements=[{
            "validation_requirement_id": "validation_requirement_003",
            "rule": "Use group-safe validation.",
            "reason": "Groups repeat across rows.",
            "evidence_refs": [evidence_ref],
        }],
        safety_constraints=[{
            "safety_constraint_id": "safety_003",
            "scope": "validation",
            "rule": "Never split a group across folds.",
            "reason": "Prevent leakage.",
            "evidence_refs": [evidence_ref],
        }],
    )


def test_direct_eda_path_is_evidence() -> None:
    catalog = build_final_strategy_reference_catalog(_pack())
    ref = "validation_evidence.primary_validation"

    resolution = catalog.resolve(ref, expected_namespace="evidence")

    assert resolution.is_resolved
    assert resolution.entry is not None
    assert resolution.entry.canonical_ref == ref
    assert catalog.get_namespace(ref) == "evidence"
    assert catalog.is_valid_evidence_ref(ref)


@pytest.mark.parametrize(("ref_id", "expected_namespace", "canonical"), [
    ("eda_hypothesis_005", "hypothesis", "eda_hypothesis_005"),
    ("risk_leakage_001", "risk", "risk_leakage_001"),
    (
        "validation_requirements.validation_requirement_003",
        "validation_requirement",
        "validation_requirement_003",
    ),
    ("safety_constraints.safety_003", "safety_constraint", "safety_003"),
])
def test_object_namespaces_and_canonical_references(
    ref_id: str,
    expected_namespace: str,
    canonical: str,
) -> None:
    catalog = build_final_strategy_reference_catalog(_pack())

    resolution = catalog.resolve(ref_id, expected_namespace=expected_namespace)

    assert resolution.is_resolved
    assert resolution.entry is not None
    assert resolution.entry.namespace == expected_namespace
    assert resolution.entry.canonical_ref == canonical
    assert not catalog.is_valid_evidence_ref(ref_id)


def test_hypothesis_returns_backing_refs_from_hypothesis_result() -> None:
    catalog = build_final_strategy_reference_catalog(_pack())

    assert catalog.get_backing_evidence_refs("eda_hypothesis_005") == (
        "validation_evidence.primary_validation",
    )


def test_risk_returns_backing_refs_from_risk_object() -> None:
    catalog = build_final_strategy_reference_catalog(_pack())

    assert catalog.get_backing_evidence_refs("risk_leakage_001") == (
        "validation_evidence.primary_validation",
    )


def test_unknown_id_returns_controlled_unresolved_result() -> None:
    catalog = build_final_strategy_reference_catalog(_pack())

    resolution = catalog.resolve("does_not_exist")

    assert resolution.status == "unresolved"
    assert resolution.entry is None
    assert resolution.diagnostics[0].code == "unknown_reference"


def test_duplicate_id_across_namespaces_is_diagnosed_and_ambiguous() -> None:
    catalog = build_final_strategy_reference_catalog(_pack(
        hypothesis_id="shared_reference",
        risk_id="shared_reference",
    ))

    resolution = catalog.resolve("shared_reference")

    assert resolution.status == "ambiguous"
    assert catalog.get_namespace("shared_reference") is None
    assert any(
        item.code == "duplicate_reference_id" and item.ref_id == "shared_reference"
        for item in catalog.diagnostics
    )


def test_broken_evidence_path_is_diagnosed_and_not_valid() -> None:
    catalog = build_final_strategy_reference_catalog(_pack(
        risk_evidence_refs=["validation_evidence.missing_policy"],
    ))

    risk = catalog.resolve("risk_leakage_001").entry

    assert risk is not None
    assert not risk.evidence_backed
    assert catalog.get_backing_evidence_refs("risk_leakage_001") == (
        "validation_evidence.missing_policy",
    )
    assert not catalog.is_valid_evidence_ref("validation_evidence.missing_policy")
    assert any(
        item.code == "broken_evidence_path"
        and item.evidence_ref == "validation_evidence.missing_policy"
        for item in catalog.diagnostics
    )


def test_object_without_backing_evidence_is_not_evidence_backed() -> None:
    catalog = build_final_strategy_reference_catalog(_pack(risk_evidence_refs=[]))
    risk = catalog.resolve("risk_leakage_001").entry

    assert risk is not None
    assert not risk.evidence_backed
    assert any(
        item.code == "missing_backing_evidence" and item.ref_id == "risk_leakage_001"
        for item in catalog.diagnostics
    )


def test_unknown_expected_namespace_is_controlled() -> None:
    catalog = build_final_strategy_reference_catalog(_pack())

    resolution = catalog.resolve("risk_leakage_001", expected_namespace="not_real")

    assert resolution.status == "invalid_namespace"
    assert resolution.diagnostics[0].code == "unknown_namespace"
    assert REFERENCE_NAMESPACES == {
        "evidence", "hypothesis", "risk", "validation_requirement",
        "safety_constraint", "source_claim",
    }


def test_existing_validated_source_claim_contract_is_registered() -> None:
    pack = _pack().model_copy(update={
        "source_claim_validation": {
            "validated_claims": [{
                "claim_id": "claim_001",
                "claim_text": "Grouped validation is required.",
                "supporting_eda_refs": ["validation_evidence.primary_validation"],
                "contradicting_eda_refs": [],
                "source_evidence_refs": ["retrieved-1"],
            }],
        },
    })

    catalog = build_final_strategy_reference_catalog(
        pack,
        source_claim_ids=["retrieved-1"],
    )

    assert catalog.get_namespace("claim_001") == "source_claim"
    assert catalog.get_backing_evidence_refs("claim_001") == (
        "retrieved-1",
        "validation_evidence.primary_validation",
    )
    assert catalog.is_valid_evidence_ref("claim_001")


def test_catalog_contract_is_immutable() -> None:
    catalog = build_final_strategy_reference_catalog(_pack())

    with pytest.raises(ValidationError):
        catalog.entries = ()
