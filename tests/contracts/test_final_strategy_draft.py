from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.final_strategy import (
    FinalStrategyAction,
    FinalStrategyResult,
)
from kaggle_researcher.contracts.final_strategy_draft import (
    FinalStrategyActionDraft,
    FinalStrategyDraft,
    FinalStrategyDraftReferenceError,
    FinalStrategySupportRef,
    normalize_legacy_final_strategy_to_draft,
)
from kaggle_researcher.contracts.reference_catalog import (
    ReferenceCatalog,
    ReferenceCatalogEntry,
    ReferenceNamespace,
)


pytestmark = pytest.mark.contract


def _entry(
    ref_id: str,
    namespace: ReferenceNamespace,
    *,
    aliases: tuple[str, ...] = (),
) -> ReferenceCatalogEntry:
    return ReferenceCatalogEntry(
        ref_id=ref_id,
        namespace=namespace,
        canonical_ref=ref_id,
        aliases=aliases,
        backing_evidence_refs=("validation_evidence.primary_validation",),
        evidence_backed=True,
    )


def _catalog() -> ReferenceCatalog:
    return ReferenceCatalog(entries=(
        _entry("validation_evidence.primary_validation", "evidence"),
        _entry("source_001", "source_claim"),
        _entry("risk_leakage_001", "risk"),
        _entry(
            "validation_requirement_003",
            "validation_requirement",
            aliases=("validation_requirements.validation_requirement_003",),
        ),
        _entry(
            "safety_003",
            "safety_constraint",
            aliases=("safety_constraints.safety_003",),
        ),
        *(
            _entry(hypothesis_id, "hypothesis")
            for hypothesis_id in (
                "eda_hypothesis_002",
                "eda_hypothesis_003",
                "eda_hypothesis_005",
                "eda_hypothesis_006",
                "eda_hypothesis_008",
            )
        ),
    ))


def _draft_action(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "priority": "P1",
        "action": "Use the catalog-backed validation policy.",
        "reason": "The measured validation evidence supports it.",
        "support_refs": [{
            "namespace": "evidence",
            "ref_id": "validation_evidence.primary_validation",
        }],
    }
    payload.update(updates)
    return payload


def _draft_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "competition_id": "demo",
        "sections": [{
            "section_id": "executive_summary",
            "title": "Executive Summary",
            "narrative": "Use evidence-backed experiments.",
            "actions": [],
        }],
        "actions": [_draft_action()],
    }
    payload.update(updates)
    return payload


def _legacy_payload(evidence_refs: list[str]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "competition_id": "demo",
        "sections": [{
            "section_id": "executive_summary",
            "title": "Executive Summary",
            "summary": "Use evidence-backed experiments.",
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
        "actions": [{
            "priority": "P1",
            "action": "Run the supported experiment.",
            "reason": "The catalog resolves every supporting object.",
            "evidence_refs": evidence_refs,
            "related_hypothesis_ids": ["eda_hypothesis_005"],
        }],
    }


def test_valid_final_strategy_draft_passes_validation() -> None:
    draft = FinalStrategyDraft.model_validate(_draft_payload())

    assert draft.competition_id == "demo"
    assert draft.actions[0].support_refs[0].namespace == "evidence"


def test_unknown_support_namespace_is_rejected() -> None:
    payload = _draft_payload(actions=[_draft_action(support_refs=[{
        "namespace": "experiment",
        "ref_id": "exp_001",
    }])])

    with pytest.raises(ValidationError):
        FinalStrategyDraft.model_validate(payload)


def test_action_draft_rejects_empty_support_refs() -> None:
    with pytest.raises(ValidationError, match="support_refs"):
        FinalStrategyActionDraft.model_validate(_draft_action(support_refs=[]))


@pytest.mark.parametrize(("namespace", "ref_id"), [
    ("evidence", "validation_evidence.primary_validation"),
    ("hypothesis", "eda_hypothesis_005"),
    ("risk", "risk_leakage_001"),
    ("validation_requirement", "validation_requirement_003"),
    ("safety_constraint", "safety_003"),
    ("source_claim", "source_001"),
])
def test_typed_support_reference_namespaces(
    namespace: ReferenceNamespace,
    ref_id: str,
) -> None:
    support_ref = FinalStrategySupportRef(namespace=namespace, ref_id=ref_id)

    assert support_ref.model_dump(mode="json") == {
        "namespace": namespace,
        "ref_id": ref_id,
    }


def test_legacy_refs_are_resolved_through_reference_catalog() -> None:
    draft = normalize_legacy_final_strategy_to_draft(
        _legacy_payload([
            "validation_evidence.primary_validation",
            "risk_leakage_001",
        ]),
        _catalog(),
    )

    assert [(item.namespace, item.ref_id) for item in draft.actions[0].support_refs] == [
        ("evidence", "validation_evidence.primary_validation"),
        ("risk", "risk_leakage_001"),
        ("hypothesis", "eda_hypothesis_005"),
    ]


def test_legacy_related_hypotheses_become_typed_support_refs() -> None:
    draft = normalize_legacy_final_strategy_to_draft(
        _legacy_payload(["validation_evidence.primary_validation"]),
        _catalog(),
    )

    assert FinalStrategySupportRef(
        namespace="hypothesis",
        ref_id="eda_hypothesis_005",
    ) in draft.actions[0].support_refs
    assert draft.actions[0].related_hypothesis_ids == ["eda_hypothesis_005"]


def test_unknown_legacy_ref_raises_typed_error_early() -> None:
    with pytest.raises(FinalStrategyDraftReferenceError) as raised:
        normalize_legacy_final_strategy_to_draft(
            _legacy_payload(["not_in_catalog"]),
            _catalog(),
        )

    assert raised.value.stage == "final_strategy_draft_normalization"
    assert raised.value.invalid_ids == ("not_in_catalog",)
    assert raised.value.issues[0].resolution_status == "unresolved"


def test_legacy_normalization_is_idempotent() -> None:
    first = normalize_legacy_final_strategy_to_draft(
        _legacy_payload([
            "validation_evidence.primary_validation",
            "validation_evidence.primary_validation",
            "eda_hypothesis_005",
        ]),
        _catalog(),
    )
    second = normalize_legacy_final_strategy_to_draft(first, _catalog())

    assert second == first


def test_action_draft_forbids_raw_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        FinalStrategyActionDraft.model_validate(_draft_action(
            evidence_refs=["validation_evidence.primary_validation"],
        ))


def test_reported_mixed_refs_normalize_to_exact_typed_support() -> None:
    raw_refs = [
        "validation_requirements.validation_requirement_003",
        "safety_constraints.safety_003",
        "risk_leakage_001",
        "eda_hypothesis_002",
        "eda_hypothesis_003",
        "eda_hypothesis_005",
        "eda_hypothesis_006",
        "eda_hypothesis_008",
    ]

    draft = normalize_legacy_final_strategy_to_draft(
        _legacy_payload(raw_refs),
        _catalog(),
    )

    assert [(item.namespace, item.ref_id) for item in draft.actions[0].support_refs] == [
        ("validation_requirement", "validation_requirement_003"),
        ("safety_constraint", "safety_003"),
        ("risk", "risk_leakage_001"),
        ("hypothesis", "eda_hypothesis_002"),
        ("hypothesis", "eda_hypothesis_003"),
        ("hypothesis", "eda_hypothesis_005"),
        ("hypothesis", "eda_hypothesis_006"),
        ("hypothesis", "eda_hypothesis_008"),
    ]


def test_public_final_strategy_models_remain_strict_and_unchanged() -> None:
    action_schema = FinalStrategyAction.model_json_schema()["properties"]
    result_schema = FinalStrategyResult.model_json_schema()["properties"]

    assert "evidence_refs" in action_schema
    assert "support_refs" not in action_schema
    assert "sections" in result_schema
    assert "support_refs" not in result_schema
