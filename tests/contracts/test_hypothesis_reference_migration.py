from __future__ import annotations

from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.hypothesis_reference_migration import (
    migrate_final_strategy_hypothesis_references,
    migrate_hypothesis_references,
)
from kaggle_researcher.contracts.reference_catalog import (
    build_final_strategy_reference_catalog,
)


CURRENT_FAILURE_IDS = (
    "eda_hypothesis_002",
    "eda_hypothesis_003",
    "eda_hypothesis_005",
    "eda_hypothesis_006",
    "eda_hypothesis_008",
)


def _pack(*, empty_id: str | None = None) -> EdaEvidencePack:
    backing = {
        "eda_hypothesis_002": "validation_evidence.primary_validation",
        "eda_hypothesis_003": "validation_evidence.diagnostic_validations",
        "eda_hypothesis_005": "metric_evidence.metric_name",
        "eda_hypothesis_006": "drift_evidence.summary",
        "eda_hypothesis_008": "feature_diagnostics.temporal_signal",
    }
    return EdaEvidencePack(
        competition_id="demo",
        created_at="2026-07-14T00:00:00Z",
        run_id="run",
        validation_evidence={
            "primary_validation": {"method": "temporal_cv"},
            "diagnostic_validations": [{"method": "group_kfold"}],
        },
        metric_evidence={"metric_name": "roc_auc"},
        drift_evidence={"summary": "Temporal drift is material."},
        feature_diagnostics={"temporal_signal": "Date-derived features may help."},
        hypothesis_results=[{
            "hypothesis_id": hypothesis_id,
            "category": "validation",
            "status": "confirmed",
            "confidence_after_eda": "high",
            "finding": f"Finding for {hypothesis_id}.",
            "evidence_refs": [] if hypothesis_id == empty_id else [backing[hypothesis_id]],
            "impact_on_strategy": "Test the evidence-backed hypothesis.",
        } for hypothesis_id in CURRENT_FAILURE_IDS],
    )


def _catalog(*, empty_id: str | None = None):
    return build_final_strategy_reference_catalog(_pack(empty_id=empty_id))


def test_hypothesis_moves_and_inherits_backing_evidence() -> None:
    action = {
        "action_id": "temporal-action",
        "action": "Test a temporal feature family.",
        "evidence_refs": ["eda_hypothesis_005"],
        "related_hypothesis_ids": [],
    }

    migrated, diagnostics = migrate_hypothesis_references(action, _catalog())

    assert migrated["evidence_refs"] == ["metric_evidence.metric_name"]
    assert migrated["related_hypothesis_ids"] == ["eda_hypothesis_005"]
    assert diagnostics.moved_hypothesis_refs == ("eda_hypothesis_005",)
    assert diagnostics.inherited_evidence_refs == ("metric_evidence.metric_name",)


def test_multiple_hypotheses_preserve_evidence_and_dedupe_in_order() -> None:
    action = {
        "evidence_refs": [
            "validation_evidence.primary_validation",
            "eda_hypothesis_003",
            "eda_hypothesis_002",
            "validation_evidence.primary_validation",
            "eda_hypothesis_003",
        ],
        "related_hypothesis_ids": ["eda_hypothesis_005", "eda_hypothesis_005"],
    }

    migrated, diagnostics = migrate_hypothesis_references(action, _catalog())

    assert migrated["evidence_refs"] == [
        "validation_evidence.primary_validation",
        "validation_evidence.diagnostic_validations",
    ]
    assert migrated["related_hypothesis_ids"] == [
        "eda_hypothesis_005",
        "eda_hypothesis_003",
        "eda_hypothesis_002",
    ]
    assert diagnostics.moved_hypothesis_refs == (
        "eda_hypothesis_003",
        "eda_hypothesis_002",
    )


def test_migration_is_idempotent() -> None:
    action = {
        "evidence_refs": ["eda_hypothesis_002", "eda_hypothesis_003"],
        "related_hypothesis_ids": [],
    }

    once, first_diagnostics = migrate_hypothesis_references(action, _catalog())
    twice, second_diagnostics = migrate_hypothesis_references(once, _catalog())

    assert twice == once
    assert first_diagnostics.changed
    assert not second_diagnostics.changed


def test_unknown_hypothesis_is_removed_from_evidence_with_diagnostic() -> None:
    action = {
        "evidence_refs": [
            "validation_evidence.primary_validation",
            "eda_hypothesis_999",
        ],
        "related_hypothesis_ids": [],
    }

    migrated, diagnostics = migrate_hypothesis_references(action, _catalog())

    assert migrated["evidence_refs"] == ["validation_evidence.primary_validation"]
    assert "eda_hypothesis_999" not in migrated["related_hypothesis_ids"]
    assert diagnostics.unknown_hypothesis_refs == ("eda_hypothesis_999",)


def test_hypothesis_without_backing_has_no_fabricated_evidence() -> None:
    action = {
        "evidence_refs": ["eda_hypothesis_006"],
        "related_hypothesis_ids": [],
    }

    migrated, diagnostics = migrate_hypothesis_references(
        action,
        _catalog(empty_id="eda_hypothesis_006"),
    )

    assert migrated["evidence_refs"] == []
    assert migrated["related_hypothesis_ids"] == ["eda_hypothesis_006"]
    assert diagnostics.hypotheses_without_backing_evidence == (
        "eda_hypothesis_006",
    )


def test_all_ids_from_current_failure_are_migrated() -> None:
    action = {
        "evidence_refs": list(CURRENT_FAILURE_IDS),
        "related_hypothesis_ids": [],
    }

    migrated, diagnostics = migrate_hypothesis_references(action, _catalog())

    assert migrated["related_hypothesis_ids"] == list(CURRENT_FAILURE_IDS)
    assert not set(CURRENT_FAILURE_IDS) & set(migrated["evidence_refs"])
    assert diagnostics.moved_hypothesis_refs == CURRENT_FAILURE_IDS
    assert len(migrated["evidence_refs"]) == len(CURRENT_FAILURE_IDS)


def test_top_level_and_section_copies_share_one_action_id_migration() -> None:
    payload = {
        "actions": [{
            "action_id": "same-action",
            "evidence_refs": ["eda_hypothesis_002"],
            "related_hypothesis_ids": [],
        }],
        "sections": [{
            "section_id": "experiments_queue",
            "actions": [{
                "action_id": "same-action",
                "evidence_refs": ["eda_hypothesis_003"],
                "related_hypothesis_ids": [],
            }],
        }],
    }

    migrated, diagnostics = migrate_final_strategy_hypothesis_references(
        payload,
        _catalog(),
    )
    top = migrated["actions"][0]
    nested = migrated["sections"][0]["actions"][0]

    assert top["evidence_refs"] == nested["evidence_refs"] == [
        "validation_evidence.primary_validation",
        "validation_evidence.diagnostic_validations",
    ]
    assert top["related_hypothesis_ids"] == nested["related_hypothesis_ids"] == [
        "eda_hypothesis_002",
        "eda_hypothesis_003",
    ]
    assert diagnostics.moved_hypothesis_refs == (
        "eda_hypothesis_002",
        "eda_hypothesis_003",
    )


def test_non_hypothesis_namespaces_are_not_migrated_in_p02() -> None:
    pack_payload = _pack().model_dump(mode="json")
    pack_payload["eda_risks"] = [{
            "risk_id": "risk_leakage_001",
            "risk_type": "leakage",
            "severity": "high",
            "status": "confirmed",
            "confidence": "high",
            "title": "Leakage",
            "finding": "Leakage risk.",
            "impact": "Invalid score.",
            "evidence_refs": ["validation_evidence.primary_validation"],
        }]
    pack = EdaEvidencePack.model_validate(pack_payload)
    catalog = build_final_strategy_reference_catalog(pack)
    action = {
        "evidence_refs": ["risk_leakage_001"],
        "related_hypothesis_ids": [],
    }

    migrated, diagnostics = migrate_hypothesis_references(action, catalog)

    assert migrated["evidence_refs"] == ["risk_leakage_001"]
    assert not diagnostics.changed
