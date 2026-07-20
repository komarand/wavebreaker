from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.evidence import (
    EvidencePathResolutionError,
    resolve_evidence_ref,
)
from kaggle_researcher.contracts.final_strategy import FinalStrategyAction
from kaggle_researcher.contracts.final_strategy_evidence import (
    bounded_evidence_preview,
    build_action_evidence_bindings,
    validate_action_evidence_consistency,
)


def _pack(*, drift: str = "high", threshold: bool = True) -> dict:
    return {
        "inferred_schema": {"primary_id_column": "PassengerId"},
        "validation_evidence": {
            "primary_validation": {"method": "stratified_kfold"},
        },
        "drift_evidence": {
            "feature_drift_severity": drift,
            "diagnostics": [{"column": "Fare", "psi": 0.31}],
        },
        "metric_evidence": {"requires_threshold": threshold},
        "baseline_evidence": {"status": "completed"},
        "feature_probe_evidence": [
            {"feature_family": "family_size", "status": "high_potential"},
        ],
    }


def _action(text: str, refs: list[str], **updates) -> FinalStrategyAction:
    payload = {
        "action_id": "action_001",
        "priority": "P1",
        "action": text,
        "reason": "EDA supports this action.",
        "evidence_refs": refs,
        "eda_result_refs": list(refs),
        "related_hypothesis_ids": ["hyp_001"],
    }
    payload.update(updates)
    return FinalStrategyAction.model_validate(payload)


def _codes(action: FinalStrategyAction, pack: dict) -> set[str]:
    return {
        issue.code
        for issue in validate_action_evidence_consistency(action, pack)
    }


def test_broad_drift_ref_does_not_certify_high_drift_claim() -> None:
    action = _action("EDA reported high drift severity.", ["drift_evidence"])

    assert "drift_requires_precise_ref" in _codes(action, _pack())


def test_precise_high_drift_ref_passes_when_value_is_high() -> None:
    action = _action(
        "Treat high drift as leaderboard risk.",
        ["drift_evidence.feature_drift_severity"],
    )

    assert not validate_action_evidence_consistency(action, _pack(drift="high"))


def test_high_drift_claim_fails_when_resolved_value_is_low() -> None:
    action = _action(
        "Treat high drift as leaderboard risk.",
        ["drift_evidence.feature_drift_severity"],
    )

    assert "drift_value_contradiction" in _codes(action, _pack(drift="low"))


def test_primary_id_action_requires_primary_id_leaf() -> None:
    action = _action(
        "Exclude PassengerId because it is the primary identifier.",
        ["inferred_schema"],
    )

    assert "primary_id_requires_precise_ref" in _codes(action, _pack())


def test_threshold_tuning_fails_when_metric_flag_is_false() -> None:
    action = _action(
        "Tune the decision threshold inside validation.",
        ["metric_evidence.requires_threshold"],
    )

    assert "threshold_value_contradiction" in _codes(action, _pack(threshold=False))


def test_eda_result_refs_must_be_subset_of_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="must be a subset"):
        _action(
            "Inspect drift diagnostics.",
            ["drift_evidence"],
            eda_result_refs=["drift_evidence.feature_drift_severity"],
        )


def test_resolver_supports_bracket_and_semantic_list_paths() -> None:
    pack = _pack()

    assert resolve_evidence_ref(pack, "drift_evidence.diagnostics[0].psi") == 0.31
    assert resolve_evidence_ref(pack, "feature_probe_evidence.family_size.status") == "high_potential"
    with pytest.raises(EvidencePathResolutionError):
        resolve_evidence_ref(pack, "drift_evidence.missing.value")


def test_large_evidence_preview_is_bounded() -> None:
    preview = bounded_evidence_preview([{"value": "x" * 1000} for _ in range(100)])

    assert preview["type"] == "list"
    assert preview["item_count"] == 100
    assert len(str(preview)) < 1000


def test_generic_inspection_action_may_use_root_ref() -> None:
    action = _action("Inspect the available drift diagnostics before modeling.", ["drift_evidence"])

    assert not validate_action_evidence_consistency(action, _pack(drift="low"))


def test_bindings_include_bounded_resolved_value_and_primary_role() -> None:
    action = _action(
        "Use stratified_kfold as the primary validation policy.",
        ["validation_evidence.primary_validation"],
        validation_strategy="stratified_kfold",
    )

    bindings = build_action_evidence_bindings(action, _pack())

    assert len(bindings) == 1
    assert bindings[0].ref == "validation_evidence.primary_validation"
    assert bindings[0].role == "primary"
    assert bindings[0].resolved_value_preview == {"method": "stratified_kfold"}
