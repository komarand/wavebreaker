from __future__ import annotations

import pytest

from kaggle_researcher.contracts.evidence import (
    AmbiguousEvidencePathError,
    EvidencePathResolutionError,
    build_evidence_registry,
    generate_allowed_evidence_refs,
    resolve_evidence_path,
)
from kaggle_researcher.contracts.experiments import (
    CrossNamespaceReferenceError,
    build_experiment_registry,
    reference_registries,
    validate_final_strategy_namespaces,
)
from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyResult,
    _derive_eda_result_refs,
)
from kaggle_researcher.schemas import ExperimentItem, ReviewResult


pytestmark = pytest.mark.contract


def _evidence_pack() -> dict[str, object]:
    return {
        "baseline_ablation_evidence": {
            "best_ablation": {"ablation_id": "abl_all"},
            "ablations": [
                {"ablation_id": "abl_all", "metric_value": 0.74},
            ],
            "feature_block_findings": [
                {"feature_block": "low_cardinality_categorical", "status": "helped"},
                {"feature_block": "high_cardinality_categorical", "status": "neutral"},
                {"feature_block": "text_code_simple", "status": "helped"},
            ],
        },
        "testable_hypotheses": [
            {"hypothesis_id": "eda_hypothesis_001", "hypothesis": "Test encoding."},
        ],
    }


@pytest.mark.parametrize(("reference", "kind"), [
    ("baseline_ablation_evidence", "top_level"),
    ("baseline_ablation_evidence.best_ablation", "dict_path"),
    ("baseline_ablation_evidence.feature_block_findings.low_cardinality_categorical", "semantic_collection_item"),
    ("baseline_ablation_evidence.feature_block_findings.high_cardinality_categorical", "semantic_collection_item"),
    ("baseline_ablation_evidence.feature_block_findings.text_code_simple", "semantic_collection_item"),
])
def test_canonical_evidence_path_resolution(reference: str, kind: str) -> None:
    resolved = resolve_evidence_path(reference, _evidence_pack())

    assert resolved.reference == reference
    assert resolved.root_id == "baseline_ablation_evidence"
    assert resolved.resolution_kind == kind


def test_unknown_semantic_identity_is_not_fuzzily_replaced() -> None:
    with pytest.raises(EvidencePathResolutionError) as raised:
        resolve_evidence_path(
            "baseline_ablation_evidence.feature_block_findings.low_cardinality",
            _evidence_pack(),
        )

    assert raised.value.reason == "unknown"


def test_duplicate_semantic_identity_is_ambiguous() -> None:
    payload = _evidence_pack()
    findings = payload["baseline_ablation_evidence"]["feature_block_findings"]
    findings.append({"feature_block": "text_code_simple", "status": "neutral"})

    with pytest.raises(AmbiguousEvidencePathError) as raised:
        resolve_evidence_path(
            "baseline_ablation_evidence.feature_block_findings.text_code_simple",
            payload,
        )
    assert raised.value.match_count == 2
    with pytest.raises(AmbiguousEvidencePathError):
        build_evidence_registry(payload)


def test_allowed_refs_are_generated_from_current_collection_values() -> None:
    refs = generate_allowed_evidence_refs(_evidence_pack())

    assert "baseline_ablation_evidence.best_ablation" in refs
    assert "baseline_ablation_evidence.ablations.abl_all" in refs
    assert "baseline_ablation_evidence.feature_block_findings.text_code_simple" in refs
    assert "testable_hypotheses.eda_hypothesis_001" in refs
    assert "approved_experiments" not in refs


def test_context_and_cross_namespace_values_are_not_evidence() -> None:
    experiment = ExperimentItem(
        experiment_id="exp_001", source_hypothesis_ids=["eda_hypothesis_001"],
        priority="P1", experiment="Test encoding", why="Test the hypothesis.",
        cost="low", expected_gain="diagnostic", risk="variance",
    )
    review = ReviewResult(
        confidence="medium", reviewed_experiment_ids=["exp_001"],
        approved_experiment_ids=["exp_001"],
    )
    registry = build_experiment_registry([experiment], review)
    strategy = FinalStrategyResult.model_validate({
        "competition_id": "demo",
        "actions": [{
            "priority": "P1", "action": "Run approved test.",
            "reason": "It has review approval.",
            "evidence_refs": ["approved_experiments", "exp_001", "eda_hypothesis_001"],
            "related_hypothesis_ids": ["eda_hypothesis_001"],
            "experiment_ids": ["exp_001"],
        }],
    })
    issues = validate_final_strategy_namespaces(
        strategy,
        reference_registries(
            registry,
            hypothesis_ids={"eda_hypothesis_001"},
            evidence_ids={"baseline_ablation_evidence"},
            eda_evidence_ids={"baseline_ablation_evidence"},
        ),
    )

    by_value = {issue.invalid_value: issue for issue in issues}
    assert by_value["approved_experiments"].reason == "context_label_not_reference"
    assert by_value["approved_experiments"].actual_namespace == "context_label"
    assert by_value["exp_001"].actual_namespace == "approved_experiment"
    assert by_value["eda_hypothesis_001"].actual_namespace == "hypothesis"
    error = CrossNamespaceReferenceError([by_value["approved_experiments"]])
    assert "context collection name, not an evidence reference" in str(error)
    assert "concrete experiment_ids" in str(error)


def test_eda_result_refs_are_derived_only_from_resolvable_eda_refs() -> None:
    strategy = FinalStrategyResult.model_validate({
        "competition_id": "demo",
        "actions": [{
            "priority": "P1", "action": "Compare the ablation.",
            "reason": "The source and EDA finding agree.",
            "evidence_refs": ["source-001", "baseline_ablation_evidence.best_ablation"],
            "related_hypothesis_ids": ["eda_hypothesis_001"],
        }],
    })

    _derive_eda_result_refs(strategy, {"baseline_ablation_evidence.best_ablation"})

    assert strategy.actions[0].eda_result_refs == [
        "baseline_ablation_evidence.best_ablation"
    ]
