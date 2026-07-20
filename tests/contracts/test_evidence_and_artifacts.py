from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.evidence import (
    ReferenceResolutionError,
    build_evidence_registry,
    resolve_evidence_reference,
)
from kaggle_researcher.contracts.pipeline import (
    ArtifactContractValidationError,
    validate_full_run_artifacts,
    validate_final_strategy_references,
)
from kaggle_researcher.contracts.research_hypotheses import ResearchHypotheses
from kaggle_researcher.eda.schemas import EdaEvidencePack
from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult


pytestmark = pytest.mark.contract


def _hypotheses() -> ResearchHypotheses:
    return ResearchHypotheses.model_validate({
        "competition_id": "demo",
        "hypotheses": [{
            "hypothesis_id": "val_001",
            "category": "validation",
            "claim": "Use stratified validation.",
            "confidence_before_eda": "medium",
        }],
    })


def _pack() -> EdaEvidencePack:
    return EdaEvidencePack(
        competition_id="demo",
        created_at="2026-07-13T00:00:00Z",
        run_id="run-001",
        validation_evidence={"primary_validation": {"method": "stratified_kfold"}},
        safety_constraints=[{
            "safety_constraint_id": "safe-001", "scope": "validation",
            "rule": "Fit transforms within folds.", "reason": "Prevent leakage.",
        }],
        validation_requirements=[{
            "validation_requirement_id": "val-001",
            "rule": "Use stratified folds.", "reason": "Preserve class balance.",
        }],
    )


def _strategy() -> FinalStrategyResult:
    payload = json.loads(Path("tests/fixtures/reasoning/final_strategy_valid.json").read_text(encoding="utf-8"))
    return FinalStrategyResult.model_validate(payload)


def test_typed_evidence_registry_resolves_exact_nested_paths() -> None:
    registry = build_evidence_registry(_pack(), source_ids=["source-001"], hypothesis_ids=["val_001"])

    nested = resolve_evidence_reference(
        "validation_evidence.primary_validation",
        registry,
        namespaces=("eda_evidence",),
    )
    assert nested.reference.namespace == "eda_evidence"
    assert nested.value["method"] == "stratified_kfold"
    with pytest.raises(ReferenceResolutionError):
        resolve_evidence_reference("validation_policy", registry, namespaces=("eda_evidence",))
    with pytest.raises(ReferenceResolutionError):
        resolve_evidence_reference("val_001", registry, namespaces=("eda_evidence",))

    ambiguous = build_evidence_registry(source_ids=["shared-id"], hypothesis_ids=["shared-id"])
    with pytest.raises(ReferenceResolutionError, match="ambiguous namespace"):
        resolve_evidence_reference("shared-id", ambiguous)


def test_final_strategy_references_are_checked_by_namespace() -> None:
    validate_final_strategy_references(
        _strategy(), _pack(), hypothesis_ids={"val_001"}, source_ids=set()
    )

    bad = _strategy().model_copy(deep=True)
    bad.actions[0].evidence_refs = ["unknown.measurement"]
    bad.actions[0].related_hypothesis_ids = ["unknown-hypothesis"]
    with pytest.raises(ArtifactContractValidationError) as raised:
        validate_final_strategy_references(
            bad, _pack(), hypothesis_ids={"val_001"}, source_ids=set()
        )
    assert set(raised.value.invalid_ids) == {"unknown.measurement", "unknown-hypothesis"}
    assert raised.value.suggested_rerun_stage == "final_strategy"


def test_final_strategy_eda_result_refs_reject_global_source_ids() -> None:
    bad = _strategy().model_copy(deep=True)
    bad.actions[0].evidence_refs = ["source-001"]
    bad.actions[0].eda_result_refs = ["source-001"]

    with pytest.raises(ArtifactContractValidationError) as raised:
        validate_final_strategy_references(
            bad,
            _pack(),
            hypothesis_ids={"val_001"},
            source_ids={"source-001"},
        )

    assert raised.value.invalid_ids == ("source-001",)
    assert raised.value.field_paths == ("actions[0].eda_result_refs",)


def test_real_artifact_validator_checks_bundle_and_final_references(tmp_path: Path) -> None:
    research = tmp_path / "research"
    eda = tmp_path / "eda"
    final = tmp_path / "final"
    research.mkdir(); eda.mkdir(); final.mkdir()
    hypotheses = _hypotheses()
    (research / "research_hypotheses.json").write_text(hypotheses.model_dump_json(indent=2), encoding="utf-8")
    (research / "eda_task_plan.json").write_text(json.dumps({
        "schema_version": "1.0",
        "competition_id": "demo",
        "eda_tasks": [{
            "task_id": "validation-001",
            "module": "validation_analyzer",
            "priority": "P0",
            "related_hypothesis_ids": ["val_001"],
        }],
        "hypothesis_index": {"val_001": ["validation-001"]},
    }), encoding="utf-8")
    (eda / "eda_evidence_pack.json").write_text(_pack().model_dump_json(indent=2), encoding="utf-8")
    (final / "final_strategy.json").write_text(_strategy().model_dump_json(indent=2), encoding="utf-8")
    (final / "final_report.md").write_text("# Final report\n", encoding="utf-8")

    validate_full_run_artifacts(tmp_path)

    broken = _strategy().model_copy(deep=True)
    broken.actions[0].evidence_refs = ["missing.path"]
    (final / "final_strategy.json").write_text(broken.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ArtifactContractValidationError) as raised:
        validate_full_run_artifacts(tmp_path)
    assert "missing.path" in raised.value.invalid_ids
