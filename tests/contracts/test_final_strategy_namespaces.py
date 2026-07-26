from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    ReasoningStageResult,
    ResearchStageResult,
)
from kaggle_researcher.contracts.bundle_validation import validate_final_synthesis_bundle
from kaggle_researcher.contracts.eda import EdaEvidencePack, EdaTaskPlan
from kaggle_researcher.contracts.errors import (
    AmbiguousReferenceError,
    CrossArtifactReferenceError,
)
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.final_strategy import (
    FinalStrategyAction,
    FinalStrategyResult,
    migrate_legacy_final_strategy_payload,
)
from kaggle_researcher.contracts.manifest import ManifestConfigSnapshot, new_run_manifest
from kaggle_researcher.contracts.registries import build_contract_registries
from kaggle_researcher.contracts.research import ResearchHypotheses, ResearchHypothesis
from kaggle_researcher.contracts.review import SkepticalReview
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceConflictPolicy,
    publish_eda_evidence_bundle,
)
from kaggle_researcher.contracts.validation import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ValidationResult,
)
from kaggle_researcher.orchestration.state import (
    FullRunConfig,
    FullRunState,
    MissingStageDependencyError,
    RuntimeServices,
)
from kaggle_researcher.progress import ProgressConfig
from kaggle_researcher.reasoning.final_synthesizer import render_final_strategy
from kaggle_researcher.schemas import PlanData


pytestmark = pytest.mark.contract


def _stage_bundle(*, shared_constraint_id: str = "safe-1"):
    hypotheses = ResearchHypotheses(
        competition_id="demo",
        hypotheses=[ResearchHypothesis(
            hypothesis_id="hyp-1",
            category="validation",
            claim="Use honest validation.",
            confidence_before_eda="high",
        )],
    )
    pack = EdaEvidencePack(
        competition_id="demo",
        created_at="2026-07-13T00:00:00Z",
        run_id="run",
        validation_evidence={"primary_validation": {"method": "stratified_kfold"}},
        eda_risks=[{
            "risk_id": "risk-1", "risk_type": "leakage", "severity": "critical",
            "status": "confirmed", "confidence": "high", "title": "Leakage",
            "finding": "Leakage is possible.", "impact": "Scores can be invalid.",
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
        validation_requirements=[{
            "validation_requirement_id": "req-1", "rule": "Use fixed folds.",
            "reason": "Comparisons must be paired.", "mandatory": True,
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
        safety_constraints=[{
            "safety_constraint_id": shared_constraint_id, "scope": "leakage",
            "rule": "Fit transforms inside folds.", "reason": "Prevent leakage.",
            "blocking": True,
            "evidence_refs": ["validation_evidence.primary_validation"],
        }],
    )
    experiment_plan = ExperimentPlan(experiments=[{
        "experiment_id": "exp-1", "source_hypothesis_ids": ["hyp-1"],
        "priority": "P0", "experiment": "Run fixed-fold baseline", "why": "Anchor results.",
        "cost": "low", "expected_gain": "diagnostic", "risk": "variance",
        "evidence_ids": ["validation_result"],
    }])
    review = SkepticalReview(
        confidence="high",
        reviewed_experiment_ids=["exp-1"],
        approved_experiment_ids=["exp-1"],
    )
    research = ResearchStageResult(
        hypotheses,
        EdaTaskPlan(competition_id="demo"),
        Path("research_hypotheses.json"),
        Path("eda_task_plan.json"),
        PlanData(task_type="binary", metric="roc_auc", domain="tabular"),
        (),
        (),
    )
    eda = EdaStageResult(pack, Path("eda.json"), Path("eda.md"))
    published = publish_eda_evidence_bundle(
        pack, conflict_policy=EvidenceConflictPolicy.DEGRADED
    )
    eda = replace(
        eda,
        evidence_pack=published.evidence_pack,
        evidence_manifest=published.evidence_manifest,
        published_bundle=published,
    )
    reasoning = ReasoningStageResult(
        MetricResult(
            confidence="high", metric_explanation="Ranking.", needs_calibration=False,
            rank_averaging_useful=True, threshold_search_needed=False,
            surrogate_loss_suggestion="Binary objective.",
        ),
        ValidationResult(
            confidence="high", recommended_cv="StratifiedKFold", validation_risk="medium",
            likely_split="iid", reasoning="Preserve balance.",
            primary_validation={"method": "stratified_kfold"},
        ),
        LeakageRiskResult(confidence="high", risk_level="high"),
        LeaderboardAuditResult(
            confidence="medium", shake_up_risk="medium",
            submission_selection_rule="Use CV.", public_lb_trust="low",
        ),
        experiment_plan,
        review,
    )
    registries = build_contract_registries(
        research=research, eda=eda, reasoning=reasoning
    )
    return research, eda, reasoning, registries


def _strategy(*, safety_id: str = "safe-1") -> FinalStrategyResult:
    return FinalStrategyResult(
        competition_id="demo",
        synthesis_status="llm_success",
        llm_output_valid=True,
        repair_attempted=False,
        repair_succeeded=False,
        fallback_used=False,
        synthesis_diagnostics_path=None,
        actions=[FinalStrategyAction(
            action_id="action-1", priority="P0", action="Run the baseline.",
            reason="It is traceable.",
            evidence_refs=["validation_evidence.primary_validation"],
            related_hypothesis_ids=["hyp-1"], experiment_ids=["exp-1"],
            risk_ids=["risk-1"], validation_requirement_ids=["req-1"],
            safety_constraint_ids=[safety_id],
        )],
    )


def _validate(strategy: FinalStrategyResult, bundle) -> None:
    research, eda, reasoning, _ = bundle
    validate_final_synthesis_bundle(
        eda.evidence_pack,
        reasoning.experiments,
        reasoning.review,
        strategy,
        hypotheses=research.hypotheses,
        evidence_manifest=eda.evidence_manifest,
    )


def test_dedicated_namespace_references_validate_and_render() -> None:
    strategy = _strategy()
    _validate(strategy, _stage_bundle())
    rendered = render_final_strategy(strategy)
    assert "Risks: risk-1" in rendered
    assert "Validation requirements: req-1" in rendered
    assert "Safety constraints: safe-1" in rendered


@pytest.mark.parametrize(("field", "value"), [
    ("evidence_refs", "risk-1"),
    ("evidence_refs", "req-1"),
    ("hypothesis_ids", "safe-1"),
    ("risk_ids", "unknown-risk"),
])
def test_cross_namespace_and_unknown_references_fail(field: str, value: str) -> None:
    strategy = _strategy().model_copy(deep=True)
    setattr(strategy.actions[0], field, [value])
    with pytest.raises(CrossArtifactReferenceError):
        _validate(strategy, _stage_bundle())


@pytest.mark.parametrize("field", [
    "risk_ids", "validation_requirement_ids", "safety_constraint_ids",
])
def test_mandatory_structural_reference_omission_fails(field: str) -> None:
    strategy = _strategy().model_copy(deep=True)
    setattr(strategy.actions[0], field, [])
    with pytest.raises(CrossArtifactReferenceError):
        _validate(strategy, _stage_bundle())


def test_legacy_generic_reference_migration_is_exact_and_ordered() -> None:
    bundle = _stage_bundle()
    registries = bundle[-1]
    payload = _strategy().model_dump(mode="json")
    action = payload["actions"][0]
    action["evidence_refs"] = [
        "validation_evidence.primary_validation", "risk-1", "req-1", "safe-1", "risk-1",
    ]
    action["risk_ids"] = []
    action["validation_requirement_ids"] = []
    action["safety_constraint_ids"] = []

    migration = migrate_legacy_final_strategy_payload(payload, registries=registries)
    migrated = migration.value["actions"][0]
    assert migrated["evidence_refs"] == ["validation_evidence.primary_validation"]
    assert migrated["risk_ids"] == ["risk-1"]
    assert migrated["validation_requirement_ids"] == ["req-1"]
    assert migrated["safety_constraint_ids"] == ["safe-1"]


def test_legacy_generic_reference_migration_rejects_ambiguity() -> None:
    research, eda, reasoning, _ = _stage_bundle(shared_constraint_id="risk-1")
    registries = build_contract_registries(
        research=research, eda=eda, reasoning=reasoning
    )
    payload = _strategy(safety_id="risk-1").model_dump(mode="json")
    payload["actions"][0]["evidence_refs"].append("risk-1")
    with pytest.raises(AmbiguousReferenceError):
        migrate_legacy_final_strategy_payload(payload, registries=registries)


def test_prepared_context_contains_every_exact_namespace_allowlist() -> None:
    research, eda, reasoning, registries = _stage_bundle()
    context = build_final_synthesis_context(
        competition_desc="demo", research=research,
        published_eda_bundle=publish_eda_evidence_bundle(eda.evidence_pack), reasoning=reasoning,
        registries=registries, eda_summary_text="# EDA",
        optional_stage_failures=[SimpleNamespace(message="Optional audit unavailable.")],
    )
    payload = context.reference_prompt_payload()
    assert payload["allowed_risk_ids"] == ["risk-1"]
    assert payload["allowed_validation_requirement_ids"] == ["req-1"]
    assert payload["allowed_safety_constraint_ids"] == ["safe-1"]
    assert payload["optional_stage_failure_messages"] == ["Optional audit unavailable."]


def test_optional_stage_failure_must_be_preserved_structurally() -> None:
    research, eda, reasoning, _ = _stage_bundle()
    strategy = _strategy()
    with pytest.raises(CrossArtifactReferenceError, match="optional failure omitted"):
        validate_final_synthesis_bundle(
            eda.evidence_pack,
            reasoning.experiments,
            reasoning.review,
            strategy,
            hypotheses=research.hypotheses,
            optional_stage_failures=["Optional audit unavailable."],
            evidence_manifest=eda.evidence_manifest,
        )

    strategy.limitations.append("Optional audit unavailable.")
    validate_final_synthesis_bundle(
        eda.evidence_pack,
        reasoning.experiments,
        reasoning.review,
        strategy,
        hypotheses=research.hypotheses,
        optional_stage_failures=["Optional audit unavailable."],
        evidence_manifest=eda.evidence_manifest,
    )


def test_missing_dependency_is_typed_and_stage_results_are_immutable(tmp_path: Path) -> None:
    research, _, _, _ = _stage_bundle()
    services = RuntimeServices(
        reasoning_client=SimpleNamespace(), reasoning_model="test",
        progress=ProgressConfig(enabled=False), logger=SimpleNamespace(),
    )
    state = FullRunState(
        run_dir=tmp_path,
        config=FullRunConfig(competition_id="demo"),
        services=services,
        manifest=new_run_manifest(
            run_id="run", competition_id="demo", config=ManifestConfigSnapshot()
        ),
    )
    with pytest.raises(MissingStageDependencyError) as raised:
        state.require_reasoning("final_strategy")
    assert raised.value.stage_id == "final_strategy"
    assert raised.value.missing_dependency == "reasoning_context"
    with pytest.raises(FrozenInstanceError):
        research.plan_data = PlanData(task_type="x", metric="x", domain="x")
