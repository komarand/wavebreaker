from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    ReasoningStageResult,
    ResearchStageResult,
)
from kaggle_researcher.contracts.eda import EdaEvidencePack, EdaTaskPlan
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceConflictPolicy,
    publish_eda_evidence_bundle,
)
from kaggle_researcher.contracts.final_strategy import FinalStrategyAction, FinalStrategyResult
from kaggle_researcher.contracts.registries import build_contract_registries
from kaggle_researcher.contracts.research import ResearchHypotheses, ResearchHypothesis
from kaggle_researcher.contracts.validation import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ValidationResult,
)
from kaggle_researcher.schemas import PlanData


def representative_evidence_pack() -> EdaEvidencePack:
    """A domain-neutral pack covering direct, indexed, and semantic references."""
    return EdaEvidencePack(
        competition_id="contract-demo",
        created_at="2026-07-20T00:00:00Z",
        run_id="contract-run",
        inferred_schema={"primary_id_column": "record_id", "target_column": "label"},
        metric_evidence={"metric_name": "roc_auc", "direction": "maximize"},
        validation_evidence={
            "primary_validation": {"method": "stratified_kfold", "folds": 5}
        },
        baseline_evidence={"metric_value": 0.61, "model": "linear"},
        feature_probe_evidence=[{
            "feature_family": "aggregate_signal",
            "status": "promising",
            "details": {"columns": ["amount", "count"]},
        }],
        eda_risks=[{
            "risk_id": "risk-shift",
            "risk_type": "drift",
            "severity": "medium",
            "status": "suspected",
            "confidence": "medium",
            "title": "Distribution shift",
            "finding": "A feature distribution may move between splits.",
            "impact": "Validation may be optimistic.",
            "evidence_refs": ["feature_probe_evidence.aggregate_signal"],
        }],
    )


def stage_bundle(pack: EdaEvidencePack | None = None):
    pack = pack or representative_evidence_pack()
    hypotheses = ResearchHypotheses(
        competition_id=pack.competition_id,
        hypotheses=[ResearchHypothesis(
            hypothesis_id="hyp-validation",
            category="validation",
            claim="Use an honest validation split.",
            confidence_before_eda="high",
        )],
    )
    research = ResearchStageResult(
        hypotheses,
        EdaTaskPlan(competition_id=pack.competition_id),
        Path("research.json"),
        Path("eda-plan.json"),
        PlanData(task_type="binary", metric="roc_auc", domain="generic_tabular"),
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
            confidence="high", metric_explanation="Ranking metric.",
            needs_calibration=False, rank_averaging_useful=True,
            threshold_search_needed=False, surrogate_loss_suggestion="Binary objective.",
        ),
        ValidationResult(
            confidence="high", recommended_cv="stratified_kfold",
            validation_risk="medium", likely_split="iid", reasoning="Preserve balance.",
            primary_validation={"method": "stratified_kfold"},
        ),
        LeakageRiskResult(confidence="medium", risk_level="medium"),
        LeaderboardAuditResult(
            confidence="medium", shake_up_risk="medium",
            submission_selection_rule="Use validation.", public_lb_trust="low",
        ),
        ExperimentPlan(),
        None,
    )
    registries = build_contract_registries(research=research, eda=eda, reasoning=reasoning)
    return research, eda, reasoning, registries


def strategy_citing(
    refs: list[str], *, eda_result_refs: list[str] | None = None
) -> FinalStrategyResult:
    return FinalStrategyResult(
        competition_id="contract-demo",
        synthesis_status="llm_success",
        llm_output_valid=True,
        repair_attempted=False,
        repair_succeeded=False,
        fallback_used=False,
        synthesis_diagnostics_path=None,
        actions=[FinalStrategyAction(
            action_id="action-evidence-contract",
            priority="P1",
            action="Evaluate the evidence-backed candidate.",
            reason="The published evidence supports measuring it.",
            evidence_refs=refs,
            eda_result_refs=refs if eda_result_refs is None else eda_result_refs,
            related_hypothesis_ids=["hyp-validation"],
        )],
    )
