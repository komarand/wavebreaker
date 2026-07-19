from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    ReasoningStageResult,
    ResearchStageResult,
)
from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.registries import build_contract_registries
from kaggle_researcher.contracts.review import SkepticalReview
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.contracts.validation import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ValidationResult,
)
from kaggle_researcher.reasoning.final_synthesizer import synthesize_final_strategy


async def synthesize_for_test(
    *,
    competition_desc: str,
    plan_data: Any,
    retrieved_documents: list[Any],
    domain_patterns: list[dict[str, Any]],
    research_hypotheses: Any,
    eda_evidence_pack: Any,
    reasoning_outputs: dict[str, Any],
    client: Any,
    model: str,
    eda_summary_text: str | None = None,
    diagnostics_dir: Path | None = None,
):
    primary = (eda_evidence_pack.validation_evidence.get("primary_validation") or {}).get("method") or "stratified_kfold"
    metric = MetricResult(
        confidence="medium",
        metric_explanation="Ranking metric.",
        needs_calibration=False,
        rank_averaging_useful=True,
        threshold_search_needed=False,
        surrogate_loss_suggestion="Use a compatible objective.",
    )
    validation = ValidationResult(
        confidence="medium",
        recommended_cv=str(primary),
        validation_risk="medium",
        likely_split="iid",
        reasoning="Use the EDA-selected validation policy.",
        primary_validation={"method": str(primary)},
    )
    leakage = LeakageRiskResult(
        confidence="medium", risk_level="medium", possible_issues=[], recommended_checks=[]
    )
    leaderboard = LeaderboardAuditResult(
        confidence="medium",
        shake_up_risk="medium",
        submission_selection_rule="Use validation.",
        public_lb_trust="low",
        warnings=[],
    )
    experiments = ExperimentPlan(experiments=reasoning_outputs.get("experiments") or [])
    raw_review = reasoning_outputs.get("review")
    review = SkepticalReview.model_validate(raw_review) if raw_review else None
    research = ResearchStageResult(
        research_hypotheses,
        EdaTaskPlan(competition_id=research_hypotheses.competition_id),
        Path("research_hypotheses.json"),
        Path("eda_task_plan.json"),
        plan_data,
        tuple(retrieved_documents),
        tuple(domain_patterns),
    )
    eda = EdaStageResult(
        eda_evidence_pack, Path("eda_evidence_pack.json"), Path("eda_summary.md")
    )
    reasoning = ReasoningStageResult(
        metric, validation, leakage, leaderboard, experiments, review
    )
    registries = build_contract_registries(research=research, eda=eda, reasoning=reasoning)
    context = build_final_synthesis_context(
        competition_desc=competition_desc,
        research=research,
        eda=eda,
        reasoning=reasoning,
        registries=registries,
        eda_summary_text=eda_summary_text,
    )
    return await synthesize_final_strategy(
        context=context,
        registries=registries,
        client=client,
        model=model,
        diagnostics_dir=diagnostics_dir,
    )
