from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from kaggle_researcher.contracts.eda_task_plan import EdaTaskPlan
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.manifest import RunManifest
from kaggle_researcher.contracts.research_hypotheses import ResearchHypotheses
from kaggle_researcher.eda.schemas import EdaEvidencePack
from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ReviewResult,
    ValidationResult,
)


@dataclass(frozen=True)
class ContractDefinition:
    contract_id: str
    producer_stage: str
    consumer_stages: tuple[str, ...]
    model: type[BaseModel] | None
    schema_version: str | None
    reference_fields: tuple[str, ...] = ()
    artifact_name: str | None = None
    nullable_fields: tuple[str, ...] = ()
    collection_fields: tuple[str, ...] = ()
    renderer_consumers: tuple[str, ...] = ()
    migration_support: str = "none"


CONTRACT_DEFINITIONS = (
    ContractDefinition(
        "research_hypotheses", "research_scout", ("eda_engine", "final_strategy"),
        ResearchHypotheses, "1.0", ("hypotheses[].hypothesis_id", "hypotheses[].source_refs"),
        "research_hypotheses.json", ("created_at", "hypotheses[].rationale"),
        ("hypotheses", "eda_tasks", "structured_findings", "scout_limitations"),
        migration_support="legacy unversioned to 1.0",
    ),
    ContractDefinition(
        "eda_task_plan", "research_scout", ("eda_engine",), EdaTaskPlan, "1.0",
        ("eda_tasks[].task_id", "eda_tasks[].related_hypothesis_ids", "hypothesis_index"),
        "eda_task_plan.json", ("task_type",),
        ("eda_tasks", "recommended_module_sequence", "recommended_human_checklist", "blocking_tasks"),
        migration_support="legacy unversioned to 1.0",
    ),
    ContractDefinition(
        "eda_evidence_pack", "eda_engine", ("reasoning_context", "final_strategy", "artifact_validation"),
        EdaEvidencePack, "1.0", ("*.evidence_refs", "*.source_refs", "*.related_hypothesis_ids"),
        "eda_evidence_pack.json", collection_fields=("safety_constraints", "validation_requirements", "testable_hypotheses", "warnings", "limitations"),
        renderer_consumers=("eda_summary", "final_strategy"),
    ),
    ContractDefinition(
        "validation_result", "validation_architect", ("experiment_planner", "leaderboard_auditor", "final_strategy"),
        ValidationResult, "1.0", ("evidence_ids",), "validation_result.json",
        ("secondary_validation",), ("evidence_ids", "failure_modes", "do_not_use", "policy_notes"),
        ("final_report",),
    ),
    ContractDefinition(
        "metric_result", "metric_specialist", ("experiment_planner", "final_strategy"),
        MetricResult, None, ("evidence_ids",), "metric_result.json",
        collection_fields=("evidence_ids",), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "leakage_risk_result", "leakage_risk_analyst", ("experiment_planner", "final_strategy"),
        LeakageRiskResult, None, ("evidence_ids",), "leakage_result.json",
        collection_fields=("evidence_ids", "possible_issues", "recommended_checks"), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "leaderboard_audit_result", "leaderboard_auditor", ("final_strategy",),
        LeaderboardAuditResult, None, ("evidence_ids",), "leaderboard_audit.json",
        collection_fields=("evidence_ids", "warnings"), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "experiment_plan", "experiment_planner", ("skeptical_reviewer", "final_strategy"),
        ExperimentPlan, "1.0", ("experiments[].experiment_id", "experiments[].source_hypothesis_ids", "experiments[].evidence_ids"), "experiment_plan.json",
        collection_fields=("[].evidence_ids",), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "skeptical_review", "skeptical_reviewer", ("final_strategy",), ReviewResult, "1.0",
        ("evidence_ids", "reviewed_experiment_ids", "approved_experiment_ids", "rejected_experiment_ids"), "skeptical_review.json",
        collection_fields=("evidence_ids", "unsupported_claims", "too_generic", "unnecessary_experiments", "approved_experiment_ids", "rejected_experiment_ids"),
        renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "final_strategy", "final_strategy", ("final_report", "artifact_validation"),
        FinalStrategyResult, "1.0",
        (
            "actions[].evidence_refs", "actions[].related_hypothesis_ids",
            "actions[].hypothesis_ids", "actions[].experiment_ids", "actions[].source_refs",
            "actions[].risk_ids", "actions[].validation_requirement_ids",
            "actions[].safety_constraint_ids", "acknowledged_risk_ids",
            "selected_validation_requirement_ids", "enforced_safety_constraint_ids",
        ),
        "final_strategy.json", ("task_type", "recommended_validation"),
        ("sections", "actions", "limitations"), ("final_report",),
    ),
    ContractDefinition(
        "run_manifest", "full_run", ("resume", "run_summary"), RunManifest, "1.0",
        ("stages.*.outputs",), "run_manifest.json",
    ),
    ContractDefinition(
        "full_run_result", "full_run", ("cli",), None, None,
        ("run_dir", "manifest_path", "final_strategy_path", "final_report_path"),
    ),
    ContractDefinition(
        "final_report", "final_report", ("artifact_validation", "human"), None, None,
        artifact_name="final_report.md",
    ),
)


def contract_by_id(contract_id: str) -> ContractDefinition:
    try:
        return next(item for item in CONTRACT_DEFINITIONS if item.contract_id == contract_id)
    except StopIteration as exc:
        raise KeyError(contract_id) from exc
