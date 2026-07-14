from __future__ import annotations

from typing import Any, Iterable

from kaggle_researcher.contracts.errors import (
    ContractIssue,
    CrossArtifactReferenceError,
)
from kaggle_researcher.contracts.experiments import FORBIDDEN_CONTEXT_LABELS, ExperimentPlan
from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.contracts.references import generate_allowed_evidence_refs
from kaggle_researcher.contracts.registries import (
    ExperimentRegistry,
    HypothesisRegistry,
    RiskRegistry,
    SafetyConstraintRegistry,
    ValidationRequirementRegistry,
    classify_namespace,
)
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.review import SkepticalReview
from kaggle_researcher.contracts.validation import ValidationResult


def validate_reasoning_artifact_bundle(
    validation: ValidationResult,
    experiments: ExperimentPlan,
    review: SkepticalReview | None,
    *,
    hypotheses: ResearchHypotheses,
    evidence_ids: Iterable[str],
) -> None:
    hypothesis_registry = HypothesisRegistry.from_contract(hypotheses)
    allowed_evidence = set(evidence_ids)
    issues: list[ContractIssue] = []
    for index, experiment in enumerate(experiments.experiments):
        for value in set(experiment.source_hypothesis_ids) - set(hypothesis_registry.by_id):
            issues.append(ContractIssue(
                f"experiments[{index}].source_hypothesis_ids", value,
                "known hypothesis ID", "unknown reference", "unknown",
            ))
        for value in set(experiment.evidence_ids) - allowed_evidence:
            issues.append(ContractIssue(
                f"experiments[{index}].evidence_ids", value,
                "known evidence ID", "unknown reference", "unknown",
            ))
    try:
        ExperimentRegistry.from_contract(experiments, review)
    except ValueError as exc:
        issues.append(ContractIssue("review.experiment_ids", None, "planned experiment IDs", str(exc)))
    if set(validation.evidence_ids) - allowed_evidence:
        for value in set(validation.evidence_ids) - allowed_evidence:
            issues.append(ContractIssue("validation.evidence_ids", value, "known evidence ID", "unknown reference", "unknown"))
    if issues:
        raise CrossArtifactReferenceError(
            "Reasoning artifact bundle validation failed",
            issues=issues,
            contract="reasoning_artifact_bundle",
        )


def validate_final_synthesis_bundle(
    evidence_pack: Any,
    experiments: ExperimentPlan,
    review: SkepticalReview,
    strategy: FinalStrategyResult,
    *,
    hypotheses: ResearchHypotheses,
    source_ids: Iterable[str] = (),
    optional_stage_failures: Iterable[Any] = (),
) -> None:
    hypothesis_ids = set(HypothesisRegistry.from_contracts(hypotheses, evidence_pack).by_id)
    experiment_registry = ExperimentRegistry.from_contract(experiments, review)
    allowed_evidence = set(generate_allowed_evidence_refs(evidence_pack)) | set(source_ids) | {"final_synthesizer.repaired"}
    risk_registry = RiskRegistry.from_contract(evidence_pack)
    requirement_registry = ValidationRequirementRegistry.from_contract(evidence_pack)
    constraint_registry = SafetyConstraintRegistry.from_contract(evidence_pack)
    namespaces = {
        "hypothesis": hypothesis_ids,
        "experiment": set(experiment_registry.by_id),
        "evidence": allowed_evidence,
        "risk": set(risk_registry.by_id),
        "validation_requirement": set(requirement_registry.by_id),
        "safety_constraint": set(constraint_registry.by_id),
    }
    issues: list[ContractIssue] = []
    actions = list(strategy.actions)
    for index, action in enumerate(actions):
        for value in set(action.hypothesis_ids) - hypothesis_ids:
            issues.append(ContractIssue(f"actions[{index}].hypothesis_ids", value, "known hypothesis ID", "unknown reference", "unknown"))
        for value in action.experiment_ids:
            if value not in experiment_registry.approved_ids:
                namespace = "rejected_experiment" if value in experiment_registry.rejected_ids else "unknown"
                issues.append(ContractIssue(f"actions[{index}].experiment_ids", value, "approved experiment ID", "unapproved or unknown reference", namespace))
        for field, values in (("evidence_refs", action.evidence_refs), ("eda_result_refs", action.eda_result_refs)):
            for value in values:
                if value in FORBIDDEN_CONTEXT_LABELS or value not in allowed_evidence:
                    namespace = "context_label" if value in FORBIDDEN_CONTEXT_LABELS else classify_namespace(str(value), **namespaces) or "unknown"
                    issues.append(ContractIssue(f"actions[{index}].{field}", value, "allowed evidence reference", "invalid evidence reference", namespace))
        for field, values, allowed, expected in (
            ("risk_ids", action.risk_ids, set(risk_registry.by_id), "risk"),
            ("validation_requirement_ids", action.validation_requirement_ids, set(requirement_registry.by_id), "validation requirement"),
            ("safety_constraint_ids", action.safety_constraint_ids, set(constraint_registry.by_id), "safety constraint"),
        ):
            for value in values:
                if value not in allowed:
                    issues.append(ContractIssue(
                        f"actions[{index}].{field}", value, expected,
                        "unknown or cross-namespace reference",
                        classify_namespace(str(value), **namespaces) or "unknown",
                    ))
    payload = evidence_pack.model_dump(mode="json") if hasattr(evidence_pack, "model_dump") else dict(evidence_pack)
    represented_risks = set(strategy.acknowledged_risk_ids) | {
        value for action in actions for value in action.risk_ids
    }
    represented_requirements = set(strategy.selected_validation_requirement_ids) | {
        value for action in actions for value in action.validation_requirement_ids
    }
    represented_constraints = set(strategy.enforced_safety_constraint_ids) | {
        value for action in actions for value in action.safety_constraint_ids
    }
    for value in set(strategy.acknowledged_risk_ids) - set(risk_registry.by_id):
        issues.append(ContractIssue("acknowledged_risk_ids", value, "known risk", "unknown reference", classify_namespace(str(value), **namespaces) or "unknown"))
    for value in set(strategy.selected_validation_requirement_ids) - set(requirement_registry.by_id):
        issues.append(ContractIssue("selected_validation_requirement_ids", value, "known validation requirement", "unknown reference", classify_namespace(str(value), **namespaces) or "unknown"))
    for value in set(strategy.enforced_safety_constraint_ids) - set(constraint_registry.by_id):
        issues.append(ContractIssue("enforced_safety_constraint_ids", value, "known safety constraint", "unknown reference", classify_namespace(str(value), **namespaces) or "unknown"))
    mandatory_requirements = {
        identifier for identifier, requirement in requirement_registry.by_id.items()
        if requirement.mandatory or requirement.status == "required"
    }
    blocking_constraints = {
        identifier for identifier, constraint in constraint_registry.by_id.items()
        if constraint.blocking or constraint.severity == "blocking"
    }
    critical_risks = {
        identifier for identifier, risk in risk_registry.by_id.items()
        if risk.severity == "critical"
    }
    for value in sorted(mandatory_requirements - represented_requirements):
        issues.append(ContractIssue("selected_validation_requirement_ids", value, "represented mandatory validation requirement", "mandatory requirement omitted", "validation_requirement"))
    for value in sorted(blocking_constraints - represented_constraints):
        issues.append(ContractIssue("enforced_safety_constraint_ids", value, "represented blocking safety constraint", "blocking constraint omitted", "safety_constraint"))
    for value in sorted(critical_risks - represented_risks):
        issues.append(ContractIssue("acknowledged_risk_ids", value, "acknowledged critical risk", "critical risk omitted", "risk"))
    missing_limitations = set(payload.get("limitations") or []) - set(strategy.limitations)
    for limitation in missing_limitations:
        issues.append(ContractIssue("limitations", limitation, "preserved EDA limitation", "limitation lost during synthesis"))
    for failure in optional_stage_failures:
        message = getattr(failure, "message", str(failure))
        if message not in strategy.limitations:
            issues.append(ContractIssue("limitations", message, "optional stage failure limitation", "optional failure omitted"))
    if issues:
        raise CrossArtifactReferenceError(
            "Final synthesis bundle validation failed",
            issues=issues,
            contract="final_synthesis_bundle",
        )


__all__ = ["validate_final_synthesis_bundle", "validate_reasoning_artifact_bundle"]
