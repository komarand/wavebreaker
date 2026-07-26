from __future__ import annotations

from typing import Any, Iterable

from kaggle_researcher.contracts.errors import (
    ContractIssue,
    CrossArtifactReferenceError,
)
from kaggle_researcher.contracts.experiments import FORBIDDEN_CONTEXT_LABELS, ExperimentPlan
from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.contracts.evidence_manifest import EvidenceReferenceManifest
from kaggle_researcher.contracts.final_strategy_evidence import (
    validate_action_evidence_consistency,
)
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
    evidence_manifest: EvidenceReferenceManifest,
) -> None:
    hypothesis_ids = set(HypothesisRegistry.from_contracts(hypotheses, evidence_pack).by_id)
    experiment_registry = ExperimentRegistry.from_contract(experiments, review)
    allowed_eda_evidence = {
        entry.ref for entry in evidence_manifest.entries
        if entry.available
        and entry.namespace == "eda_evidence"
        and entry.reference_kind in {"direct_path", "semantic_ref"}
    }
    allowed_evidence = allowed_eda_evidence | set(source_ids) | {"final_synthesizer.repaired"}
    allowed_sources = set(source_ids)
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
    source_hypothesis_pairs = {
        (link.source_ref, link.hypothesis_id)
        for link in strategy.source_to_hypothesis_links
    }
    for index, action in enumerate(actions):
        for value in set(action.hypothesis_ids) - hypothesis_ids:
            issues.append(ContractIssue(f"actions[{index}].hypothesis_ids", value, "known hypothesis ID", "unknown reference", "unknown"))
        for value in action.experiment_ids:
            if value not in experiment_registry.approved_ids:
                namespace = "rejected_experiment" if value in experiment_registry.rejected_ids else "unknown"
                issues.append(ContractIssue(f"actions[{index}].experiment_ids", value, "approved experiment ID", "unapproved or unknown reference", namespace))
        for field, values, allowed_field in (
            ("evidence_refs", action.evidence_refs, allowed_evidence),
            ("eda_result_refs", action.eda_result_refs, allowed_eda_evidence),
        ):
            for value in values:
                if value in FORBIDDEN_CONTEXT_LABELS or value not in allowed_field:
                    namespace = "context_label" if value in FORBIDDEN_CONTEXT_LABELS else classify_namespace(str(value), **namespaces) or "unknown"
                    issues.append(ContractIssue(f"actions[{index}].{field}", value, "allowed evidence reference", "invalid evidence reference", namespace))
        for evidence_issue in validate_action_evidence_consistency(
            action,
            evidence_pack,
            allowed_non_eda_refs=allowed_sources,
        ):
            issues.append(ContractIssue(
                f"actions[{index}].evidence_refs",
                evidence_issue.ref,
                "precise EDA path with a claim-consistent resolved value",
                evidence_issue.message,
                "eda_evidence",
            ))
        for value in action.source_refs:
            if value not in allowed_sources:
                issues.append(ContractIssue(
                    f"actions[{index}].source_refs", value,
                    "retrieved source reference", "unknown source reference", "unknown",
                ))
            elif not any(
                (value, hypothesis_id) in source_hypothesis_pairs
                for hypothesis_id in action.hypothesis_ids
            ):
                issues.append(ContractIssue(
                    f"actions[{index}].source_refs", value,
                    "source linked to an action hypothesis",
                    "unrelated source attribution",
                    "source_claim",
                ))
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
    for index, link in enumerate(strategy.source_to_hypothesis_links):
        if link.source_ref not in allowed_sources:
            issues.append(ContractIssue(
                f"source_to_hypothesis_links[{index}].source_ref",
                link.source_ref,
                "retrieved source reference",
                "unknown source reference",
                "unknown",
            ))
        if link.hypothesis_id not in hypothesis_ids:
            issues.append(ContractIssue(
                f"source_to_hypothesis_links[{index}].hypothesis_id",
                link.hypothesis_id,
                "known hypothesis ID",
                "unknown or removed hypothesis",
                "unknown",
            ))
    for index, link in enumerate(strategy.hypothesis_to_eda_links):
        if link.hypothesis_id not in hypothesis_ids:
            issues.append(ContractIssue(
                f"hypothesis_to_eda_links[{index}].hypothesis_id",
                link.hypothesis_id,
                "known hypothesis ID",
                "unknown or removed hypothesis",
                "unknown",
            ))
        if link.eda_result_ref not in allowed_eda_evidence:
            issues.append(ContractIssue(
                f"hypothesis_to_eda_links[{index}].eda_result_ref",
                link.eda_result_ref,
                "allowed EDA evidence reference",
                "invalid EDA result reference",
                "unknown",
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
