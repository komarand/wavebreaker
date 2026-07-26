from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kaggle_researcher.contracts.eda_task_plan import load_eda_task_plan
from kaggle_researcher.contracts.artifacts import (
    load_eda_publication_bundle,
    load_experiment_plan,
    load_final_strategy,
)
from kaggle_researcher.contracts.evidence import (
    ReferenceResolutionError,
    build_evidence_registry_from_manifest,
    resolve_evidence_reference,
)
from kaggle_researcher.contracts.evidence_manifest import EvidenceReferenceManifest
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.registries import ExperimentRegistry
from kaggle_researcher.contracts.manifest import (
    StageManifestEntry,
    StageStatus,
    load_run_manifest,
    validate_artifact_pointer,
)
from kaggle_researcher.contracts.research_hypotheses import load_research_hypotheses
from kaggle_researcher.eda.schemas import EdaEvidencePack
from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult
from kaggle_researcher.schemas import ExperimentItem, RetrievedDocument, ReviewResult, ValidationResult


@dataclass(frozen=True)
class ContractIssue:
    contract: str
    field_path: str
    message: str
    invalid_ids: tuple[str, ...] = ()


class ArtifactContractValidationError(ValueError):
    def __init__(
        self,
        issues: list[ContractIssue],
        *,
        stage: str = "artifact_validation",
        recoverable: bool = True,
        suggested_rerun_stage: str = "final_strategy",
    ) -> None:
        self.stage = stage
        self.contract = "full_research_pipeline"
        self.issues = tuple(issues)
        self.field_paths = tuple(issue.field_path for issue in issues)
        self.invalid_ids = tuple(identifier for issue in issues for identifier in issue.invalid_ids)
        self.recoverable = recoverable
        self.suggested_rerun_stage = suggested_rerun_stage
        summary = "; ".join(f"{item.contract}.{item.field_path}: {item.message}" for item in issues[:8])
        super().__init__(f"Full-pipeline artifact contract validation failed: {summary}")


class CrossArtifactReferenceError(ArtifactContractValidationError):
    pass


class FinalArtifactValidationError(ArtifactContractValidationError):
    pass


def validate_final_strategy_references(
    strategy: FinalStrategyResult,
    evidence_pack: EdaEvidencePack,
    *,
    hypothesis_ids: set[str],
    source_ids: set[str] | None = None,
    experiment_ids: set[str] | None = None,
    approved_experiment_ids: set[str] | None = None,
    rejected_experiment_ids: set[str] | None = None,
    evidence_manifest: EvidenceReferenceManifest,
) -> None:
    source_ids = source_ids or set()
    experiment_ids = experiment_ids or set()
    approved_experiment_ids = approved_experiment_ids if approved_experiment_ids is not None else experiment_ids
    rejected_experiment_ids = rejected_experiment_ids or set()
    registry = build_evidence_registry_from_manifest(
        evidence_manifest,
        evidence_pack,
        source_ids=source_ids,
        hypothesis_ids=hypothesis_ids,
    )
    issues: list[ContractIssue] = []
    actions = list(strategy.actions)
    for index, action in enumerate(actions):
        unknown_hypotheses = sorted((set(action.related_hypothesis_ids) | set(action.hypothesis_ids)) - hypothesis_ids)
        if unknown_hypotheses:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].related_hypothesis_ids", "unknown hypothesis reference", tuple(unknown_hypotheses)))
        for reference_id in action.evidence_refs:
            try:
                resolve_evidence_reference(
                    reference_id,
                    registry,
                    namespaces=("eda_evidence", "source_claim", "synthetic_inference"),
                )
            except ReferenceResolutionError:
                issues.append(ContractIssue("final_strategy", f"actions[{index}].evidence_refs", "unknown evidence reference", (reference_id,)))
        for reference_id in action.eda_result_refs:
            try:
                resolve_evidence_reference(
                    reference_id,
                    registry,
                    namespaces=("eda_evidence",),
                )
            except ReferenceResolutionError:
                issues.append(ContractIssue(
                    "final_strategy",
                    f"actions[{index}].eda_result_refs",
                    "unknown EDA evidence reference",
                    (reference_id,),
                ))
        unknown_sources = sorted(set(action.source_refs) - source_ids)
        if unknown_sources:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].source_refs", "unknown source reference", tuple(unknown_sources)))
        unknown_experiments = sorted(set(action.experiment_ids) - experiment_ids)
        if unknown_experiments:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].experiment_ids", "unknown experiment reference", tuple(unknown_experiments)))
        restored = sorted(set(action.experiment_ids) & rejected_experiment_ids)
        if restored:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].experiment_ids", "reviewer-rejected experiment restored", tuple(restored)))
        unapproved = sorted(
            set(action.experiment_ids) - approved_experiment_ids - rejected_experiment_ids
        )
        if unapproved:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].experiment_ids", "experiment was not reviewer-approved", tuple(unapproved)))
    for index, section in enumerate(strategy.sections):
        for reference_id in section.evidence_refs:
            try:
                resolve_evidence_reference(reference_id, registry, namespaces=("eda_evidence", "source_claim", "synthetic_inference"))
            except ReferenceResolutionError:
                issues.append(ContractIssue("final_strategy", f"sections[{index}].evidence_refs", "unknown evidence reference", (reference_id,)))
    if issues:
        raise CrossArtifactReferenceError(issues)


def validate_full_run_artifacts(run_dir: Path) -> None:
    issues: list[ContractIssue] = []
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = load_run_manifest(manifest_path, run_dir=run_dir).value
            for name in (
                "final_strategy",
                "final_report",
                "final_synthesis_diagnostics",
            ):
                pointer = getattr(manifest.final_outputs, name)
                if pointer is None:
                    continue
                valid, reason = validate_artifact_pointer(pointer, run_dir=run_dir)
                if not valid:
                    issues.append(ContractIssue(
                        "run_manifest", f"final_outputs.{name}", str(reason)
                    ))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            issues.append(ContractIssue("run_manifest", "run_manifest.json", str(exc)))
    research_dir = run_dir / "research"
    try:
        hypotheses, _ = load_research_hypotheses(research_dir / "research_hypotheses.json")
        load_eda_task_plan(research_dir / "eda_task_plan.json", hypotheses=hypotheses)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise ArtifactContractValidationError([
            ContractIssue("research_artifacts", "research", str(exc))
        ], suggested_rerun_stage="research_scout") from exc

    try:
        published_bundle, _ = load_eda_publication_bundle(run_dir / "eda")
        evidence_pack = published_bundle.evidence_pack
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise ArtifactContractValidationError([
            ContractIssue("published_eda_evidence_bundle", "eda", str(exc))
        ], suggested_rerun_stage="eda_engine") from exc

    if evidence_pack.competition_id != hypotheses.competition_id:
        issues.append(ContractIssue("eda_evidence_pack", "competition_id", "does not match ResearchHypotheses"))

    reasoning_dir = run_dir / "reasoning"
    validation_path = reasoning_dir / "validation_result.json"
    if validation_path.is_file():
        try:
            ValidationResult.model_validate_json(validation_path.read_text(encoding="utf-8"))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            issues.append(ContractIssue("ValidationResult", "validation_result.json", str(exc)))

    review_result: ReviewResult | None = None
    review_path = reasoning_dir / "skeptical_review.json"
    if review_path.is_file():
        try:
            review_result = ReviewResult.model_validate_json(review_path.read_text(encoding="utf-8"))
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            issues.append(ContractIssue("ReviewResult", "skeptical_review.json", str(exc)))
    experiment_path = reasoning_dir / "experiment_plan.json"
    experiments: list[ExperimentItem] = []
    experiment_ids: set[str] = set()
    if experiment_path.is_file():
        try:
            for experiment in load_experiment_plan(experiment_path).experiments:
                if not experiment.experiment_id:
                    raise ValueError("Canonical experiment artifacts require experiment_id.")
                if experiment.experiment_id in experiment_ids:
                    raise ValueError(f"Duplicate experiment_id: {experiment.experiment_id!r}.")
                experiment_ids.add(experiment.experiment_id)
                experiments.append(experiment)
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            issues.append(ContractIssue("experiment_plan", "experiment_plan.json", str(exc)))
    rejected_experiment_ids: set[str] = set()
    if review_result is not None:
        reviewer_ids = set(review_result.reviewed_experiment_ids) | set(review_result.approved_experiment_ids) | set(review_result.rejected_experiment_ids)
        unknown_reviewer_ids = sorted(reviewer_ids - experiment_ids)
        if unknown_reviewer_ids:
            issues.append(ContractIssue("skeptical_review", "experiment_ids", "unknown experiment reference", tuple(unknown_reviewer_ids)))
        rejected_experiment_ids = set(review_result.rejected_experiment_ids)

    hypothesis_ids = {
        item.hypothesis_id for item in hypotheses.hypotheses
    } | {
        str(item.hypothesis_id) for item in evidence_pack.testable_hypotheses
    } | {
        str(item.hypothesis_id) for item in evidence_pack.hypothesis_results
    }
    for experiment in experiments:
        unknown_sources = sorted(set(experiment.source_hypothesis_ids) - hypothesis_ids)
        if unknown_sources:
            issues.append(ContractIssue(
                "experiment_plan", f"{experiment.experiment_id}.source_hypothesis_ids",
                "unknown hypothesis reference", tuple(unknown_sources),
            ))
        if experiment.experiment_id in hypothesis_ids:
            issues.append(ContractIssue(
                "experiment_plan", f"{experiment.experiment_id}.experiment_id",
                "experiment identity reuses hypothesis ID", (experiment.experiment_id,),
            ))
    try:
        experiment_registry = ExperimentRegistry.from_contract(
            ExperimentPlan(experiments=experiments), review_result
        )
    except ValueError as exc:
        issues.append(ContractIssue("experiment_registry", "experiment_ids", str(exc)))
        experiment_registry = None

    try:
        strategy = load_final_strategy(run_dir / "final" / "final_strategy.json")
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        issues.append(ContractIssue("final_strategy", "final/final_strategy.json", str(exc)))
        strategy = None

    source_ids: set[str] = set()
    documents_path = research_dir / "retrieved_documents.json"
    if documents_path.is_file():
        try:
            source_ids = {
                RetrievedDocument.model_validate(item).id
                for item in json.loads(documents_path.read_text(encoding="utf-8"))
            }
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            issues.append(ContractIssue("retrieved_documents", "retrieved_documents.json", str(exc)))

    if strategy is not None:
        if strategy.competition_id != hypotheses.competition_id:
            issues.append(ContractIssue("final_strategy", "competition_id", "does not match ResearchHypotheses"))
        try:
            validate_final_strategy_references(
                strategy,
                evidence_pack,
                hypothesis_ids=hypothesis_ids,
                source_ids=source_ids,
                experiment_ids=experiment_ids,
                approved_experiment_ids=(
                    set(experiment_registry.approved_experiment_ids)
                    if experiment_registry is not None else set()
                ),
                rejected_experiment_ids=rejected_experiment_ids,
                evidence_manifest=published_bundle.evidence_manifest,
            )
        except ArtifactContractValidationError as exc:
            issues.extend(exc.issues)

    report_path = run_dir / "final" / "final_report.md"
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        issues.append(ContractIssue("final_report", "final/final_report.md", "missing or empty"))
    if issues:
        raise FinalArtifactValidationError(issues)
