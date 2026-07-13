from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from kaggle_researcher.contracts.eda_task_plan import load_eda_task_plan
from kaggle_researcher.contracts.evidence import (
    ReferenceResolutionError,
    build_evidence_registry,
    resolve_evidence_reference,
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


StageStatus = Literal["pending", "running", "completed", "failed", "partial", "skipped", "reused"]


class StageManifestEntry(BaseModel):
    status: StageStatus
    attempt: int = 0
    outputs: dict[str, str] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class RunManifest(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    competition_id: str
    status: StageStatus
    stages: dict[str, StageManifestEntry]
    final_outputs: dict[str, str] = Field(default_factory=dict)


def validate_final_strategy_references(
    strategy: FinalStrategyResult,
    evidence_pack: EdaEvidencePack,
    *,
    hypothesis_ids: set[str],
    source_ids: set[str] | None = None,
    experiment_ids: set[str] | None = None,
    rejected_experiment_ids: set[str] | None = None,
) -> None:
    source_ids = source_ids or set()
    experiment_ids = experiment_ids or set()
    rejected_experiment_ids = rejected_experiment_ids or set()
    registry = build_evidence_registry(
        evidence_pack,
        source_ids=source_ids,
        hypothesis_ids=hypothesis_ids,
    )
    issues: list[ContractIssue] = []
    actions = [*strategy.actions, *(action for section in strategy.sections for action in section.actions)]
    for index, action in enumerate(actions):
        unknown_hypotheses = sorted(set(action.related_hypothesis_ids) - hypothesis_ids)
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
        unknown_sources = sorted(set(action.source_refs) - source_ids)
        if unknown_sources:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].source_refs", "unknown source reference", tuple(unknown_sources)))
        unknown_experiments = sorted(set(action.experiment_ids) - experiment_ids)
        if unknown_experiments:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].experiment_ids", "unknown experiment reference", tuple(unknown_experiments)))
        restored = sorted(set(action.experiment_ids) & rejected_experiment_ids)
        if restored:
            issues.append(ContractIssue("final_strategy", f"actions[{index}].experiment_ids", "reviewer-rejected experiment restored", tuple(restored)))
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
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            for field_path, relative_path in (
                (f"final_outputs.{name}", value)
                for name, value in manifest.final_outputs.items()
            ):
                candidate = (run_dir / relative_path).resolve()
                if run_dir.resolve() not in (candidate, *candidate.parents):
                    issues.append(ContractIssue("run_manifest", field_path, "path escapes run directory"))
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
        evidence_pack = EdaEvidencePack.model_validate_json(
            (run_dir / "eda" / "eda_evidence_pack.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise ArtifactContractValidationError([
            ContractIssue("eda_evidence_pack", "eda/eda_evidence_pack.json", str(exc))
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
    experiment_ids: set[str] = set()
    if experiment_path.is_file():
        try:
            for item in json.loads(experiment_path.read_text(encoding="utf-8")):
                experiment = ExperimentItem.model_validate(item)
                if not experiment.experiment_id:
                    raise ValueError("Canonical experiment artifacts require experiment_id.")
                if experiment.experiment_id in experiment_ids:
                    raise ValueError(f"Duplicate experiment_id: {experiment.experiment_id!r}.")
                experiment_ids.add(experiment.experiment_id)
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            issues.append(ContractIssue("experiment_plan", "experiment_plan.json", str(exc)))
    rejected_experiment_ids: set[str] = set()
    if review_result is not None:
        reviewer_ids = set(review_result.approved_experiment_ids) | set(review_result.rejected_experiment_ids)
        unknown_reviewer_ids = sorted(reviewer_ids - experiment_ids)
        if unknown_reviewer_ids:
            issues.append(ContractIssue("skeptical_review", "experiment_ids", "unknown experiment reference", tuple(unknown_reviewer_ids)))
        rejected_experiment_ids = set(review_result.rejected_experiment_ids)

    try:
        strategy = FinalStrategyResult.model_validate_json(
            (run_dir / "final" / "final_strategy.json").read_text(encoding="utf-8")
        )
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
                hypothesis_ids={item.hypothesis_id for item in hypotheses.hypotheses},
                source_ids=source_ids,
                experiment_ids=experiment_ids,
                rejected_experiment_ids=rejected_experiment_ids,
            )
        except ArtifactContractValidationError as exc:
            issues.extend(exc.issues)

    report_path = run_dir / "final" / "final_report.md"
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        issues.append(ContractIssue("final_report", "final/final_report.md", "missing or empty"))
    if issues:
        raise FinalArtifactValidationError(issues)
