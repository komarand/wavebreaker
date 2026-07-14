from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.errors import (
    ArtifactContractError,
    ContractIssue,
    ContractValidationError,
)
from kaggle_researcher.contracts.experiments import ExperimentItem, ExperimentPlan
from kaggle_researcher.contracts.migration import (
    EdaTaskPlanMigrationResult,
    HypothesisMigrationResult,
    migrate_eda_task_plan_payload,
    migrate_research_hypotheses_payload,
)
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.review import SkepticalReview
from kaggle_researcher.contracts.validation import ValidationResult
from kaggle_researcher.contracts.validation import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
)
if TYPE_CHECKING:
    from kaggle_researcher.schemas import PlanData, RetrievedDocument


@dataclass(frozen=True)
class ResearchStageResult:
    hypotheses: ResearchHypotheses
    task_plan: EdaTaskPlan
    hypotheses_path: Path
    task_plan_path: Path
    plan_data: PlanData
    retrieved_documents: tuple[RetrievedDocument, ...]
    domain_patterns: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EdaStageResult:
    evidence_pack: BaseModel
    evidence_pack_path: Path
    summary_path: Path


@dataclass(frozen=True)
class ReasoningStageResult:
    metric: MetricResult
    validation: ValidationResult
    leakage: LeakageRiskResult
    leaderboard: LeaderboardAuditResult
    experiments: ExperimentPlan | None = None
    review: SkepticalReview | None = None


@dataclass(frozen=True)
class FinalStageResult:
    strategy: BaseModel
    strategy_path: Path
    report_path: Path | None


def load_research_hypotheses(path: Path) -> tuple[ResearchHypotheses, HypothesisMigrationResult]:
    payload = _read_object(path)
    migration = migrate_research_hypotheses_payload(payload)
    return _validate(ResearchHypotheses, migration.value, "research_hypotheses"), migration


def load_eda_task_plan(
    path: Path, *, hypotheses: ResearchHypotheses, validate_bundle: bool = True
) -> tuple[EdaTaskPlan, EdaTaskPlanMigrationResult]:
    payload = _read_object(path)
    migration = migrate_eda_task_plan_payload(payload)
    plan = _validate(EdaTaskPlan, migration.value, "eda_task_plan")
    if validate_bundle:
        validate_research_artifact_bundle(hypotheses, plan)
    return plan, migration


def validate_research_artifact_bundle(
    hypotheses: ResearchHypotheses, task_plan: EdaTaskPlan
) -> None:
    issues: list[ContractIssue] = []
    if hypotheses.competition_id != task_plan.competition_id:
        issues.append(ContractIssue(
            "competition_id", task_plan.competition_id, hypotheses.competition_id,
            "research_hypotheses and eda_task_plan target different competitions",
        ))
    hypothesis_ids = [item.hypothesis_id for item in hypotheses.hypotheses]
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        issues.append(ContractIssue("hypotheses", hypothesis_ids, "unique hypothesis IDs", "duplicate hypothesis_id"))
    known_hypotheses = set(hypothesis_ids)
    known_tasks = {item.task_id for item in task_plan.eda_tasks}
    for index, task in enumerate(task_plan.eda_tasks):
        for reference in set(task.related_hypothesis_ids) - known_hypotheses:
            issues.append(ContractIssue(
                f"eda_tasks[{index}].related_hypothesis_ids", reference, "known hypothesis ID",
                "unknown hypotheses reference", "unknown",
            ))
    for hypothesis_id, task_ids in task_plan.hypothesis_index.items():
        if hypothesis_id not in known_hypotheses:
            issues.append(ContractIssue(
                f"hypothesis_index.{hypothesis_id}", hypothesis_id, "known hypothesis ID",
                "unknown hypothesis reference", "unknown",
            ))
        for task_id in set(task_ids) - known_tasks:
            issues.append(ContractIssue(
                f"hypothesis_index.{hypothesis_id}", task_id, "known EDA task ID",
                "unknown task reference", "unknown",
            ))
    if issues:
        raise ArtifactContractError(
            "Research artifact bundle validation failed",
            issues=issues,
            contract="research_artifact_bundle",
        )


def load_eda_evidence_pack(path: Path) -> BaseModel:
    from kaggle_researcher.eda.schemas import EdaEvidencePack
    return _validate(EdaEvidencePack, _read_object(path), "eda_evidence_pack")


def load_validation_result(path: Path) -> ValidationResult:
    return _validate(ValidationResult, _read_object(path), "validation_result")


def load_experiment_plan(path: Path) -> ExperimentPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {
            "contract_family": "experiment_plan",
            "schema_version": "1.0",
            "experiments": payload,
        }
    return _validate(ExperimentPlan, payload, "experiment_plan")


def load_skeptical_review(path: Path) -> SkepticalReview:
    return _validate(SkepticalReview, _read_object(path), "skeptical_review")


def load_final_strategy(path: Path) -> BaseModel:
    from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult
    return _validate(FinalStrategyResult, _read_object(path), "final_strategy")


def write_research_hypotheses_atomic(path: Path, value: ResearchHypotheses) -> None:
    _write_contract_atomic(path, _validate(ResearchHypotheses, value.model_dump(mode="json"), "research_hypotheses"))


def write_eda_task_plan_atomic(path: Path, value: EdaTaskPlan) -> None:
    _write_contract_atomic(path, _validate(EdaTaskPlan, value.model_dump(mode="json"), "eda_task_plan"))


def write_eda_evidence_pack(path: Path, value: BaseModel) -> None:
    from kaggle_researcher.eda.schemas import EdaEvidencePack
    _write_contract_atomic(path, _validate(EdaEvidencePack, value.model_dump(mode="json"), "eda_evidence_pack"))


def write_experiment_plan(path: Path, value: ExperimentPlan) -> None:
    _write_contract_atomic(path, _validate(ExperimentPlan, value.model_dump(mode="json"), "experiment_plan"))


def write_final_strategy(path: Path, value: BaseModel) -> None:
    from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult
    _write_contract_atomic(path, _validate(FinalStrategyResult, value.model_dump(mode="json"), "final_strategy"))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(
            f"Could not read contract artifact {Path(path).name}"
        ) from exc
    except ValueError as exc:
        raise ArtifactContractError(
            f"Contract artifact {Path(path).name} contains duplicate JSON keys"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactContractError(
            f"Contract artifact {Path(path).name} must contain a JSON object"
        )
    return value


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _validate(model: type[BaseModel], payload: Any, family: str) -> Any:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        issues = [
            ContractIssue(
                ".".join(str(part) for part in error.get("loc", ())),
                _bounded_error_input(error.get("input")),
                error.get("type", "valid value"),
                error.get("msg", "validation failed"),
            )
            for error in exc.errors(include_url=False)
        ]
        raise ContractValidationError(
            f"Canonical {family} validation failed",
            issues=issues,
            contract=family,
        ) from exc


def _bounded_error_input(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("password", "api_key", "token", "postgresql://", "sk-", "ghp_")):
            return "[REDACTED]"
        return value[:160]
    if isinstance(value, (list, dict, tuple)):
        return f"<{type(value).__name__} omitted>"
    return f"<{type(value).__name__}>"


def _write_contract_atomic(path: Path, value: BaseModel) -> None:
    write_json_atomic(path, value.model_dump(mode="json"))


def write_json_atomic(path: Path, value: Any) -> None:
    """Write any JSON-compatible artifact with the canonical atomic policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(
        value.model_dump(mode="json") if isinstance(value, BaseModel) else value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise ArtifactContractError(f"Could not atomically write contract artifact {path}") from exc


def validate_contract_definitions() -> dict[str, Any]:
    """Offline self-check for the canonical registry and compatibility aliases."""
    from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
    from kaggle_researcher.contracts.manifest import RunManifest
    from kaggle_researcher.contracts.review import SkepticalReview
    from kaggle_researcher.contracts.versions import CURRENT_CONTRACT_VERSIONS
    from kaggle_researcher.eda.schemas import (
        EdaEvidencePack,
        EdaTaskPlan as EdaTaskPlanAlias,
        ResearchHypotheses as ResearchHypothesesAlias,
    )

    models = {
        "research_hypotheses": ResearchHypotheses,
        "eda_task_plan": EdaTaskPlan,
        "eda_evidence_pack": EdaEvidencePack,
        "validation_result": ValidationResult,
        "experiment_plan": ExperimentPlan,
        "skeptical_review": SkepticalReview,
        "final_strategy": FinalStrategyResult,
        "run_manifest": RunManifest,
    }
    for family, model in models.items():
        schema = model.model_json_schema()
        if schema.get("additionalProperties") is not False:
            raise ArtifactContractError(f"Canonical {family} model must forbid extra fields")
        if family not in CURRENT_CONTRACT_VERSIONS:
            raise ArtifactContractError(f"Canonical {family} version is not registered")
    if ResearchHypothesesAlias is not ResearchHypotheses or EdaTaskPlanAlias is not EdaTaskPlan:
        raise ArtifactContractError("Scout and EDA compatibility imports do not share canonical classes")
    return {
        "status": "ok",
        "contracts_checked": sorted(models),
        "versions": dict(CURRENT_CONTRACT_VERSIONS),
    }


__all__ = [
    "EdaStageResult", "FinalStageResult", "ReasoningStageResult", "ResearchStageResult",
    "load_eda_evidence_pack", "load_eda_task_plan", "load_experiment_plan",
    "load_final_strategy", "load_research_hypotheses", "load_skeptical_review",
    "load_validation_result", "validate_contract_definitions", "validate_research_artifact_bundle",
    "write_eda_evidence_pack", "write_eda_task_plan_atomic", "write_experiment_plan",
    "write_final_strategy", "write_research_hypotheses_atomic",
    "write_json_atomic",
]
