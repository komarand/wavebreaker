from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field

from kaggle_researcher.contracts.research_hypotheses import (
    LEGACY_CATEGORY_ALIASES,
    ResearchHypotheses,
    SCHEMA_VERSION,
    UnsupportedSchemaVersionError,
)


class ResearchArtifactContractError(ValueError):
    pass


class EdaTask(BaseModel):
    task_id: str
    module: str
    priority: str
    blocking: bool = False
    related_hypothesis_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class EdaTaskPlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    competition_id: str
    task_type: str | None = None
    metric: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    eda_tasks: list[EdaTask] = Field(default_factory=list)
    hypothesis_index: dict[str, list[str]] = Field(default_factory=dict)
    recommended_module_sequence: list[str] = Field(default_factory=list)
    recommended_human_checklist: list[str] = Field(default_factory=list)
    blocking_tasks: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class EdaTaskPlanMigrationResult:
    canonical_payload: dict[str, Any]
    source_schema_version: str | None
    target_schema_version: str
    migrated: bool
    applied_migrations: list[str]
    warnings: list[str]


def migrate_eda_task_plan_payload(payload: Mapping[str, Any]) -> EdaTaskPlanMigrationResult:
    source_version = payload.get("schema_version")
    if source_version not in {None, SCHEMA_VERSION}:
        raise UnsupportedSchemaVersionError(f"Unsupported EdaTaskPlan schema version: {source_version!r}.")
    migrations: list[str] = []
    tasks: list[dict[str, Any]] = []
    for index, raw_task in enumerate(payload.get("eda_tasks") or []):
        if not isinstance(raw_task, Mapping):
            tasks.append(raw_task)
            continue
        task = dict(raw_task)
        task_id, legacy_id = task.get("task_id"), task.get("id")
        if task_id is None and legacy_id is not None:
            task_id = legacy_id
            migrations.append(f"eda_tasks[{index}].id->task_id")
        elif task_id is not None and legacy_id is not None and task_id != legacy_id:
            raise ResearchArtifactContractError(f"eda_tasks[{index}] has conflicting id={legacy_id!r} and task_id={task_id!r}.")
        hypothesis_ids = task.get("related_hypothesis_ids", task.get("hypothesis_ids", [])) or []
        if not isinstance(hypothesis_ids, list) or not all(isinstance(value, str) for value in hypothesis_ids):
            raise ResearchArtifactContractError(f"eda_tasks[{index}].hypothesis_ids must be a list of strings.")
        tasks.append({"task_id": task_id, "module": task.get("module") or "schema_inferer", "priority": task.get("priority") or "P1", "blocking": bool(task.get("blocking", False)), "related_hypothesis_ids": _unique(hypothesis_ids), "params": dict(task.get("params") or {})})
    task_ids_by_hypothesis: dict[str, list[str]] = {}
    for task in tasks:
        if isinstance(task, Mapping):
            for hypothesis_id in task.get("related_hypothesis_ids") or []:
                task_ids_by_hypothesis.setdefault(hypothesis_id, []).append(task.get("task_id"))
    raw_index = payload.get("hypothesis_index") or {}
    if not isinstance(raw_index, Mapping):
        raise ResearchArtifactContractError("hypothesis_index must be an object.")
    hypothesis_index: dict[str, list[str]] = {}
    for hypothesis_id, value in raw_index.items():
        values = value
        if isinstance(value, Mapping):
            values = [value]
            migrations.append(f"hypothesis_index.{hypothesis_id}:object->list")
        if not isinstance(values, list):
            raise ResearchArtifactContractError(f"hypothesis_index.{hypothesis_id} must be a list or supported legacy object.")
        task_ids: list[str] = []
        for entry in values:
            if isinstance(entry, str):
                task_ids.append(entry)
            elif isinstance(entry, Mapping):
                category = entry.get("category")
                if category in LEGACY_CATEGORY_ALIASES:
                    migrations.append(f"hypothesis_index.{hypothesis_id}.category normalized")
                entry_ids = entry.get("task_ids") or entry.get("related_task_ids") or task_ids_by_hypothesis.get(str(hypothesis_id), [])
                if not isinstance(entry_ids, list) or not all(isinstance(item, str) for item in entry_ids):
                    raise ResearchArtifactContractError(f"hypothesis_index.{hypothesis_id} has invalid task_ids.")
                task_ids.extend(entry_ids)
            else:
                raise ResearchArtifactContractError(f"hypothesis_index.{hypothesis_id} contains an invalid entry.")
        hypothesis_index[str(hypothesis_id)] = _unique(task_ids)
    canonical = {"schema_version": SCHEMA_VERSION, "competition_id": payload.get("competition_id"), "task_type": payload.get("task_type"), "metric": dict(payload.get("metric") or {}), "dataset": dict(payload.get("dataset") or {}), "eda_tasks": tasks, "hypothesis_index": hypothesis_index, "recommended_module_sequence": list(payload.get("recommended_module_sequence") or []), "recommended_human_checklist": list(payload.get("recommended_human_checklist") or []), "blocking_tasks": list(payload.get("blocking_tasks") or [])}
    if source_version is None:
        migrations.append("schema_version:missing->1.0")
    return EdaTaskPlanMigrationResult(canonical, source_version, SCHEMA_VERSION, bool(migrations), migrations, [])


def validate_research_artifact_bundle(hypotheses: ResearchHypotheses, task_plan: EdaTaskPlan) -> None:
    if hypotheses.competition_id != task_plan.competition_id:
        raise ResearchArtifactContractError("research_hypotheses and eda_task_plan target different competitions.")
    hypothesis_ids = [item.hypothesis_id for item in hypotheses.hypotheses]
    task_ids = [item.task_id for item in task_plan.eda_tasks]
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise ResearchArtifactContractError("research_hypotheses contains duplicate hypothesis_id values.")
    if len(task_ids) != len(set(task_ids)):
        raise ResearchArtifactContractError("eda_task_plan contains duplicate task_id values.")
    known_hypotheses, known_tasks = set(hypothesis_ids), set(task_ids)
    for task in task_plan.eda_tasks:
        unknown = set(task.related_hypothesis_ids) - known_hypotheses
        if unknown:
            raise ResearchArtifactContractError(f"Task {task.task_id!r} references unknown hypotheses: {', '.join(sorted(unknown))}.")
    for hypothesis_id, indexed_task_ids in task_plan.hypothesis_index.items():
        if hypothesis_id not in known_hypotheses:
            raise ResearchArtifactContractError(f"hypothesis_index references unknown hypothesis: {hypothesis_id!r}.")
        unknown = set(indexed_task_ids) - known_tasks
        if unknown:
            raise ResearchArtifactContractError(f"hypothesis_index.{hypothesis_id} references unknown tasks: {', '.join(sorted(unknown))}.")


def load_eda_task_plan(path: Path, *, hypotheses: ResearchHypotheses) -> tuple[EdaTaskPlan, EdaTaskPlanMigrationResult]:
    migration = migrate_eda_task_plan_payload(json.loads(Path(path).read_text(encoding="utf-8")))
    task_plan = EdaTaskPlan.model_validate(migration.canonical_payload)
    validate_research_artifact_bundle(hypotheses, task_plan)
    return task_plan, migration


def write_eda_task_plan_atomic(path: Path, task_plan: EdaTaskPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(task_plan.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
