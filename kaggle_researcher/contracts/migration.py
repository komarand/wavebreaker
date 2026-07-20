from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar

from kaggle_researcher.contracts.errors import (
    ArtifactContractError,
    ContractMigrationError,
    UnsupportedSchemaVersionError,
)
from kaggle_researcher.contracts.normalization import HYPOTHESIS_CATEGORY_ALIASES
from kaggle_researcher.contracts.versions import CURRENT_SCHEMA_VERSION


T = TypeVar("T")


@dataclass(frozen=True)
class MigrationResult(Generic[T]):
    value: T
    source_version: str | None
    target_version: str
    migrated: bool
    # Lists are retained for compatibility with existing manifest/report code;
    # callers should treat them as immutable migration metadata.
    applied_migrations: list[str]
    warnings: list[str]

    @property
    def canonical_payload(self) -> T:
        return self.value

    @property
    def source_schema_version(self) -> str | None:
        return self.source_version

    @property
    def target_schema_version(self) -> str:
        return self.target_version


HypothesisMigrationResult = MigrationResult[dict[str, Any]]
EdaTaskPlanMigrationResult = MigrationResult[dict[str, Any]]

LEGACY_CATEGORY_ALIASES = HYPOTHESIS_CATEGORY_ALIASES
LEGACY_STATUS_ALIASES = {
    "needs_eda": "needs_eda", "pending": "needs_eda", "untested": "needs_eda",
    "source_supported": "supported_by_source", "supported_by_source": "supported_by_source",
    "heuristic": "analogous_only", "analogous_only": "analogous_only",
    "not_testable": "not_testable",
}


def _inspect_version(payload: Mapping[str, Any], family: str) -> str | None:
    source = payload.get("schema_version")
    if source not in {None, CURRENT_SCHEMA_VERSION}:
        raise UnsupportedSchemaVersionError(
            f"Unsupported {family} schema version: {source!r}",
            contract=family,
        )
    declared_family = payload.get("contract_family")
    if declared_family not in {None, family}:
        raise ContractMigrationError(
            f"Artifact declares contract_family={declared_family!r}; expected {family!r}",
            contract=family,
        )
    return source if isinstance(source, str) else None


def migrate_research_hypotheses_payload(payload: Mapping[str, Any]) -> HypothesisMigrationResult:
    source = _inspect_version(payload, "research_hypotheses")
    if source == CURRENT_SCHEMA_VERSION:
        _assert_current_fields(
            payload,
            {
                "contract_family", "schema_version", "competition_id", "created_at",
                "hypotheses", "eda_tasks", "structured_findings", "scout_limitations",
                "models_used",
            },
            "research_hypotheses",
        )
        for index, raw in enumerate(payload.get("hypotheses") or []):
            if isinstance(raw, Mapping):
                _assert_current_fields(
                    raw,
                    {
                        "hypothesis_id", "category", "claim", "rationale",
                        "expected_eda_checks", "priority", "confidence_before_eda",
                        "source_refs", "status", "limitations",
                    },
                    f"research_hypotheses.hypotheses[{index}]",
                )
    migrations: list[str] = []
    warnings: list[str] = []
    canonical: dict[str, Any] = {
        "contract_family": "research_hypotheses",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "competition_id": payload.get("competition_id"),
        "created_at": payload.get("created_at"),
        "hypotheses": [],
        "eda_tasks": list(payload.get("eda_tasks") or []),
        "structured_findings": list(payload.get("structured_findings") or []),
        "scout_limitations": list(payload.get("scout_limitations") or []),
        "models_used": dict(payload.get("models_used") or {}),
    }
    if source is None:
        migrations.append("schema_version:missing->1.0")
    if payload.get("contract_family") is None:
        migrations.append("contract_family:missing->research_hypotheses")
    for index, raw in enumerate(payload.get("hypotheses") or []):
        if not isinstance(raw, Mapping):
            canonical["hypotheses"].append(raw)
            continue
        item = dict(raw)
        hypothesis_id, legacy_id = item.get("hypothesis_id"), item.get("id")
        if hypothesis_id is None and legacy_id is not None:
            hypothesis_id = legacy_id
            migrations.append(f"hypotheses[{index}].id->hypothesis_id")
        elif hypothesis_id is not None and legacy_id is not None and hypothesis_id != legacy_id:
            raise ContractMigrationError(
                f"hypotheses[{index}] has conflicting id and hypothesis_id",
                contract="research_hypotheses",
            )
        category = item.get("category")
        if category in LEGACY_CATEGORY_ALIASES:
            normalized = LEGACY_CATEGORY_ALIASES[category]
            migrations.append(f"hypotheses[{index}].category:{category}->{normalized}")
            category = normalized
        confidence = item.get("confidence_before_eda")
        if confidence is None:
            legacy_confidence = item.get("confidence", item.get("source_confidence"))
            confidence = legacy_confidence if legacy_confidence in {"low", "medium", "high"} else "medium"
            migrations.append(f"hypotheses[{index}].confidence->confidence_before_eda")
            if legacy_confidence not in {"low", "medium", "high"}:
                warnings.append(f"hypotheses[{index}].confidence_before_eda defaulted to medium")
        status = item.get("status", "needs_eda")
        if status in LEGACY_STATUS_ALIASES:
            normalized_status = LEGACY_STATUS_ALIASES[status]
            if normalized_status != status:
                migrations.append(f"hypotheses[{index}].status:{status}->{normalized_status}")
            status = normalized_status
        claim = item.get("claim") or item.get("statement") or item.get("hypothesis") or item.get("text")
        if claim != item.get("claim"):
            migrations.append(f"hypotheses[{index}].text_alias->claim")
        canonical["hypotheses"].append({
            "hypothesis_id": hypothesis_id,
            "category": category,
            "claim": claim,
            "rationale": item.get("rationale") or item.get("why_it_matters"),
            "expected_eda_checks": item.get("expected_eda_checks") or item.get("how_to_verify") or [],
            "priority": item.get("priority") or "P1",
            "confidence_before_eda": confidence,
            "source_refs": item.get("source_refs") or item.get("supporting_source_ids") or [],
            "status": status,
            "limitations": item.get("limitations") or [],
        })
    return MigrationResult(
        canonical, source, CURRENT_SCHEMA_VERSION, bool(migrations or warnings),
        migrations, warnings,
    )


def migrate_eda_task_plan_payload(payload: Mapping[str, Any]) -> EdaTaskPlanMigrationResult:
    source = _inspect_version(payload, "eda_task_plan")
    if source == CURRENT_SCHEMA_VERSION:
        _assert_current_fields(
            payload,
            {
                "contract_family", "schema_version", "competition_id", "task_type",
                "metric", "dataset", "eda_tasks", "hypothesis_index",
                "recommended_module_sequence", "recommended_human_checklist",
                "blocking_tasks",
            },
            "eda_task_plan",
        )
        for index, raw in enumerate(payload.get("eda_tasks") or []):
            if isinstance(raw, Mapping):
                _assert_current_fields(
                    raw,
                    {
                        "task_id", "module", "priority", "blocking",
                        "related_hypothesis_ids", "dependencies", "expected_outputs", "params",
                    },
                    f"eda_task_plan.eda_tasks[{index}]",
                )
    migrations: list[str] = []
    tasks: list[Any] = []
    for index, raw in enumerate(payload.get("eda_tasks") or []):
        if not isinstance(raw, Mapping):
            tasks.append(raw)
            continue
        task = dict(raw)
        task_id, legacy_id = task.get("task_id"), task.get("id")
        if task_id is None and legacy_id is not None:
            task_id = legacy_id
            migrations.append(f"eda_tasks[{index}].id->task_id")
        elif task_id is not None and legacy_id is not None and task_id != legacy_id:
            raise ContractMigrationError(
                f"eda_tasks[{index}] has conflicting id and task_id", contract="eda_task_plan"
            )
        hypothesis_ids = task.get("related_hypothesis_ids", task.get("hypothesis_ids", [])) or []
        if not isinstance(hypothesis_ids, list) or not all(isinstance(value, str) for value in hypothesis_ids):
            raise ArtifactContractError(
                f"eda_tasks[{index}].hypothesis_ids must be a list of strings",
                contract="eda_task_plan",
            )
        tasks.append({
            "task_id": task_id,
            "module": task.get("module") or "schema_inferer",
            "priority": task.get("priority") or "P1",
            "blocking": bool(task.get("blocking", False)),
            "related_hypothesis_ids": _unique(hypothesis_ids),
            "dependencies": _unique(list(task.get("dependencies") or [])),
            "expected_outputs": list(task.get("expected_outputs") or []),
            "params": dict(task.get("params") or {}),
        })
    task_ids_by_hypothesis: dict[str, list[str]] = {}
    for task in tasks:
        if isinstance(task, Mapping):
            for hypothesis_id in task.get("related_hypothesis_ids") or []:
                task_ids_by_hypothesis.setdefault(str(hypothesis_id), []).append(task.get("task_id"))
    raw_index = payload.get("hypothesis_index") or {}
    if not isinstance(raw_index, Mapping):
        raise ArtifactContractError("hypothesis_index must be an object", contract="eda_task_plan")
    hypothesis_index: dict[str, list[str]] = {}
    for hypothesis_id, raw_value in raw_index.items():
        values: Any = raw_value
        if isinstance(raw_value, Mapping):
            values = [raw_value]
            migrations.append(f"hypothesis_index.{hypothesis_id}:object->list")
        if not isinstance(values, list):
            raise ArtifactContractError(
                f"hypothesis_index.{hypothesis_id} must be a list or supported legacy object",
                contract="eda_task_plan",
            )
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
                    raise ArtifactContractError(
                        f"hypothesis_index.{hypothesis_id} has invalid task_ids",
                        contract="eda_task_plan",
                    )
                task_ids.extend(entry_ids)
            else:
                raise ArtifactContractError(
                    f"hypothesis_index.{hypothesis_id} contains an invalid entry",
                    contract="eda_task_plan",
                )
        hypothesis_index[str(hypothesis_id)] = _unique(task_ids)
    if source is None:
        migrations.append("schema_version:missing->1.0")
    if payload.get("contract_family") is None:
        migrations.append("contract_family:missing->eda_task_plan")
    canonical = {
        "contract_family": "eda_task_plan",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "competition_id": payload.get("competition_id"),
        "task_type": payload.get("task_type"),
        "metric": dict(payload.get("metric") or {}),
        "dataset": dict(payload.get("dataset") or {}),
        "eda_tasks": tasks,
        "hypothesis_index": hypothesis_index,
        "recommended_module_sequence": list(payload.get("recommended_module_sequence") or []),
        "recommended_human_checklist": list(payload.get("recommended_human_checklist") or []),
        "blocking_tasks": list(payload.get("blocking_tasks") or []),
    }
    return MigrationResult(
        canonical, source, CURRENT_SCHEMA_VERSION, bool(migrations), migrations, []
    )


def _unique(values: list[T]) -> list[T]:
    return list(dict.fromkeys(values))


def migration_is_idempotent(payload: Mapping[str, Any], migrator: Any) -> bool:
    first = migrator(deepcopy(payload)).value
    second = migrator(deepcopy(first)).value
    return first == second


def _assert_current_fields(
    payload: Mapping[str, Any], allowed: set[str], contract: str
) -> None:
    unknown = sorted(str(key) for key in set(payload) - allowed)
    if unknown:
        raise ArtifactContractError(
            f"Current {contract} artifact contains unknown fields: {', '.join(unknown[:8])}",
            contract=contract.split(".", 1)[0],
        )
