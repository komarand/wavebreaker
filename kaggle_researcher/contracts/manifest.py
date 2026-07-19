from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from kaggle_researcher.contracts.artifacts import write_json_atomic
from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.errors import (
    ArtifactContractError,
    ContractMigrationError,
    UnsupportedSchemaVersionError,
)
from kaggle_researcher.contracts.ids import StageId
from kaggle_researcher.contracts.migration import MigrationResult


CURRENT_MANIFEST_VERSION = "1.0"
KNOWN_STAGE_IDS = (
    "input_validation",
    "research_scout",
    "eda_engine",
    "reasoning_context",
    "experiment_planner",
    "skeptical_reviewer",
    "final_strategy",
    "final_report",
    "artifact_validation",
)
LEGACY_STAGE_ID_ALIASES = {
    "scout": "research_scout",
    "research": "research_scout",
    "eda": "eda_engine",
    "reasoning": "reasoning_context",
    "strategy": "final_strategy",
    "report": "final_report",
}
LEGACY_STATUS_ALIASES = {
    "success": "completed",
    "done": "completed",
    "error": "failed",
    "cached": "reused",
    "in_progress": "running",
}


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    REUSED = "reused"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"


class ArtifactPointer(ContractModel):
    relative_path: str
    contract_family: str | None = None
    schema_version: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)

    @field_validator("relative_path")
    @classmethod
    def _relative_and_traversal_free(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePath(normalized)
        if not normalized or path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must be a safe run-relative path")
        if len(normalized) >= 2 and normalized[1] == ":":
            raise ValueError("artifact path must not contain a drive prefix")
        return normalized


class StageErrorRecord(ContractModel):
    error_type: str
    message: str = Field(max_length=2000)
    recoverable: bool = False
    suggested_rerun_stage: StageId | None = None
    contract: str | None = None
    invalid_fields: list[str] = Field(default_factory=list)
    invalid_ids: list[str] = Field(default_factory=list)


class StageManifestEntry(ContractModel):
    stage_id: StageId
    status: StageStatus = StageStatus.PENDING
    attempt: int = Field(default=0, ge=0)
    reused: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_sec: float | None = Field(default=None, ge=0)
    inputs: dict[str, ArtifactPointer] = Field(default_factory=dict)
    outputs: dict[str, ArtifactPointer] = Field(default_factory=dict)
    contract_versions: dict[str, str] = Field(default_factory=dict)
    migrations_applied: list[str] = Field(default_factory=list)
    error: StageErrorRecord | None = None
    warnings: list[str] = Field(default_factory=list)
    invalidations: list[str] = Field(default_factory=list)


class ManifestConfigSnapshot(ContractModel):
    competition_url: str | None = None
    competition_description: str = ""
    local_dataset_path: str | None = None
    download_dataset: bool = True
    output_root: str = "runs"
    profile: str = "standard"
    enable_p1_modules: bool = False
    enable_baseline: bool = False
    enable_baseline_ablations: bool = False
    enable_interaction_diagnostics: bool = False
    enable_slice_diagnostics: bool = False
    enable_source_claim_validation: bool = False
    enable_visual_diagnostics: bool = False
    fail_fast: bool = False
    force_rerun_stages: list[StageId] = Field(default_factory=list)
    disable_progress: bool = False


class FinalOutputManifest(ContractModel):
    final_strategy: ArtifactPointer | None = None
    final_report: ArtifactPointer | None = None
    final_synthesis_diagnostics: ArtifactPointer | None = None


class ManifestMigrationRecord(ContractModel):
    source_version: str | None
    target_version: str = CURRENT_MANIFEST_VERSION
    applied_at: datetime
    applied_migrations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunManifest(ContractModel):
    contract_family: Literal["run_manifest"] = "run_manifest"
    schema_version: Literal["1.0"] = CURRENT_MANIFEST_VERSION
    run_id: str
    competition_id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING
    config: ManifestConfigSnapshot = Field(default_factory=ManifestConfigSnapshot)
    stages: dict[StageId, StageManifestEntry]
    final_outputs: FinalOutputManifest = Field(default_factory=FinalOutputManifest)
    migrations: list[ManifestMigrationRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _stage_keys_agree(self) -> "RunManifest":
        for stage_id, entry in self.stages.items():
            if stage_id != entry.stage_id:
                raise ValueError(f"stage key {stage_id!r} disagrees with entry stage_id {entry.stage_id!r}")
        return self


RunManifestMigrationResult = MigrationResult[RunManifest]


def new_run_manifest(*, run_id: str, competition_id: str, config: ManifestConfigSnapshot) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        competition_id=competition_id,
        config=config,
        stages={
            StageId(stage_id): StageManifestEntry(stage_id=StageId(stage_id))
            for stage_id in KNOWN_STAGE_IDS
        },
    )


def migrate_run_manifest(
    payload: Mapping[str, Any], *, run_dir: Path
) -> RunManifestMigrationResult:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("stages"), Mapping):
        raise ContractMigrationError("JSON does not match a supported run-manifest signature", contract="run_manifest")
    source = payload.get("schema_version")
    if source not in {None, CURRENT_MANIFEST_VERSION}:
        raise UnsupportedSchemaVersionError(
            f"Unsupported run_manifest schema version: {source!r}", contract="run_manifest"
        )
    family = payload.get("contract_family")
    if family not in {None, "run_manifest"}:
        raise ContractMigrationError(
            f"Artifact declares contract_family={family!r}; expected 'run_manifest'",
            contract="run_manifest",
        )

    changes: list[str] = []
    warnings: list[str] = []
    stages: dict[StageId, StageManifestEntry] = {}
    for raw_stage_id, raw_entry in payload["stages"].items():
        stage_id = _canonical_stage_id(str(raw_stage_id), changes)
        if stage_id in stages:
            raise ContractMigrationError(f"Duplicate canonical stage ID {stage_id!r}", contract="run_manifest")
        if not isinstance(raw_entry, Mapping):
            raise ContractMigrationError(f"Stage {raw_stage_id!r} must be an object", contract="run_manifest")
        stages[StageId(stage_id)] = _migrate_stage_entry(stage_id, raw_entry, run_dir, changes)

    # Historical manifests occasionally omitted disabled or not-yet-added stages.
    for stage_id in KNOWN_STAGE_IDS:
        typed_id = StageId(stage_id)
        if typed_id not in stages:
            stages[typed_id] = StageManifestEntry(stage_id=typed_id)
            changes.append(f"added missing pending stage {stage_id!r}")

    raw_status = str(payload.get("status") or "pending")
    status = _canonical_status(raw_status, run=True, changes=changes)
    config = _migrate_config(payload.get("config"), changes)
    final_outputs = _migrate_final_outputs(payload.get("final_outputs"), run_dir, changes)
    existing_records = list(payload.get("migrations") or [])
    canonical_shape = (
        source == CURRENT_MANIFEST_VERSION
        and family == "run_manifest"
        and not changes
    )
    if source is None:
        changes.insert(0, "added schema_version=1.0")
    if family is None:
        changes.insert(0, "added contract_family=run_manifest")
    if changes:
        existing_records.append({
            "source_version": source if isinstance(source, str) else None,
            "target_version": CURRENT_MANIFEST_VERSION,
            "applied_at": datetime.now(timezone.utc),
            "applied_migrations": list(changes),
            "warnings": warnings,
        })
    try:
        manifest = RunManifest(
            run_id=str(payload.get("run_id") or run_dir.name),
            competition_id=str(payload.get("competition_id") or ""),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            status=status,
            config=config,
            stages=stages,
            final_outputs=final_outputs,
            migrations=existing_records,
        )
    except Exception as exc:
        raise ContractMigrationError("Migrated run manifest is not canonical", contract="run_manifest") from exc
    return MigrationResult(
        manifest,
        source if isinstance(source, str) else None,
        CURRENT_MANIFEST_VERSION,
        not canonical_shape,
        changes,
        warnings,
    )


def load_run_manifest(path: Path, *, run_dir: Path) -> RunManifestMigrationResult:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Could not read run manifest {path}", contract="run_manifest") from exc
    if not isinstance(payload, Mapping):
        raise ContractMigrationError("Run manifest must contain a JSON object", contract="run_manifest")
    result = migrate_run_manifest(payload, run_dir=run_dir)
    if result.migrated:
        backup = path.with_name("run_manifest.legacy.json")
        if not backup.exists():
            shutil.copy2(path, backup)
        write_run_manifest_atomic(result.value, path)
    return result


def write_run_manifest_atomic(manifest: RunManifest, path: Path) -> None:
    # Validate a detached copy before touching the existing file.
    canonical = RunManifest.model_validate(manifest.model_dump(mode="json"))
    write_json_atomic(path, canonical)


def mark_stage_running(
    manifest: RunManifest, *, stage_id: StageId | str, started_at: datetime
) -> RunManifest:
    identifier = StageId(str(stage_id))
    entry = manifest.stages[identifier]
    updated = entry.model_copy(update={
        "status": StageStatus.RUNNING,
        "attempt": entry.attempt + 1,
        "reused": False,
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": None,
        "error": None,
    })
    return _replace_stage(manifest, identifier, updated)


def mark_stage_completed(
    manifest: RunManifest,
    *,
    stage_id: StageId | str,
    outputs: Mapping[str, ArtifactPointer],
    finished_at: datetime,
    duration_sec: float | None = None,
) -> RunManifest:
    identifier = StageId(str(stage_id))
    entry = manifest.stages[identifier]
    updated = entry.model_copy(update={
        "status": StageStatus.COMPLETED,
        "outputs": dict(outputs),
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "error": None,
    })
    return _replace_stage(manifest, identifier, updated)


def mark_stage_failed(
    manifest: RunManifest,
    *,
    stage_id: StageId | str,
    error: StageErrorRecord,
    finished_at: datetime,
    duration_sec: float | None = None,
    partial: bool = False,
    outputs: dict[str, ArtifactPointer] | None = None,
) -> RunManifest:
    identifier = StageId(str(stage_id))
    entry = manifest.stages[identifier]
    updated = entry.model_copy(update={
        "status": StageStatus.PARTIAL if partial else StageStatus.FAILED,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "error": error,
        "outputs": outputs if outputs is not None else entry.outputs,
    })
    return _replace_stage(manifest, identifier, updated)


def mark_stage_reused(manifest: RunManifest, *, stage_id: StageId | str) -> RunManifest:
    identifier = StageId(str(stage_id))
    entry = manifest.stages[identifier]
    return _replace_stage(
        manifest,
        identifier,
        entry.model_copy(update={"status": StageStatus.REUSED, "reused": True}),
    )


def mark_stage_stale(manifest: RunManifest, *, stage_id: StageId | str, reason: str) -> RunManifest:
    identifier = StageId(str(stage_id))
    entry = manifest.stages[identifier]
    return _replace_stage(
        manifest,
        identifier,
        entry.model_copy(update={
            "status": StageStatus.STALE,
            "reused": False,
            "invalidations": [*entry.invalidations, reason],
        }),
    )


def artifact_pointer(path: Path, *, run_dir: Path, contract_family: str | None = None, schema_version: str | None = None) -> ArtifactPointer:
    resolved = path.resolve()
    root = run_dir.resolve()
    if not resolved.is_relative_to(root):
        raise ArtifactContractError(f"Artifact path is outside run directory: {path}", contract="run_manifest")
    relative = resolved.relative_to(root).as_posix()
    if not resolved.is_file():
        return ArtifactPointer(relative_path=relative, contract_family=contract_family, schema_version=schema_version)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return ArtifactPointer(
        relative_path=relative,
        contract_family=contract_family,
        schema_version=schema_version,
        sha256=digest,
        size_bytes=resolved.stat().st_size,
    )


def validate_artifact_pointer(pointer: ArtifactPointer, *, run_dir: Path) -> tuple[bool, str | None]:
    path = (run_dir / pointer.relative_path).resolve()
    root = run_dir.resolve()
    if not path.is_relative_to(root):
        return False, "artifact path escapes run directory"
    if not path.is_file():
        return False, "artifact is missing"
    if pointer.size_bytes is not None and path.stat().st_size != pointer.size_bytes:
        return False, "artifact size mismatch"
    if pointer.sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != pointer.sha256:
        return False, "artifact hash mismatch"
    return True, None


def _replace_stage(manifest: RunManifest, stage_id: StageId, entry: StageManifestEntry) -> RunManifest:
    stages = dict(manifest.stages)
    stages[stage_id] = entry
    return manifest.model_copy(update={"stages": stages})


def _canonical_stage_id(value: str, changes: list[str]) -> str:
    if value in KNOWN_STAGE_IDS:
        return value
    if value in LEGACY_STAGE_ID_ALIASES:
        mapped = LEGACY_STAGE_ID_ALIASES[value]
        changes.append(f"mapped stage {value!r} to {mapped!r}")
        return mapped
    raise ContractMigrationError(f"Unknown legacy stage ID: {value!r}", contract="run_manifest")


def _canonical_status(value: str, *, run: bool, changes: list[str]) -> RunStatus | StageStatus:
    mapped = LEGACY_STATUS_ALIASES.get(value, value)
    if mapped != value:
        changes.append(f"mapped status {value!r} to {mapped!r}")
    enum_type = RunStatus if run else StageStatus
    try:
        return enum_type(mapped)
    except ValueError as exc:
        raise ContractMigrationError(f"Unknown {'run' if run else 'stage'} status: {value!r}", contract="run_manifest") from exc


def _migrate_stage_entry(stage_id: str, raw: Mapping[str, Any], run_dir: Path, changes: list[str]) -> StageManifestEntry:
    status = _canonical_status(str(raw.get("status") or "pending"), run=False, changes=changes)
    attempt_default = 1 if status in {StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.REUSED} else 0
    attempt = int(raw.get("attempt", attempt_default))
    if "attempt" not in raw and attempt_default:
        changes.append(f"stage {stage_id!r}: defaulted attempt={attempt_default}")
    reused = bool(raw.get("reused", status is StageStatus.REUSED))
    if "reused" not in raw:
        changes.append(f"stage {stage_id!r}: defaulted reused={str(reused).lower()}")
    outputs = _migrate_pointer_map(raw.get("outputs"), run_dir, changes, f"stage {stage_id!r} output")
    inputs = _migrate_pointer_map(raw.get("inputs"), run_dir, changes, f"stage {stage_id!r} input")
    error = _migrate_error(raw.get("error"), stage_id, changes)
    return StageManifestEntry(
        stage_id=StageId(stage_id),
        status=status,
        attempt=attempt,
        reused=reused,
        started_at=raw.get("started_at"),
        finished_at=raw.get("finished_at"),
        duration_sec=raw.get("duration_sec"),
        inputs=inputs,
        outputs=outputs,
        contract_versions=dict(raw.get("contract_versions") or {}),
        migrations_applied=list(raw.get("migrations_applied") or []),
        error=error,
        warnings=list(raw.get("warnings") or []),
        invalidations=list(raw.get("invalidations") or []),
    )


def _migrate_pointer_map(raw: Any, run_dir: Path, changes: list[str], label: str) -> dict[str, ArtifactPointer]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ContractMigrationError(f"{label}s must be an object", contract="run_manifest")
    result: dict[str, ArtifactPointer] = {}
    for name, value in raw.items():
        if isinstance(value, str):
            result[str(name)] = _migrate_path(value, run_dir, changes, label)
        elif isinstance(value, Mapping):
            pointer_payload = dict(value)
            raw_path = pointer_payload.get("relative_path") or pointer_payload.get("path")
            if not isinstance(raw_path, str):
                raise ContractMigrationError(f"{label} {name!r} has no path", contract="run_manifest")
            migrated = _migrate_path(raw_path, run_dir, changes, label)
            result[str(name)] = ArtifactPointer(
                relative_path=migrated.relative_path,
                contract_family=pointer_payload.get("contract_family"),
                schema_version=pointer_payload.get("schema_version"),
                sha256=pointer_payload.get("sha256"),
                size_bytes=pointer_payload.get("size_bytes"),
            )
        else:
            raise ContractMigrationError(f"{label} {name!r} has invalid pointer", contract="run_manifest")
    return result


def _migrate_path(value: str, run_dir: Path, changes: list[str], label: str) -> ArtifactPointer:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        root = run_dir.resolve()
        if not resolved.is_relative_to(root):
            raise ContractMigrationError(f"Unsafe external {label} path: {value!r}", contract="run_manifest")
        relative = resolved.relative_to(root).as_posix()
        changes.append(f"converted absolute {label} path to {relative!r}")
        return ArtifactPointer(relative_path=relative)
    normalized = value.replace("\\", "/")
    try:
        pointer = ArtifactPointer(relative_path=normalized)
    except Exception as exc:
        raise ContractMigrationError(f"Unsafe {label} path: {value!r}", contract="run_manifest") from exc
    resolved = (run_dir / pointer.relative_path).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ContractMigrationError(f"Unsafe {label} path traversal: {value!r}", contract="run_manifest")
    return pointer


def _migrate_error(raw: Any, stage_id: str, changes: list[str]) -> StageErrorRecord | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        changes.append(f"stage {stage_id!r}: converted raw error string")
        return StageErrorRecord(error_type="LegacyStageError", message=raw[:2000], recoverable=False, suggested_rerun_stage=StageId(stage_id))
    if not isinstance(raw, Mapping):
        raise ContractMigrationError(f"Stage {stage_id!r} has invalid error", contract="run_manifest")
    suggested = raw.get("suggested_rerun_stage") or raw.get("stage")
    if suggested not in KNOWN_STAGE_IDS:
        suggested = stage_id
    return StageErrorRecord(
        error_type=str(raw.get("error_type") or raw.get("type") or "StageError"),
        message=str(raw.get("message") or raw.get("error") or "")[:2000],
        recoverable=bool(raw.get("recoverable", False)),
        suggested_rerun_stage=StageId(str(suggested)),
        contract=raw.get("contract"),
        invalid_fields=[str(value) for value in (raw.get("invalid_fields") or [])][:20],
        invalid_ids=[str(value) for value in (raw.get("invalid_ids") or [])][:50],
    )


def _migrate_config(raw: Any, changes: list[str]) -> ManifestConfigSnapshot:
    if raw is None:
        changes.append("defaulted missing config snapshot")
        return ManifestConfigSnapshot()
    if not isinstance(raw, Mapping):
        raise ContractMigrationError("Manifest config must be an object", contract="run_manifest")
    allowed = set(ManifestConfigSnapshot.model_fields)
    ignored = sorted(set(raw) - allowed - {"resume_run_dir"})
    if ignored:
        changes.append(f"removed unsupported legacy config keys: {', '.join(ignored)}")
    payload = {key: raw[key] for key in allowed if key in raw}
    if "force_rerun_stages" in payload:
        payload["force_rerun_stages"] = [StageId(str(value)) for value in payload["force_rerun_stages"]]
    return ManifestConfigSnapshot.model_validate(payload)


def _migrate_final_outputs(raw: Any, run_dir: Path, changes: list[str]) -> FinalOutputManifest:
    if raw is None:
        return FinalOutputManifest()
    if not isinstance(raw, Mapping):
        raise ContractMigrationError("final_outputs must be an object", contract="run_manifest")
    aliases = {
        "strategy": "final_strategy",
        "report": "final_report",
        "diagnostics": "final_synthesis_diagnostics",
        "final_strategy_path": "final_strategy",
        "final_report_path": "final_report",
        "final_synthesis_diagnostics_path": "final_synthesis_diagnostics",
    }
    values: dict[str, ArtifactPointer] = {}
    for raw_name, value in raw.items():
        if value is None:
            continue
        name = aliases.get(str(raw_name), str(raw_name))
        if name not in {
            "final_strategy",
            "final_report",
            "final_synthesis_diagnostics",
        }:
            raise ContractMigrationError(f"Unknown final output key: {raw_name!r}", contract="run_manifest")
        if name != raw_name:
            changes.append(f"mapped final output {raw_name!r} to {name!r}")
        if isinstance(value, str):
            values[name] = _migrate_path(value, run_dir, changes, "final output")
        elif isinstance(value, Mapping):
            raw_path = value.get("relative_path") or value.get("path")
            if not isinstance(raw_path, str):
                raise ContractMigrationError(f"Final output {raw_name!r} has no path", contract="run_manifest")
            base = _migrate_path(raw_path, run_dir, changes, "final output")
            values[name] = ArtifactPointer(
                relative_path=base.relative_path,
                contract_family=value.get("contract_family"),
                schema_version=value.get("schema_version"),
                sha256=value.get("sha256"),
                size_bytes=value.get("size_bytes"),
            )
        else:
            raise ContractMigrationError(f"Final output {raw_name!r} has invalid pointer", contract="run_manifest")
    return FinalOutputManifest(**values)


__all__ = [
    "ArtifactPointer", "CURRENT_MANIFEST_VERSION", "FinalOutputManifest",
    "KNOWN_STAGE_IDS", "LEGACY_STAGE_ID_ALIASES", "LEGACY_STATUS_ALIASES",
    "ManifestConfigSnapshot", "ManifestMigrationRecord", "RunManifest",
    "RunManifestMigrationResult", "RunStatus", "StageErrorRecord",
    "StageManifestEntry", "StageStatus", "artifact_pointer", "load_run_manifest",
    "mark_stage_completed", "mark_stage_failed", "mark_stage_reused",
    "mark_stage_running", "mark_stage_stale", "migrate_run_manifest",
    "new_run_manifest", "validate_artifact_pointer", "write_run_manifest_atomic",
]
