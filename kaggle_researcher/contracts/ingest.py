from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import inspect
import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from kaggle_researcher.contracts.errors import (
    BoundaryRepairError,
    ContractIssue,
    ContractMigrationError,
    ContractValidationError,
    InternalContractValidationError,
)
from kaggle_researcher.contracts.hashing import sha256_contract
from kaggle_researcher.contracts.migration import (
    CONTRACT_MIGRATIONS,
    UNVERSIONED_SCHEMA_VERSION,
    ContractMigrationGraph,
)
from kaggle_researcher.contracts.normalization import normalize_contract_payload
from kaggle_researcher.contracts.registry import (
    CONTRACT_REGISTRY,
    ContractHeader,
    ContractRegistry,
)


SourceKind = Literal["internal_deterministic", "external_artifact", "llm_generated"]


class MigrationPolicy(str, Enum):
    FORBID = "forbid"
    ALLOW = "allow"
    REQUIRE_CURRENT = "require_current"


class RepairPolicy(str, Enum):
    FORBID = "forbid"
    NORMALIZE = "normalize"
    ONE_BOUNDED_REPAIR = "one_bounded_repair"


class ContractRepairProvider(Protocol):
    def repair(self, request: dict[str, Any]) -> Mapping[str, Any]: ...


RepairProvider = ContractRepairProvider | Callable[[dict[str, Any]], Mapping[str, Any]]
Normalizer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class ContractIngestResult:
    contract: BaseModel
    original_family: str
    original_version: str
    final_family: str
    final_version: str
    migrations_applied: tuple[str, ...]
    repair_attempted: bool
    repair_succeeded: bool
    issues: tuple[ContractIssue, ...]
    warnings: tuple[str, ...]
    input_hash: str
    output_hash: str


def ingest_contract(
    raw: Any,
    *,
    expected_family: str | None,
    source_kind: SourceKind,
    migration_policy: MigrationPolicy,
    repair_policy: RepairPolicy,
    repair_provider: RepairProvider | None = None,
    allowed_references: Iterable[str] = (),
    normalizer: Normalizer | None = None,
    registry: ContractRegistry = CONTRACT_REGISTRY,
    migration_graph: ContractMigrationGraph = CONTRACT_MIGRATIONS,
) -> ContractIngestResult:
    """Parse, dispatch, migrate, validate, diagnose, and hash one serialized contract."""
    migration_policy = MigrationPolicy(migration_policy)
    repair_policy = RepairPolicy(repair_policy)
    _validate_policy(source_kind, repair_policy, repair_provider)
    payload = _parse_raw(raw)
    input_hash = sha256_contract(payload)
    original_payload = deepcopy(payload)

    family, version = _dispatch_header(
        payload,
        expected_family=expected_family,
        source_kind=source_kind,
        migration_policy=migration_policy,
    )
    original_family, original_version = family, version
    if expected_family is not None and family != expected_family:
        raise ContractMigrationError(
            f"Contract declares family {family!r}; expected {expected_family!r}",
            issues=(ContractIssue(
                "contract_family", family, expected_family, "unexpected contract family",
                stage="contract_header", issue_type="family_mismatch",
            ),),
            stage="contract_header",
            contract=expected_family,
        )

    current_version = registry.current_version(family)
    migrations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    if version != current_version:
        if migration_policy == MigrationPolicy.FORBID:
            registry.resolve(family, version)
        else:
            try:
                execution = migration_graph.apply(
                    payload,
                    contract_family=family,
                    from_version=version,
                    to_version=current_version,
                    validator=lambda target_family, target_version, value: _validate_migration_target(
                        registry, target_family, target_version, value
                    ),
                )
            except ContractMigrationError:
                if (
                    migration_policy == MigrationPolicy.ALLOW
                    and version != UNVERSIONED_SCHEMA_VERSION
                ):
                    registry.resolve(family, version)
                else:
                    raise
            else:
                payload = execution.payload
                version = current_version
                migrations = tuple(execution.migrations_applied)
                warnings = tuple(execution.warnings)
    else:
        registry.resolve(family, version)

    model = registry.resolve(family, version)
    if repair_policy == RepairPolicy.NORMALIZE:
        payload = normalize_contract_payload(payload, model.__name__)
        if normalizer is not None:
            payload = deepcopy(dict(normalizer(deepcopy(payload))))
            warnings = (*warnings, "Applied configured narrow external normalization.")

    try:
        contract = model.model_validate(payload)
        initial_issues: tuple[ContractIssue, ...] = ()
    except ValidationError as error:
        initial_issues = _validation_issues(error, stage="schema_validation")
        if source_kind == "internal_deterministic":
            raise InternalContractValidationError(
                "Deterministic producer emitted an invalid contract",
                issues=initial_issues,
                stage="contract_ingest",
                contract=family,
            ) from error
        if repair_policy != RepairPolicy.ONE_BOUNDED_REPAIR:
            raise ContractValidationError(
                "Incoming contract failed schema validation",
                issues=initial_issues,
                stage="contract_ingest",
                contract=family,
            ) from error
        repaired_payload = _repair_once(
            payload,
            provider=repair_provider,
            family=family,
            version=version,
            model=model,
            issues=initial_issues,
            allowed_references=allowed_references,
        )
        try:
            contract = model.model_validate(repaired_payload)
        except ValidationError as final_error:
            raise BoundaryRepairError(
                f"{family} remained invalid after one bounded repair",
                issues=_validation_issues(final_error, stage="repair_validation"),
                stage="contract_ingest",
                contract=family,
            ) from final_error
        payload = repaired_payload
        repair_attempted = True
        repair_succeeded = True
    else:
        repair_attempted = False
        repair_succeeded = False

    if original_payload != _parse_raw(raw):
        raise RuntimeError("Contract ingest mutated its raw input")
    return ContractIngestResult(
        contract=contract,
        original_family=original_family,
        original_version=original_version,
        final_family=str(getattr(contract, "contract_family", family)),
        final_version=str(getattr(contract, "schema_version", version)),
        migrations_applied=migrations,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        issues=initial_issues,
        warnings=warnings,
        input_hash=input_hash,
        output_hash=sha256_contract(contract),
    )


def ingest_internal_contract(
    raw: Any,
    *,
    model: type[BaseModel],
    repair: Callable[[Any], Any] | None = None,
    allow_migration: bool = False,
) -> BaseModel:
    """Compatibility helper proving deterministic producer output is never repaired."""
    del repair, allow_migration
    try:
        return model.model_validate(deepcopy(raw))
    except ValidationError as error:
        raise InternalContractValidationError(
            "Deterministic producer emitted an invalid contract",
            issues=_validation_issues(error, stage="schema_validation"),
            stage="contract_ingest",
            contract=model.__name__,
        ) from error


def _validate_policy(
    source_kind: SourceKind,
    repair_policy: RepairPolicy,
    repair_provider: RepairProvider | None,
) -> None:
    if source_kind == "internal_deterministic" and repair_policy != RepairPolicy.FORBID:
        raise InternalContractValidationError(
            "Internal deterministic contracts cannot use normalization or repair",
            stage="contract_ingest_policy",
        )
    if source_kind != "llm_generated" and repair_policy == RepairPolicy.ONE_BOUNDED_REPAIR:
        raise ContractValidationError(
            "Bounded schema repair is restricted to LLM-generated contracts",
            stage="contract_ingest_policy",
        )
    if repair_policy == RepairPolicy.ONE_BOUNDED_REPAIR and repair_provider is None:
        raise ContractValidationError(
            "One bounded repair requires an explicit repair provider",
            stage="contract_ingest_policy",
        )


def _dispatch_header(
    payload: dict[str, Any],
    *,
    expected_family: str | None,
    source_kind: SourceKind,
    migration_policy: MigrationPolicy,
) -> tuple[str, str]:
    try:
        header = ContractHeader.model_validate(payload)
    except ValidationError as error:
        if (
            source_kind == "external_artifact"
            and expected_family is not None
            and migration_policy != MigrationPolicy.FORBID
            and ("contract_family" not in payload or "schema_version" not in payload)
        ):
            return expected_family, UNVERSIONED_SCHEMA_VERSION
        issues = _validation_issues(error, stage="contract_header")
        if source_kind == "internal_deterministic":
            raise InternalContractValidationError(
                "Deterministic contract has no valid dispatch header",
                issues=issues,
                stage="contract_header",
                contract=expected_family,
            ) from error
        raise ContractValidationError(
            "Incoming contract has no valid dispatch header",
            issues=issues,
            stage="contract_header",
            contract=expected_family,
        ) from error
    return header.contract_family, header.schema_version


def _validate_migration_target(
    registry: ContractRegistry,
    family: str,
    version: str,
    payload: Mapping[str, Any],
) -> None:
    model = registry.resolve(family, version)
    try:
        model.model_validate(payload)
    except ValidationError as error:
        raise ContractMigrationError(
            f"Migrated {family} output failed target schema {version}",
            issues=_validation_issues(error, stage="migration_validation"),
            stage="contract_migration",
            contract=family,
        ) from error


def _repair_once(
    payload: dict[str, Any],
    *,
    provider: RepairProvider | None,
    family: str,
    version: str,
    model: type[BaseModel],
    issues: tuple[ContractIssue, ...],
    allowed_references: Iterable[str],
) -> dict[str, Any]:
    allowed = frozenset(str(value) for value in allowed_references)
    request = {
        "contract_family": family,
        "schema_version": version,
        "validation_issues": [issue.as_dict() for issue in issues],
        "canonical_fields": sorted(model.model_fields),
        "allowed_references": sorted(allowed),
        "original_payload": deepcopy(payload),
    }
    repair = provider.repair if hasattr(provider, "repair") else provider
    assert repair is not None
    repaired = repair(request)
    if inspect.isawaitable(repaired):
        raise BoundaryRepairError(
            "Async repair providers require an async boundary adapter",
            stage="contract_ingest",
            contract=family,
        )
    if not isinstance(repaired, Mapping):
        raise BoundaryRepairError(
            "Repair provider returned a non-object payload",
            stage="contract_ingest",
            contract=family,
        )
    repaired_payload = deepcopy(dict(repaired))
    original_refs = _reference_values(payload)
    repaired_refs = _reference_values(repaired_payload)
    unsupported = (
        repaired_refs - allowed
        if allowed
        else repaired_refs - original_refs
    )
    if unsupported:
        raise BoundaryRepairError(
            "Repair introduced unsupported references",
            issues=tuple(ContractIssue(
                "repair.references", reference, "allowed reference",
                "repair introduced unsupported reference",
                stage="repair_validation",
                issue_type="unsupported_reference",
                reference=reference,
            ) for reference in sorted(unsupported)),
            stage="contract_ingest",
            contract=family,
        )
    return repaired_payload


def _reference_values(value: Any, *, parent_key: str = "") -> set[str]:
    references: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key = str(key)
            if key.endswith(("_refs", "_ids")) and isinstance(child, list):
                references.update(str(item) for item in child if isinstance(item, str))
            elif key.endswith("_id") and isinstance(child, str):
                references.add(child)
            else:
                references.update(_reference_values(child, parent_key=key))
    elif isinstance(value, list):
        for item in value:
            references.update(_reference_values(item, parent_key=parent_key))
    return references


def _validation_issues(
    error: ValidationError, *, stage: str
) -> tuple[ContractIssue, ...]:
    return tuple(ContractIssue(
        ".".join(str(part) for part in item.get("loc", ())),
        item.get("input"),
        item.get("type", "valid contract field"),
        item.get("msg", "validation failed"),
        stage=stage,
        issue_type=str(item.get("type", "validation_error")),
    ) for item in error.errors(include_url=False))


def _parse_raw(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return deepcopy(dict(raw))
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ContractValidationError(
                "Incoming contract is not valid unambiguous JSON",
                issues=(ContractIssue(
                    "json", type(raw).__name__, "JSON object",
                    "invalid JSON or duplicate object key",
                    stage="json_parse", issue_type="json_parse_error",
                ),),
                stage="json_parse",
            ) from error
        if not isinstance(payload, dict):
            raise ContractValidationError(
                "Top-level serialized contract must be a JSON object",
                stage="json_parse",
            )
        return payload
    raise ContractValidationError(
        "Contract ingest accepts a mapping, JSON string, or JSON bytes",
        stage="json_parse",
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


__all__ = [
    "ContractIngestResult",
    "ContractRepairProvider",
    "MigrationPolicy",
    "RepairPolicy",
    "ingest_contract",
    "ingest_internal_contract",
]
