from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Literal

from pydantic import ConfigDict, Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.errors import (
    AmbiguousReferenceError,
    ContractIssue,
    EvidenceManifestBuildError,
    EvidenceManifestConflictError,
    EvidenceManifestPackMismatchError,
    UnknownReferenceError,
)
from kaggle_researcher.contracts.evidence import (
    EvidenceNamespace,
    EvidencePathResolutionError,
    SEMANTIC_COLLECTION_PATHS,
    _iter_dictionary_paths,
    resolve_evidence_path,
)
from kaggle_researcher.contracts.hashing import (
    CANONICAL_HASH_POLICY_VERSION,
    sha256_contract,
)


EvidenceReferenceKind = Literal["direct_path", "semantic_ref", "contract_ref"]
EvidenceConflictType = Literal[
    "duplicate_identity",
    "cross_namespace_collision",
    "semantic_alias_collision",
    "exact_duplicate_registration",
    "deprecated_alias",
]


class EvidenceConflictPolicy(str, Enum):
    STRICT = "strict"
    DEGRADED = "degraded"


class _FrozenContractModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        frozen=True,
    )


class EvidenceReferenceEntry(_FrozenContractModel):
    ref: str = Field(min_length=1)
    namespace: EvidenceNamespace
    reference_kind: EvidenceReferenceKind
    canonical_path: str | None = None
    value_type: str | None = None
    source_component: str = Field(min_length=1)
    available: bool = True
    semantic_identity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_identity(self) -> QualifiedEvidenceReference:
        return QualifiedEvidenceReference(namespace=self.namespace, ref=self.ref)


class QualifiedEvidenceReference(_FrozenContractModel):
    """Canonical structured identity; serialized references never infer a namespace."""

    namespace: EvidenceNamespace
    ref: str = Field(min_length=1)


class EvidenceReferenceConflict(_FrozenContractModel):
    ref: str = Field(min_length=1)
    namespaces: list[EvidenceNamespace]
    canonical_paths: list[str]
    conflict_type: EvidenceConflictType
    severity: Literal["warning", "error"]
    message: str = Field(min_length=1)


class EvidenceReferenceManifest(_FrozenContractModel):
    contract_family: Literal["evidence_reference_manifest"] = "evidence_reference_manifest"
    schema_version: Literal["1.0"] = "1.0"
    manifest_version: str = "1.0"
    pack_hash: str = Field(min_length=64, max_length=64)
    hash_policy_version: str = CANONICAL_HASH_POLICY_VERSION
    entries: list[EvidenceReferenceEntry] = Field(default_factory=list)
    conflicts: list[EvidenceReferenceConflict] = Field(default_factory=list)
    generated_at_stage: Literal["eda_publication_boundary"] = "eda_publication_boundary"
    origin: Literal["eda_publication", "legacy_migration"] = "eda_publication"
    manifest_hash: str = Field(min_length=64, max_length=64)

    @property
    def available_refs(self) -> frozenset[str]:
        return frozenset(entry.ref for entry in self.entries if entry.available)

    @property
    def entries_by_ref(self) -> dict[str, tuple[EvidenceReferenceEntry, ...]]:
        grouped: dict[str, list[EvidenceReferenceEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.ref].append(entry)
        return {key: tuple(value) for key, value in grouped.items()}

    def namespace_for(self, ref: str) -> EvidenceNamespace | None:
        namespaces = {
            entry.namespace for entry in self.entries
            if entry.ref == ref and entry.available
        }
        return next(iter(namespaces)) if len(namespaces) == 1 else None

    def entry_for(self, reference: QualifiedEvidenceReference) -> EvidenceReferenceEntry:
        matches = [
            entry for entry in self.entries
            if entry.available
            and entry.ref == reference.ref
            and entry.namespace == reference.namespace
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousReferenceError(
                f"Qualified evidence reference {reference.namespace}:{reference.ref} is duplicated",
                issues=(ContractIssue(
                    "evidence_reference", reference.model_dump(mode="json"),
                    "one active manifest entry", "ambiguous qualified reference",
                    reference.namespace,
                ),),
                contract="evidence_reference_manifest",
            )
        raise UnknownReferenceError(
            f"Unknown qualified evidence reference {reference.namespace}:{reference.ref}",
            issues=(ContractIssue(
                "evidence_reference", reference.model_dump(mode="json"),
                "active manifest entry", "unknown reference", reference.namespace,
            ),),
            contract="evidence_reference_manifest",
        )

    def migrate_legacy_ref(self, ref: str) -> QualifiedEvidenceReference:
        """Migrate an unqualified legacy ref only when ownership is unique."""
        all_matches = {
            (entry.namespace, entry.ref)
            for entry in self.entries if entry.ref == ref
        }
        active_matches = {
            (entry.namespace, entry.ref)
            for entry in self.entries if entry.available and entry.ref == ref
        }
        if len(all_matches) == 1 and len(active_matches) == 1:
            namespace, canonical_ref = next(iter(active_matches))
            return QualifiedEvidenceReference(namespace=namespace, ref=canonical_ref)
        if len(all_matches) > 1 or (all_matches and not active_matches):
            raise AmbiguousReferenceError(
                f"Legacy evidence reference {ref!r} has ambiguous namespace ownership",
                issues=(ContractIssue(
                    "legacy_evidence_ref", ref, "unique namespace-qualified reference",
                    "ambiguous legacy unqualified reference",
                ),),
                contract="evidence_reference_manifest",
            )
        raise UnknownReferenceError(
            f"Unknown legacy evidence reference {ref!r}",
            issues=(ContractIssue(
                "legacy_evidence_ref", ref, "active manifest entry", "unknown reference",
            ),),
            contract="evidence_reference_manifest",
        )


class PublishedEdaEvidenceBundle(_FrozenContractModel):
    contract_family: Literal["published_eda_evidence_bundle"] = "published_eda_evidence_bundle"
    schema_version: Literal["1.0"] = "1.0"
    evidence_pack: EdaEvidencePack
    evidence_manifest: EvidenceReferenceManifest
    pack_hash: str = Field(min_length=64, max_length=64)
    manifest_hash: str = Field(min_length=64, max_length=64)
    bundle_hash: str = Field(min_length=64, max_length=64)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class BundleValidationResult(_FrozenContractModel):
    valid: Literal[True] = True
    pack_hash: str = Field(min_length=64, max_length=64)
    manifest_hash: str = Field(min_length=64, max_length=64)
    bundle_hash: str = Field(min_length=64, max_length=64)
    schema_versions: dict[str, str]
    available_entry_count: int = Field(ge=0)
    unavailable_entry_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


_CONTRACT_REFERENCE_SPECS: tuple[tuple[str, str, EvidenceNamespace], ...] = (
    ("eda_risks", "risk_id", "risk"),
    ("eda_risk_register", "risk_id", "risk"),
    ("validation_requirements", "validation_requirement_id", "validation_requirement"),
    ("safety_constraints", "safety_constraint_id", "safety_constraint"),
    ("hypothesis_results", "hypothesis_id", "hypothesis"),
    ("testable_hypotheses", "hypothesis_id", "hypothesis"),
)


def build_evidence_reference_manifest(
    evidence_pack: EdaEvidencePack,
    *,
    origin: Literal["eda_publication", "legacy_migration"] = "eda_publication",
) -> EvidenceReferenceManifest:
    """Publish the sole canonical reference-space snapshot for a finalized EDA pack."""
    if not isinstance(evidence_pack, EdaEvidencePack):
        raise EvidenceManifestBuildError(
            "Evidence manifest requires a validated EdaEvidencePack",
            issues=(ContractIssue(
                "evidence_pack", type(evidence_pack).__name__, "EdaEvidencePack",
                "unvalidated evidence publication input",
            ),),
            stage="eda_publication_boundary",
            contract="evidence_reference_manifest",
        )
    snapshot = EdaEvidencePack.model_validate(evidence_pack.model_dump(mode="python"))
    payload = snapshot.model_dump(mode="json")
    pack_hash = sha256_contract(snapshot)
    entries: list[EvidenceReferenceEntry] = []
    conflicts: list[EvidenceReferenceConflict] = []

    for path, value in _iter_dictionary_paths(payload):
        entries.append(EvidenceReferenceEntry(
            ref=path,
            namespace="eda_evidence",
            reference_kind="direct_path",
            canonical_path=path,
            value_type=_value_type(value),
            source_component=path.split(".", 1)[0].split("[", 1)[0],
        ))

    for collection_path, spec in sorted(SEMANTIC_COLLECTION_PATHS.items()):
        collection = _dictionary_path(payload, collection_path)
        if not isinstance(collection, list):
            continue
        identities: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                continue
            identity = item.get(spec.identity_field)
            if isinstance(identity, str) and identity.strip():
                identities[identity.strip()].append(index)
        for identity, indexes in sorted(identities.items()):
            ref = f"{collection_path}.{identity}"
            paths = [f"{collection_path}[{index}]" for index in indexes]
            if len(indexes) > 1:
                conflicts.append(EvidenceReferenceConflict(
                    ref=ref,
                    namespaces=["eda_evidence"],
                    canonical_paths=paths,
                    conflict_type="duplicate_identity",
                    severity="error",
                    message=(
                        f"Semantic identity {identity!r} occurs {len(indexes)} times "
                        f"in {collection_path}."
                    ),
                ))
            entries.append(EvidenceReferenceEntry(
                ref=ref,
                namespace="eda_evidence",
                reference_kind="semantic_ref",
                canonical_path=paths[0],
                value_type="object",
                source_component=collection_path.split(".", 1)[0],
                available=len(indexes) == 1,
                semantic_identity=identity,
                metadata={"identity_field": spec.identity_field, "match_count": len(indexes)},
            ))

    entries.extend(_contract_reference_entries(payload))
    entries, normalization_conflicts = _normalize_entries(entries)
    conflicts.extend(normalization_conflicts)
    entries, collision_conflicts = _mark_collisions(entries)
    conflicts.extend(collision_conflicts)
    entries.sort(key=_entry_sort_key)
    conflicts = _collapse_conflicts(conflicts)

    manifest_payload: dict[str, Any] = {
        "contract_family": "evidence_reference_manifest",
        "schema_version": "1.0",
        "manifest_version": "1.0",
        "pack_hash": pack_hash,
        "hash_policy_version": CANONICAL_HASH_POLICY_VERSION,
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "conflicts": [conflict.model_dump(mode="json") for conflict in conflicts],
        "generated_at_stage": "eda_publication_boundary",
        "origin": origin,
    }
    manifest_hash = sha256_contract(manifest_payload)
    return EvidenceReferenceManifest(**manifest_payload, manifest_hash=manifest_hash)


def publish_eda_evidence_bundle(
    evidence_pack: EdaEvidencePack,
    *,
    conflict_policy: EvidenceConflictPolicy = EvidenceConflictPolicy.STRICT,
    manifest_origin: Literal["eda_publication", "legacy_migration"] = "eda_publication",
    migration_warnings: tuple[str, ...] = (),
) -> PublishedEdaEvidenceBundle:
    snapshot = EdaEvidencePack.model_validate(evidence_pack.model_dump(mode="python"))
    manifest = build_evidence_reference_manifest(snapshot, origin=manifest_origin)
    errors = [conflict for conflict in manifest.conflicts if conflict.severity == "error"]
    if errors and conflict_policy == EvidenceConflictPolicy.STRICT:
        raise EvidenceManifestConflictError(
            "Strict EDA evidence publication rejected reference conflicts",
            manifest=manifest,
            issues=tuple(
                ContractIssue(
                    f"conflicts.{conflict.ref}", conflict.ref,
                    "unambiguous published evidence reference", conflict.conflict_type,
                )
                for conflict in errors
            ),
        )
    warnings = [*snapshot.warnings, *migration_warnings]
    limitations = list(snapshot.limitations)
    if errors:
        warnings.append(
            f"Published degraded evidence manifest with {len(errors)} blocking conflict(s)."
        )
        limitations.append(
            "Ambiguous evidence references were excluded from the published reference space."
        )
    bundle_identity = {
        "contract_family": "published_eda_evidence_bundle",
        "schema_version": "1.0",
        "pack_hash": manifest.pack_hash,
        "manifest_hash": manifest.manifest_hash,
    }
    bundle = PublishedEdaEvidenceBundle(
        evidence_pack=snapshot,
        evidence_manifest=manifest,
        pack_hash=manifest.pack_hash,
        manifest_hash=manifest.manifest_hash,
        bundle_hash=sha256_contract(bundle_identity),
        warnings=list(dict.fromkeys([
            *warnings,
            *(
                conflict.message
                for conflict in manifest.conflicts if conflict.severity == "warning"
            ),
        ])),
        limitations=list(dict.fromkeys(limitations)),
    )
    validate_published_eda_bundle(bundle)
    return bundle


def validate_published_eda_evidence_bundle(
    bundle: PublishedEdaEvidenceBundle,
) -> BundleValidationResult:
    """Backward-compatible name for the canonical bundle validator."""
    return validate_published_eda_bundle(bundle)


def validate_published_eda_bundle(
    bundle: PublishedEdaEvidenceBundle,
) -> BundleValidationResult:
    """Validate hashes and all manifest invariants against the embedded immutable pack."""
    actual_pack_hash = sha256_contract(bundle.evidence_pack)
    if (
        actual_pack_hash != bundle.pack_hash
        or bundle.evidence_manifest.pack_hash != bundle.pack_hash
    ):
        raise EvidenceManifestPackMismatchError(
            expected_hash=bundle.pack_hash,
            actual_hash=actual_pack_hash,
            manifest_hash=bundle.manifest_hash,
            bundle_hash=bundle.bundle_hash,
            manifest_schema_version=bundle.evidence_manifest.schema_version,
            bundle_schema_version=bundle.schema_version,
        )
    manifest_payload = bundle.evidence_manifest.model_dump(mode="json")
    declared_manifest_hash = manifest_payload.pop("manifest_hash")
    actual_manifest_hash = sha256_contract(manifest_payload)
    if (
        actual_manifest_hash != declared_manifest_hash
        or bundle.manifest_hash != declared_manifest_hash
    ):
        raise EvidenceManifestBuildError(
            "Published evidence manifest hash mismatch",
            issues=(ContractIssue(
                "manifest_hash", actual_manifest_hash, declared_manifest_hash,
                "manifest content differs from its published hash",
            ),),
            stage="published_bundle_validation",
            contract="published_eda_evidence_bundle",
        )
    expected_bundle_hash = sha256_contract({
        "contract_family": bundle.contract_family,
        "schema_version": bundle.schema_version,
        "pack_hash": bundle.pack_hash,
        "manifest_hash": bundle.manifest_hash,
    })
    if expected_bundle_hash != bundle.bundle_hash:
        raise EvidenceManifestBuildError(
            "Published evidence bundle hash mismatch",
            issues=(ContractIssue(
                "bundle_hash", bundle.bundle_hash, expected_bundle_hash,
                "bundle identity differs from its published hash",
            ),),
            stage="published_bundle_validation",
            contract="published_eda_evidence_bundle",
        )

    issues: list[ContractIssue] = []
    manifest = bundle.evidence_manifest
    if manifest.hash_policy_version != CANONICAL_HASH_POLICY_VERSION:
        issues.append(ContractIssue(
            "evidence_manifest.hash_policy_version", manifest.hash_policy_version,
            CANONICAL_HASH_POLICY_VERSION, "unsupported canonical hash policy",
        ))
    if manifest.entries != sorted(manifest.entries, key=_entry_sort_key):
        issues.append(ContractIssue(
            "evidence_manifest.entries", "non-canonical order", "deterministic sorted order",
            "manifest entries are not deterministically ordered",
        ))
    expected_conflicts = sorted(manifest.conflicts, key=_conflict_sort_key)
    if manifest.conflicts != expected_conflicts:
        issues.append(ContractIssue(
            "evidence_manifest.conflicts", "non-canonical order", "deterministic sorted order",
            "manifest conflicts are not deterministically ordered",
        ))

    blocking_conflicts = [
        conflict for conflict in manifest.conflicts if conflict.severity == "error"
    ]
    active_identities: set[tuple[str, str]] = set()
    active_refs: dict[str, set[str]] = defaultdict(set)
    for index, entry in enumerate(manifest.entries):
        field = f"evidence_manifest.entries[{index}]"
        if not entry.canonical_path:
            issues.append(ContractIssue(
                f"{field}.canonical_path", entry.canonical_path, "resolvable canonical path",
                "missing canonical path",
            ))
        else:
            try:
                resolved = resolve_evidence_path(entry.canonical_path, bundle.evidence_pack)
            except EvidencePathResolutionError as exc:
                issues.append(ContractIssue(
                    f"{field}.canonical_path", entry.canonical_path,
                    "path present in embedded evidence pack",
                    (f"unresolvable canonical path {entry.canonical_path!r}: "
                     f"{type(exc).__name__}"),
                ))
            else:
                actual_type = _value_type(resolved.value)
                if entry.value_type and actual_type != entry.value_type:
                    issues.append(ContractIssue(
                        f"{field}.value_type", entry.value_type, actual_type,
                        "declared value type differs from canonical content",
                    ))
        if not entry.available and not any(
            conflict.ref == entry.ref and entry.namespace in conflict.namespaces
            for conflict in blocking_conflicts
        ):
            issues.append(ContractIssue(
                f"{field}.available", False, "recorded error-severity conflict",
                "unavailable entry has no blocking conflict",
            ))
        if entry.available:
            identity = (entry.namespace, entry.ref)
            if identity in active_identities:
                issues.append(ContractIssue(
                    field, {"namespace": entry.namespace, "ref": entry.ref},
                    "one active qualified reference", "duplicate active reference",
                    entry.namespace,
                ))
            active_identities.add(identity)
            active_refs[entry.ref].add(entry.namespace)
        if entry.reference_kind == "semantic_ref":
            if not entry.semantic_identity or not entry.metadata.get("identity_field"):
                issues.append(ContractIssue(
                    field, entry.model_dump(mode="json"),
                    "semantic identity, identity field, component, type, and canonical path",
                    "incomplete semantic reference identity",
                ))

    for ref, namespaces in sorted(active_refs.items()):
        if len(namespaces) > 1:
            issues.append(ContractIssue(
                "evidence_manifest.entries", {"ref": ref, "namespaces": sorted(namespaces)},
                "single active namespace owner", "active cross-namespace collision",
            ))

    severity_by_type = {
        "duplicate_identity": "error",
        "cross_namespace_collision": "error",
        "semantic_alias_collision": "error",
        "exact_duplicate_registration": "warning",
        "deprecated_alias": "warning",
    }
    for index, conflict in enumerate(manifest.conflicts):
        expected_severity = severity_by_type[conflict.conflict_type]
        if conflict.severity != expected_severity:
            issues.append(ContractIssue(
                f"evidence_manifest.conflicts[{index}].severity", conflict.severity,
                expected_severity, "non-deterministic conflict severity",
            ))
        if conflict.severity == "error" and any(
            entry.available and entry.ref == conflict.ref for entry in manifest.entries
        ):
            issues.append(ContractIssue(
                f"evidence_manifest.conflicts[{index}]", conflict.ref,
                "all conflicting references unavailable", "blocking conflict remains active",
            ))

    if issues:
        raise EvidenceManifestBuildError(
            "Published EDA evidence bundle invariant validation failed",
            issues=issues,
            stage="published_bundle_validation",
            contract="published_eda_evidence_bundle",
        )
    return BundleValidationResult(
        pack_hash=bundle.pack_hash,
        manifest_hash=bundle.manifest_hash,
        bundle_hash=bundle.bundle_hash,
        schema_versions={
            "evidence_pack": bundle.evidence_pack.schema_version,
            "evidence_reference_manifest": manifest.schema_version,
            "published_eda_evidence_bundle": bundle.schema_version,
        },
        available_entry_count=sum(entry.available for entry in manifest.entries),
        unavailable_entry_count=sum(not entry.available for entry in manifest.entries),
        warnings=[
            conflict.message
            for conflict in manifest.conflicts if conflict.severity == "warning"
        ],
    )


def _dictionary_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _contract_reference_entries(payload: dict[str, Any]) -> list[EvidenceReferenceEntry]:
    entries: list[EvidenceReferenceEntry] = []
    for collection_name, identity_field, namespace in _CONTRACT_REFERENCE_SPECS:
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                continue
            identity = item.get(identity_field)
            if not isinstance(identity, str) or not identity.strip():
                continue
            entries.append(EvidenceReferenceEntry(
                ref=identity.strip(),
                namespace=namespace,
                reference_kind="contract_ref",
                canonical_path=f"{collection_name}[{index}].{identity_field}",
                value_type="string",
                source_component=collection_name,
                semantic_identity=identity.strip(),
                metadata={"identity_field": identity_field},
            ))
    return entries


def _normalize_entries(
    entries: list[EvidenceReferenceEntry],
) -> tuple[list[EvidenceReferenceEntry], list[EvidenceReferenceConflict]]:
    """Collapse only proven duplicates and make every collapse observable."""
    exact: dict[str, list[EvidenceReferenceEntry]] = defaultdict(list)
    for entry in entries:
        exact[sha256_contract(entry)].append(entry)
    deduplicated: list[EvidenceReferenceEntry] = []
    conflicts: list[EvidenceReferenceConflict] = []
    for _, candidates in sorted(exact.items()):
        candidates.sort(key=_entry_sort_key)
        chosen = candidates[0]
        deduplicated.append(chosen)
        if len(candidates) > 1:
            conflicts.append(EvidenceReferenceConflict(
                ref=chosen.ref,
                namespaces=[chosen.namespace],
                canonical_paths=[chosen.canonical_path] if chosen.canonical_path else [],
                conflict_type="exact_duplicate_registration",
                severity="warning",
                message=f"Collapsed {len(candidates)} exact registrations for {chosen.ref!r}.",
            ))

    grouped: dict[tuple[str, str, str], list[EvidenceReferenceEntry]] = defaultdict(list)
    for entry in deduplicated:
        grouped[(entry.ref, entry.namespace, entry.reference_kind)].append(entry)
    normalized: list[EvidenceReferenceEntry] = []
    for _, candidates in sorted(grouped.items()):
        candidates.sort(key=_contract_preference_key)
        if len(candidates) == 1:
            normalized.append(candidates[0])
            continue
        chosen = candidates[0]
        paths = sorted({item.canonical_path for item in candidates if item.canonical_path})
        if chosen.reference_kind == "contract_ref":
            components = {item.source_component for item in candidates}
            if len(components) == 1:
                chosen = chosen.model_copy(update={"available": False})
                conflicts.append(EvidenceReferenceConflict(
                    ref=chosen.ref,
                    namespaces=[chosen.namespace],
                    canonical_paths=paths,
                    conflict_type="duplicate_identity",
                    severity="error",
                    message=(f"Unique identity {chosen.ref!r} is registered multiple times "
                             f"in {chosen.source_component}."),
                ))
            else:
                chosen = chosen.model_copy(update={
                    "metadata": {
                        **chosen.metadata,
                        "canonical_aliases": paths,
                        "deprecated_alias_paths": [
                            path for path in paths if path != chosen.canonical_path
                        ],
                    },
                })
                conflicts.append(EvidenceReferenceConflict(
                    ref=chosen.ref,
                    namespaces=[chosen.namespace],
                    canonical_paths=paths,
                    conflict_type="deprecated_alias",
                    severity="warning",
                    message=(f"Deprecated duplicate source for {chosen.ref!r} was mapped "
                             f"to canonical component {chosen.source_component!r}."),
                ))
            normalized.append(chosen)
            continue
        # Multiple semantic paths may never be represented by one active alias.
        normalized.append(chosen.model_copy(update={"available": False}))
        conflicts.append(EvidenceReferenceConflict(
            ref=chosen.ref,
            namespaces=sorted({item.namespace for item in candidates}),
            canonical_paths=paths,
            conflict_type="semantic_alias_collision",
            severity="error",
            message=f"Reference {chosen.ref!r} maps to multiple canonical paths.",
        ))
    return normalized, conflicts


def _contract_preference_key(entry: EvidenceReferenceEntry) -> tuple[int, str, str, str, str]:
    preferred_components = {
        "eda_risks": 0,
        "hypothesis_results": 0,
        "validation_requirements": 0,
        "safety_constraints": 0,
        "eda_risk_register": 1,
        "testable_hypotheses": 1,
    }
    return (
        preferred_components.get(entry.source_component, 0),
        entry.ref,
        entry.namespace,
        entry.reference_kind,
        entry.canonical_path or "",
    )


def _mark_collisions(
    entries: list[EvidenceReferenceEntry],
) -> tuple[list[EvidenceReferenceEntry], list[EvidenceReferenceConflict]]:
    by_ref: dict[str, list[EvidenceReferenceEntry]] = defaultdict(list)
    for entry in entries:
        by_ref[entry.ref].append(entry)
    blocked: set[tuple[str, str, str]] = set()
    conflicts: list[EvidenceReferenceConflict] = []
    for ref, candidates in sorted(by_ref.items()):
        namespaces = sorted({entry.namespace for entry in candidates})
        kinds = {entry.reference_kind for entry in candidates}
        conflict_type: EvidenceConflictType | None = None
        if len(namespaces) > 1:
            conflict_type = "cross_namespace_collision"
        elif "semantic_ref" in kinds and len(kinds) > 1:
            conflict_type = "semantic_alias_collision"
        elif len({entry.canonical_path for entry in candidates}) > 1 and kinds == {"semantic_ref"}:
            conflict_type = "semantic_alias_collision"
        if conflict_type is None:
            continue
        for entry in candidates:
            blocked.add((entry.ref, entry.namespace, entry.reference_kind))
        conflicts.append(EvidenceReferenceConflict(
            ref=ref,
            namespaces=namespaces,
            canonical_paths=sorted({
                entry.canonical_path for entry in candidates if entry.canonical_path
            }),
            conflict_type=conflict_type,
            severity="error",
            message=f"Reference {ref!r} has conflicting published ownership ({conflict_type}).",
        ))
    return [
        entry.model_copy(update={"available": False})
        if (entry.ref, entry.namespace, entry.reference_kind) in blocked
        else entry
        for entry in entries
    ], conflicts


def _collapse_conflicts(
    conflicts: list[EvidenceReferenceConflict],
) -> list[EvidenceReferenceConflict]:
    # Deduplicate byte-identical diagnostics only; distinct conflicts remain observable.
    by_key: dict[str, EvidenceReferenceConflict] = {}
    for conflict in conflicts:
        by_key[sha256_contract(conflict)] = conflict
    return sorted(
        by_key.values(), key=_conflict_sort_key
    )


def _conflict_sort_key(
    conflict: EvidenceReferenceConflict,
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    return (
        conflict.ref,
        conflict.conflict_type,
        conflict.message,
        tuple(conflict.namespaces),
        tuple(conflict.canonical_paths),
    )


def _entry_sort_key(entry: EvidenceReferenceEntry) -> tuple[str, str, str, str]:
    return (
        entry.ref,
        entry.namespace,
        entry.reference_kind,
        entry.canonical_path or "",
    )


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


__all__ = [
    "BundleValidationResult",
    "EvidenceConflictPolicy",
    "EvidenceReferenceConflict",
    "EvidenceReferenceEntry",
    "EvidenceReferenceKind",
    "EvidenceReferenceManifest",
    "PublishedEdaEvidenceBundle",
    "QualifiedEvidenceReference",
    "build_evidence_reference_manifest",
    "publish_eda_evidence_bundle",
    "validate_published_eda_bundle",
    "validate_published_eda_evidence_bundle",
]
