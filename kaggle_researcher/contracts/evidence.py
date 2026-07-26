from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from kaggle_researcher.contracts.evidence_manifest import EvidenceReferenceManifest


EvidenceNamespace = Literal[
    "eda_evidence",
    "reasoning_evidence",
    "source_claim",
    "hypothesis",
    "synthetic_inference",
    "risk",
    "validation_requirement",
    "safety_constraint",
]
ResolutionKind = Literal[
    "top_level", "dict_path", "list_index", "semantic_collection_item"
]


@dataclass(frozen=True)
class SemanticCollectionSpec:
    identity_field: str


SEMANTIC_COLLECTION_PATHS: dict[str, SemanticCollectionSpec] = {
    "baseline_ablation_evidence.ablations": SemanticCollectionSpec("ablation_id"),
    "baseline_ablation_evidence.feature_block_findings": SemanticCollectionSpec("feature_block"),
    "eda_implications": SemanticCollectionSpec("implication_id"),
    "eda_risks": SemanticCollectionSpec("risk_id"),
    "eda_risk_register": SemanticCollectionSpec("risk_id"),
    "feature_probe_evidence": SemanticCollectionSpec("feature_family"),
    "hypothesis_results": SemanticCollectionSpec("hypothesis_id"),
    "interaction_diagnostics.interaction_hypotheses": SemanticCollectionSpec("interaction_id"),
    "leakage_evidence": SemanticCollectionSpec("check_id"),
    "slice_diagnostics.slices": SemanticCollectionSpec("slice_id"),
    "source_claim_validation.validated_claims": SemanticCollectionSpec("claim_id"),
    "testable_hypotheses": SemanticCollectionSpec("hypothesis_id"),
}


class EvidencePathResolutionError(ValueError):
    def __init__(self, reference: str, reason: str = "unknown") -> None:
        self.reference = reference
        self.reason = reason
        super().__init__(f"Could not resolve EDA evidence path {reference!r}: {reason}.")


class AmbiguousEvidencePathError(EvidencePathResolutionError):
    def __init__(self, reference: str, match_count: int) -> None:
        self.match_count = match_count
        super().__init__(reference, "ambiguous_semantic_path")


@dataclass(frozen=True)
class ResolvedEvidencePath:
    reference: str
    root_id: str
    value: Any
    resolution_kind: ResolutionKind


class ReferenceResolutionError(ValueError):
    def __init__(
        self,
        reference_id: str,
        *,
        namespaces: Iterable[str],
        reason: str = "unknown",
    ) -> None:
        self.reference_id = reference_id
        self.namespaces = tuple(namespaces)
        self.reason = reason
        super().__init__(
            f"Could not resolve reference {reference_id!r}: {reason}; "
            f"searched namespaces: {', '.join(self.namespaces)}."
        )


@dataclass(frozen=True)
class EvidenceReference:
    namespace: EvidenceNamespace
    reference_id: str


@dataclass(frozen=True)
class ResolvedEvidence:
    reference: EvidenceReference
    value: Any


@dataclass
class EvidenceRegistry:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register(self, namespace: EvidenceNamespace, reference_id: str, value: Any) -> None:
        if not reference_id.strip():
            raise ValueError("Reference IDs must not be empty.")
        self.entries.setdefault(namespace, {})[reference_id] = value

    def ids(self, namespace: EvidenceNamespace) -> frozenset[str]:
        return frozenset(self.entries.get(namespace, {}))


def resolve_evidence_reference(
    reference_id: str,
    registry: EvidenceRegistry,
    *,
    namespaces: Iterable[EvidenceNamespace] | None = None,
) -> ResolvedEvidence:
    allowed = tuple(namespaces or registry.entries.keys())
    matches = [namespace for namespace in allowed if reference_id in registry.entries.get(namespace, {})]
    if not matches:
        raise ReferenceResolutionError(reference_id, namespaces=allowed)
    if len(matches) > 1:
        raise ReferenceResolutionError(reference_id, namespaces=matches, reason="ambiguous namespace")
    namespace = matches[0]
    return ResolvedEvidence(
        EvidenceReference(namespace=namespace, reference_id=reference_id),
        registry.entries[namespace][reference_id],
    )


def build_evidence_registry(
    eda_evidence: BaseModel | Mapping[str, Any] | None = None,
    *,
    reasoning_ids: Iterable[str] = (),
    source_ids: Iterable[str] = (),
    hypothesis_ids: Iterable[str] = (),
) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    if isinstance(eda_evidence, BaseModel):
        eda_payload = eda_evidence.model_dump(mode="json")
    else:
        eda_payload = dict(eda_evidence or {})
    for path, value in _iter_dictionary_paths(eda_payload):
        registry.register("eda_evidence", path, value)
    for path in generate_semantic_evidence_refs(eda_payload):
        resolved = resolve_evidence_path(path, eda_payload)
        registry.register("eda_evidence", path, resolved.value)
    for reference_id in reasoning_ids:
        registry.register("reasoning_evidence", reference_id, reference_id)
    for reference_id in source_ids:
        registry.register("source_claim", reference_id, reference_id)
    for reference_id in hypothesis_ids:
        registry.register("hypothesis", reference_id, reference_id)
    registry.register(
        "synthetic_inference",
        "final_synthesizer.repaired",
        "Deterministic final-strategy fallback marker",
    )
    return registry


def build_evidence_registry_from_manifest(
    manifest: "EvidenceReferenceManifest",
    evidence_pack: BaseModel | Mapping[str, Any],
    *,
    reasoning_ids: Iterable[str] = (),
    source_ids: Iterable[str] = (),
    hypothesis_ids: Iterable[str] = (),
) -> EvidenceRegistry:
    """Materialize the legacy resolver index from a published manifest.

    This adapter keeps downstream APIs stable while ensuring new published runs
    do not regenerate EDA direct and semantic reference ownership for registry
    construction.
    """
    registry = EvidenceRegistry()
    for entry in manifest.entries:
        if not entry.available or entry.namespace != "eda_evidence":
            continue
        if entry.canonical_path is None:
            continue
        value = resolve_evidence_path(entry.canonical_path, evidence_pack).value
        registry.register("eda_evidence", entry.ref, value)
    for reference_id in reasoning_ids:
        registry.register("reasoning_evidence", reference_id, reference_id)
    for reference_id in source_ids:
        registry.register("source_claim", reference_id, reference_id)
    for reference_id in hypothesis_ids:
        registry.register("hypothesis", reference_id, reference_id)
    registry.register(
        "synthetic_inference",
        "final_synthesizer.repaired",
        "Deterministic final-strategy fallback marker",
    )
    return registry


def resolve_evidence_path(
    reference: str,
    evidence_pack: Mapping[str, Any] | BaseModel,
    *,
    semantic_collections: Mapping[str, SemanticCollectionSpec] = SEMANTIC_COLLECTION_PATHS,
) -> ResolvedEvidencePath:
    payload = (
        evidence_pack.model_dump(mode="json")
        if isinstance(evidence_pack, BaseModel)
        else evidence_pack
    )
    parts = _evidence_path_parts(reference)
    if not parts or parts[0] not in payload:
        raise EvidencePathResolutionError(reference)
    if len(parts) == 1:
        return ResolvedEvidencePath(reference, parts[0], payload[parts[0]], "top_level")

    current: Any = payload
    traversed: list[str] = []
    resolution_kind: ResolutionKind = "dict_path"
    for part in parts:
        if isinstance(current, Mapping):
            if part not in current:
                raise EvidencePathResolutionError(reference, f"missing_component:{part}")
            current = current[part]
            traversed.append(part)
            continue
        if isinstance(current, list):
            if part.isdigit():
                index = int(part)
                if index < 0 or index >= len(current):
                    raise EvidencePathResolutionError(reference, "list_index_out_of_range")
                current = current[index]
                traversed.append(part)
                resolution_kind = "list_index"
                continue
            collection_path = ".".join(traversed)
            spec = semantic_collections.get(collection_path)
            if spec is None:
                raise EvidencePathResolutionError(
                    reference, f"list_component_requires_index:{part}"
                )
            matches = [
                item for item in current
                if isinstance(item, Mapping)
                and str(item.get(spec.identity_field, "")) == part
            ]
            if not matches:
                raise EvidencePathResolutionError(reference)
            if len(matches) > 1:
                raise AmbiguousEvidencePathError(reference, len(matches))
            current = matches[0]
            traversed.append(part)
            resolution_kind = "semantic_collection_item"
            continue
        raise EvidencePathResolutionError(reference, f"cannot_traverse_scalar:{part}")
    return ResolvedEvidencePath(reference, parts[0], current, resolution_kind)


def resolve_evidence_ref(
    evidence_pack: Mapping[str, Any] | BaseModel,
    path: str,
) -> Any:
    """Resolve one canonical EDA evidence reference to its concrete value.

    This value-returning API is the authoritative resolver used by strategy
    consistency checks. ``resolve_evidence_path`` remains available to callers
    that also need resolution metadata.
    """

    return resolve_evidence_path(path, evidence_pack).value


def generate_allowed_evidence_refs(
    evidence_pack: Mapping[str, Any] | BaseModel,
) -> list[str]:
    payload = (
        evidence_pack.model_dump(mode="json")
        if isinstance(evidence_pack, BaseModel)
        else evidence_pack
    )
    direct = [path for path, _ in _iter_dictionary_paths(payload)]
    return sorted({*direct, *generate_semantic_evidence_refs(payload)})


def generate_semantic_evidence_refs(
    evidence_pack: Mapping[str, Any],
    *,
    semantic_collections: Mapping[str, SemanticCollectionSpec] = SEMANTIC_COLLECTION_PATHS,
) -> list[str]:
    references: list[str] = []
    for collection_path, spec in semantic_collections.items():
        try:
            collection = _resolve_dictionary_path(evidence_pack, collection_path)
        except EvidencePathResolutionError:
            continue
        if not isinstance(collection, list):
            continue
        identities: dict[str, int] = {}
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            identity = item.get(spec.identity_field)
            if isinstance(identity, str) and identity.strip():
                identities[identity.strip()] = identities.get(identity.strip(), 0) + 1
        for identity, count in identities.items():
            reference = f"{collection_path}.{identity}"
            if count > 1:
                raise AmbiguousEvidencePathError(reference, count)
            references.append(reference)
    return sorted(references)


def _resolve_dictionary_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise EvidencePathResolutionError(path)
        current = current[part]
    return current


def _evidence_path_parts(reference: str) -> list[str]:
    reference = str(reference or "").strip()
    if not reference:
        raise EvidencePathResolutionError(reference, "empty_path")
    if reference.startswith(".") or reference.endswith("."):
        raise EvidencePathResolutionError(reference, "invalid_path_syntax")
    parts: list[str] = []
    cursor = 0
    while cursor < len(reference):
        if reference[cursor] == ".":
            cursor += 1
            if cursor == len(reference) or reference[cursor] in ".[":
                raise EvidencePathResolutionError(reference, "invalid_path_syntax")
        if reference[cursor] == "[":
            cursor += 1
            if cursor < len(reference) and reference[cursor] == '"':
                try:
                    value, consumed = json.JSONDecoder().raw_decode(reference[cursor:])
                except json.JSONDecodeError as exc:
                    raise EvidencePathResolutionError(
                        reference, "invalid_quoted_dictionary_key"
                    ) from exc
                if not isinstance(value, str):
                    raise EvidencePathResolutionError(reference, "invalid_dictionary_key")
                cursor += consumed
                if cursor >= len(reference) or reference[cursor] != "]":
                    raise EvidencePathResolutionError(reference, "invalid_path_syntax")
                cursor += 1
                parts.append(value)
                if cursor < len(reference) and reference[cursor] not in ".[":
                    raise EvidencePathResolutionError(reference, "invalid_path_syntax")
                continue
            end = reference.find("]", cursor)
            token = reference[cursor:end] if end >= 0 else ""
            if end < 0 or not token.isdigit():
                raise EvidencePathResolutionError(reference, "invalid_list_index")
            parts.append(token)
            cursor = end + 1
            if cursor < len(reference) and reference[cursor] not in ".[":
                raise EvidencePathResolutionError(reference, "invalid_path_syntax")
            continue
        end = cursor
        while end < len(reference) and reference[end] not in ".[":
            end += 1
        if end == cursor:
            raise EvidencePathResolutionError(reference, "invalid_path_syntax")
        parts.append(reference[cursor:end])
        cursor = end
    return parts


def _iter_dictionary_paths(value: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, child in value.items():
        key = str(key)
        safe_key = bool(key) and all(character not in ".[]" for character in key)
        if safe_key:
            path = f"{prefix}.{key}" if prefix else key
        else:
            encoded_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            path = f"{prefix}[{encoded_key}]" if prefix else f"[{encoded_key}]"
        yield path, child
        if isinstance(child, Mapping):
            yield from _iter_dictionary_paths(child, path)
        elif isinstance(child, list):
            yield from _iter_list_paths(child, path)


def _iter_list_paths(value: list[Any], prefix: str) -> Iterable[tuple[str, Any]]:
    for index, child in enumerate(value):
        path = f"{prefix}[{index}]"
        yield path, child
        if isinstance(child, Mapping):
            yield from _iter_dictionary_paths(child, path)
        elif isinstance(child, list):
            yield from _iter_list_paths(child, path)
