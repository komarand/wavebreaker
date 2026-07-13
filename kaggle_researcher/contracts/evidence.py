from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel


EvidenceNamespace = Literal[
    "eda_evidence",
    "reasoning_evidence",
    "source_claim",
    "hypothesis",
    "synthetic_inference",
]


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
    for path, value in _iter_documented_paths(eda_payload):
        registry.register("eda_evidence", path, value)
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


def _iter_documented_paths(value: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        yield path, child
        if isinstance(child, Mapping):
            yield from _iter_documented_paths(child, path)

