from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from pydantic import ConfigDict

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.reference_catalog import ReferenceCatalog


_COMPOSITE_NAMESPACES = {
    "risk",
    "validation_requirement",
    "safety_constraint",
    "source_claim",
}
_COMPOSITE_ID_PATTERN = re.compile(
    r"^(?:risk_.+|validation_requirement_.+|validation_requirements\..+|"
    r"safety_.+|safety_constraints\..+|claim_\d+)$"
)


class CompositeReferenceResolutionDiagnostics(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    resolved_composite_refs: tuple[str, ...] = ()
    inherited_backing_evidence_refs: tuple[str, ...] = ()
    composite_refs_without_evidence: tuple[str, ...] = ()
    unknown_composite_refs: tuple[str, ...] = ()
    policy_only_refs: tuple[str, ...] = ()
    broken_backing_evidence_refs: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(
            self.resolved_composite_refs
            or self.inherited_backing_evidence_refs
            or self.composite_refs_without_evidence
            or self.unknown_composite_refs
            or self.policy_only_refs
            or self.broken_backing_evidence_refs
        )


def resolve_composite_action_references(
    action: dict[str, Any],
    catalog: ReferenceCatalog,
) -> tuple[dict[str, Any], CompositeReferenceResolutionDiagnostics]:
    """Replace composite references with catalog-validated factual evidence."""

    resolved_action = deepcopy(action)
    evidence_refs = _unique_strings(_string_values(resolved_action.get("evidence_refs")))
    retained: list[str] = []
    resolved_refs: list[str] = []
    inherited: list[str] = []
    without_evidence: list[str] = []
    unknown: list[str] = []
    policy_only: list[str] = []
    broken: list[str] = []

    for reference in evidence_refs:
        resolution = catalog.resolve(reference)
        if resolution.is_resolved and resolution.entry is not None:
            entry = resolution.entry
            if entry.namespace not in _COMPOSITE_NAMESPACES:
                _append_unique(retained, reference)
                continue
            # Retrieved source IDs are already direct factual evidence. Validated
            # source-claim objects have backing refs and are resolved below.
            if (
                entry.namespace == "source_claim"
                and not entry.backing_evidence_refs
                and catalog.is_valid_evidence_ref(reference)
            ):
                _append_unique(retained, reference)
                continue

            _append_unique(resolved_refs, reference)
            if entry.support_kind == "policy":
                _append_unique(policy_only, reference)
                continue

            valid_backing: list[str] = []
            for backing in entry.backing_evidence_refs:
                if catalog.is_valid_evidence_ref(backing):
                    _append_unique(valid_backing, backing)
                else:
                    _append_unique(broken, backing)
            if not valid_backing:
                _append_unique(without_evidence, reference)
                continue
            for backing in valid_backing:
                _append_unique(retained, backing)
                _append_unique(inherited, backing)
            continue

        if _looks_like_composite_id(reference):
            _append_unique(unknown, reference)
            continue
        _append_unique(retained, reference)

    resolved_action["evidence_refs"] = retained
    diagnostics = CompositeReferenceResolutionDiagnostics(
        resolved_composite_refs=tuple(resolved_refs),
        inherited_backing_evidence_refs=tuple(inherited),
        composite_refs_without_evidence=tuple(without_evidence),
        unknown_composite_refs=tuple(unknown),
        policy_only_refs=tuple(policy_only),
        broken_backing_evidence_refs=tuple(broken),
    )
    return resolved_action, diagnostics


def resolve_final_strategy_composite_references(
    payload: dict[str, Any],
    catalog: ReferenceCatalog,
) -> tuple[dict[str, Any], CompositeReferenceResolutionDiagnostics]:
    """Resolve composite refs in top-level and section action payloads."""

    resolved_payload = deepcopy(payload)
    actions = _located_action_payloads(resolved_payload)
    groups: dict[str, list[dict[str, Any]]] = {}
    independent: list[dict[str, Any]] = []
    for action in actions:
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id.strip():
            groups.setdefault(action_id.strip(), []).append(action)
        else:
            independent.append(action)

    diagnostics: list[CompositeReferenceResolutionDiagnostics] = []
    for copies in groups.values():
        merged = dict(copies[0])
        merged["evidence_refs"] = [
            value for action in copies for value in _string_values(action.get("evidence_refs"))
        ]
        canonical, result = resolve_composite_action_references(merged, catalog)
        diagnostics.append(result)
        for action in copies:
            action["evidence_refs"] = list(canonical["evidence_refs"])

    for action in independent:
        canonical, result = resolve_composite_action_references(action, catalog)
        diagnostics.append(result)
        action.clear()
        action.update(canonical)

    return resolved_payload, _merge_diagnostics(diagnostics)


def _located_action_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        action for action in payload.get("actions") or [] if isinstance(action, dict)
    ]
    for section in payload.get("sections") or []:
        if isinstance(section, dict):
            actions.extend(
                action
                for action in section.get("actions") or []
                if isinstance(action, dict)
            )
    return actions


def _merge_diagnostics(
    values: list[CompositeReferenceResolutionDiagnostics],
) -> CompositeReferenceResolutionDiagnostics:
    fields = (
        "resolved_composite_refs",
        "inherited_backing_evidence_refs",
        "composite_refs_without_evidence",
        "unknown_composite_refs",
        "policy_only_refs",
        "broken_backing_evidence_refs",
    )
    return CompositeReferenceResolutionDiagnostics(**{
        field: tuple(_unique_strings([
            item for value in values for item in getattr(value, field)
        ]))
        for field in fields
    })


def _looks_like_composite_id(value: str) -> bool:
    return bool(_COMPOSITE_ID_PATTERN.fullmatch(value))


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_unique(result, value)
    return result


__all__ = [
    "CompositeReferenceResolutionDiagnostics",
    "resolve_composite_action_references",
    "resolve_final_strategy_composite_references",
]
