from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from pydantic import ConfigDict

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.reference_catalog import ReferenceCatalog


_HYPOTHESIS_ID_PATTERN = re.compile(
    r"^(?:eda_hypothesis|schema|metric|val|validation|leak|leakage|drift|feature|baseline)_\d+$"
)


class HypothesisReferenceMigrationDiagnostics(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    moved_hypothesis_refs: tuple[str, ...] = ()
    inherited_evidence_refs: tuple[str, ...] = ()
    unknown_hypothesis_refs: tuple[str, ...] = ()
    hypotheses_without_backing_evidence: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(
            self.moved_hypothesis_refs
            or self.inherited_evidence_refs
            or self.unknown_hypothesis_refs
            or self.hypotheses_without_backing_evidence
        )


def migrate_hypothesis_references(
    action: dict[str, Any],
    catalog: ReferenceCatalog,
) -> tuple[dict[str, Any], HypothesisReferenceMigrationDiagnostics]:
    """Move hypothesis IDs out of evidence_refs using catalog-owned evidence only."""

    migrated = deepcopy(action)
    evidence_refs = _unique_strings(_string_values(migrated.get("evidence_refs")))
    related = _unique_strings([
        *_string_values(migrated.get("related_hypothesis_ids")),
        *_string_values(migrated.get("hypothesis_ids")),
    ])
    retained_evidence: list[str] = []
    moved: list[str] = []
    inherited: list[str] = []
    unknown: list[str] = []
    without_backing: list[str] = []

    for reference in evidence_refs:
        resolution = catalog.resolve(reference)
        if resolution.is_resolved and resolution.entry is not None:
            entry = resolution.entry
            if entry.namespace == "hypothesis":
                _append_unique(related, entry.canonical_ref)
                _append_unique(moved, entry.canonical_ref)
                valid_backing = [
                    backing
                    for backing in entry.backing_evidence_refs
                    if catalog.is_valid_evidence_ref(backing)
                ]
                if not valid_backing:
                    _append_unique(without_backing, entry.canonical_ref)
                for backing in valid_backing:
                    _append_unique(retained_evidence, backing)
                    _append_unique(inherited, backing)
                continue
            # P0.2 intentionally leaves every non-hypothesis namespace untouched.
            _append_unique(retained_evidence, reference)
            continue

        if _looks_like_hypothesis_id(reference):
            _append_unique(unknown, reference)
            continue
        _append_unique(retained_evidence, reference)

    migrated["evidence_refs"] = retained_evidence
    migrated["related_hypothesis_ids"] = related
    # The canonical action currently exposes both names and synchronizes them.
    # Keep raw top-level/section copies equal before model validation as well.
    migrated["hypothesis_ids"] = list(related)
    diagnostics = HypothesisReferenceMigrationDiagnostics(
        moved_hypothesis_refs=tuple(moved),
        inherited_evidence_refs=tuple(inherited),
        unknown_hypothesis_refs=tuple(unknown),
        hypotheses_without_backing_evidence=tuple(without_backing),
    )
    return migrated, diagnostics


def migrate_final_strategy_hypothesis_references(
    payload: dict[str, Any],
    catalog: ReferenceCatalog,
) -> tuple[dict[str, Any], HypothesisReferenceMigrationDiagnostics]:
    """Apply hypothesis migration to top-level and section action payloads."""

    migrated = deepcopy(payload)
    located_actions = _located_action_payloads(migrated)
    groups: dict[str, list[dict[str, Any]]] = {}
    independent: list[dict[str, Any]] = []
    for action in located_actions:
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id.strip():
            groups.setdefault(action_id.strip(), []).append(action)
        else:
            independent.append(action)

    diagnostics: list[HypothesisReferenceMigrationDiagnostics] = []
    for copies in groups.values():
        merged = dict(copies[0])
        merged["evidence_refs"] = [
            value for action in copies for value in _string_values(action.get("evidence_refs"))
        ]
        merged["related_hypothesis_ids"] = [
            value
            for action in copies
            for value in _string_values(action.get("related_hypothesis_ids"))
        ]
        merged["hypothesis_ids"] = [
            value for action in copies for value in _string_values(action.get("hypothesis_ids"))
        ]
        canonical, result = migrate_hypothesis_references(merged, catalog)
        diagnostics.append(result)
        for action in copies:
            action["evidence_refs"] = list(canonical["evidence_refs"])
            action["related_hypothesis_ids"] = list(canonical["related_hypothesis_ids"])
            action["hypothesis_ids"] = list(canonical["hypothesis_ids"])

    for action in independent:
        canonical, result = migrate_hypothesis_references(action, catalog)
        diagnostics.append(result)
        action.clear()
        action.update(canonical)

    return migrated, _merge_diagnostics(diagnostics)


def _located_action_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for action in payload.get("actions") or []:
        if isinstance(action, dict):
            actions.append(action)
    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for action in section.get("actions") or []:
            if isinstance(action, dict):
                actions.append(action)
    return actions


def _merge_diagnostics(
    values: list[HypothesisReferenceMigrationDiagnostics],
) -> HypothesisReferenceMigrationDiagnostics:
    return HypothesisReferenceMigrationDiagnostics(
        moved_hypothesis_refs=tuple(_unique_strings([
            item for value in values for item in value.moved_hypothesis_refs
        ])),
        inherited_evidence_refs=tuple(_unique_strings([
            item for value in values for item in value.inherited_evidence_refs
        ])),
        unknown_hypothesis_refs=tuple(_unique_strings([
            item for value in values for item in value.unknown_hypothesis_refs
        ])),
        hypotheses_without_backing_evidence=tuple(_unique_strings([
            item for value in values for item in value.hypotheses_without_backing_evidence
        ])),
    )


def _looks_like_hypothesis_id(value: str) -> bool:
    return bool(_HYPOTHESIS_ID_PATTERN.fullmatch(value))


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
    "HypothesisReferenceMigrationDiagnostics",
    "migrate_final_strategy_hypothesis_references",
    "migrate_hypothesis_references",
]
