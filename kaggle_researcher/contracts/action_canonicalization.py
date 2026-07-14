from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping

from pydantic import ConfigDict

from kaggle_researcher.contracts.base import ContractModel


_MERGED_LIST_FIELDS = (
    "evidence_refs",
    "related_hypothesis_ids",
    "hypothesis_ids",
    "limitations",
    "eda_result_refs",
    "experiment_ids",
    "source_refs",
    "risk_ids",
    "validation_requirement_ids",
    "safety_constraint_ids",
)
_IDENTITY_FIELDS = ("action", "reason", "priority")


class SectionActionMembership(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str
    action_ids: tuple[str, ...] = ()


class ActionCanonicalizationDiagnostics(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_action_ids: tuple[str, ...] = ()
    merged_duplicate_actions: tuple[str, ...] = ()
    conflicting_action_definitions: tuple[str, ...] = ()
    dangling_action_ids: tuple[str, ...] = ()
    section_memberships: tuple[SectionActionMembership, ...] = ()


class FinalStrategyActionCanonicalizationError(RuntimeError):
    phase = "action_canonicalization"

    def __init__(self, diagnostics: ActionCanonicalizationDiagnostics) -> None:
        self.diagnostics = diagnostics
        parts: list[str] = []
        if diagnostics.conflicting_action_definitions:
            parts.append(
                f"{len(diagnostics.conflicting_action_definitions)} conflicting definitions"
            )
        if diagnostics.dangling_action_ids:
            parts.append(f"{len(diagnostics.dangling_action_ids)} dangling action IDs")
        super().__init__(
            "Final strategy action canonicalization failed: " + ", ".join(parts)
        )


def canonicalize_final_strategy_actions(
    raw_result: Mapping[str, Any],
) -> tuple[dict[str, Any], ActionCanonicalizationDiagnostics]:
    """Build one ordered action list and section membership references."""

    result = deepcopy(dict(raw_result))
    canonical: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    generated: list[str] = []
    merged: list[str] = []
    conflicts: list[str] = []
    dangling: list[str] = []

    def register(candidate: Any, source_path: str) -> str | None:
        candidate_mapping = _mapping_value(candidate)
        if candidate_mapping is None:
            return None
        action = deepcopy(dict(candidate_mapping))
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            action_id = _deterministic_action_id(action)
            action["action_id"] = action_id
            _append_unique(generated, action_id)
        existing = canonical.get(action_id)
        if existing is None:
            canonical[action_id] = action
            ordered_ids.append(action_id)
            return action_id
        definition_conflicts = [
            field
            for field in _IDENTITY_FIELDS
            if _has_conflict(existing.get(field), action.get(field))
        ]
        if definition_conflicts:
            conflicts.append(
                f"{action_id} at {source_path}: {', '.join(definition_conflicts)}"
            )
            return action_id
        for field in _MERGED_LIST_FIELDS:
            existing[field] = _unique_values([
                *_list_values(existing.get(field)),
                *_list_values(action.get(field)),
            ])
        for field, value in action.items():
            if field not in existing or existing[field] in (None, "", [], {}):
                existing[field] = deepcopy(value)
        _append_unique(merged, action_id)
        return action_id

    for index, action in enumerate(_list_values(result.get("actions"))):
        register(action, f"actions[{index}]")

    memberships: list[SectionActionMembership] = []
    sections: list[dict[str, Any]] = []
    for section_index, raw_section in enumerate(_list_values(result.get("sections"))):
        section_mapping = _mapping_value(raw_section)
        if section_mapping is None:
            continue
        section = deepcopy(dict(section_mapping))
        section_id = str(section.get("section_id") or f"section_{section_index + 1}")
        member_ids: list[str] = []
        for action_index, action in enumerate(_list_values(section.pop("actions", None))):
            action_id = register(
                action,
                f"sections[{section_index}].actions[{action_index}]",
            )
            if action_id:
                _append_unique(member_ids, action_id)
        for action_id_value in _list_values(section.get("action_ids")):
            action_id = str(action_id_value).strip()
            if action_id:
                _append_unique(member_ids, action_id)
        section["action_ids"] = member_ids
        sections.append(section)
        memberships.append(SectionActionMembership(
            section_id=section_id,
            action_ids=tuple(member_ids),
        ))

    for membership in memberships:
        for action_id in membership.action_ids:
            if action_id not in canonical:
                _append_unique(dangling, action_id)

    result["actions"] = [canonical[action_id] for action_id in ordered_ids]
    result["sections"] = sections
    diagnostics = ActionCanonicalizationDiagnostics(
        generated_action_ids=tuple(generated),
        merged_duplicate_actions=tuple(merged),
        conflicting_action_definitions=tuple(conflicts),
        dangling_action_ids=tuple(dangling),
        section_memberships=tuple(memberships),
    )
    if conflicts or dangling:
        raise FinalStrategyActionCanonicalizationError(diagnostics)
    return result, diagnostics


def _deterministic_action_id(action: Mapping[str, Any]) -> str:
    stable = {
        "priority": _normalized_text(action.get("priority")),
        "action": _normalized_text(action.get("action")),
        "reason": _normalized_text(action.get("reason")),
    }
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"action_{digest}"


def _has_conflict(left: Any, right: Any) -> bool:
    if left in (None, "") or right in (None, ""):
        return False
    return _normalized_text(left) != _normalized_text(right)


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _list_values(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _mapping_value(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _unique_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


__all__ = [
    "ActionCanonicalizationDiagnostics",
    "FinalStrategyActionCanonicalizationError",
    "SectionActionMembership",
    "canonicalize_final_strategy_actions",
]
