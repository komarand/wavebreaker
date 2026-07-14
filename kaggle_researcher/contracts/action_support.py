from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.final_strategy_compilation import (
    FinalStrategyCompilationDiagnostics,
    FinalStrategyCompilationError,
)
from kaggle_researcher.contracts.reference_catalog import ReferenceCatalog


ActionSupportOutcome = Literal["keep", "downgrade", "drop", "fail"]
ACTION_SUPPORT_PHASE = "action_support_gate"
_RESEARCH_ACTION_PATTERN = re.compile(
    r"\b(?:test|check|investigate|validate|measure|monitor)\b",
    re.IGNORECASE,
)
_COMPOSITE_NAMESPACES = {
    "risk", "validation_requirement", "safety_constraint", "source_claim",
}


class FrozenSupportModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ActionReferenceResolutionDiagnostics(FrozenSupportModel):
    original_refs: tuple[str, ...] = ()
    unresolved_refs: tuple[str, ...] = ()


class FinalStrategyCompilationContext(FrozenSupportModel):
    reference_catalog: ReferenceCatalog
    phase: str = ACTION_SUPPORT_PHASE

    def uncertainty_evidence_refs(self, hypothesis_ids: list[str]) -> tuple[str, ...]:
        refs: list[str] = []
        for hypothesis_id in hypothesis_ids:
            resolution = self.reference_catalog.resolve(
                hypothesis_id,
                expected_namespace="hypothesis",
            )
            if not resolution.is_resolved or resolution.entry is None:
                continue
            for reference in resolution.entry.uncertainty_evidence_refs:
                if self.reference_catalog.is_valid_evidence_ref(reference):
                    _append_unique(refs, reference)
        return tuple(refs)


class ActionSupportDecision(FrozenSupportModel):
    decision: ActionSupportOutcome
    action_id: str
    priority: str
    reason: str
    phase: str = ACTION_SUPPORT_PHASE
    original_refs: tuple[str, ...] = ()
    unresolved_refs: tuple[str, ...] = ()
    resulting_evidence_refs: tuple[str, ...] = ()
    limitation: str | None = None


class FinalStrategyCompilationReport(FrozenSupportModel):
    kept_actions: tuple[ActionSupportDecision, ...] = ()
    downgraded_actions: tuple[ActionSupportDecision, ...] = ()
    dropped_actions: tuple[ActionSupportDecision, ...] = ()
    failed_actions: tuple[ActionSupportDecision, ...] = ()


class UnsupportedFinalStrategyActionError(FinalStrategyCompilationError):
    def __init__(
        self,
        decision: ActionSupportDecision,
        *,
        compilation_report: FinalStrategyCompilationReport,
    ) -> None:
        self.action_id = decision.action_id
        self.priority = decision.priority
        self.original_refs = decision.original_refs
        self.unresolved_refs = decision.unresolved_refs
        self.decision_reason = decision.reason
        self.compilation_report = compilation_report
        diagnostics = FinalStrategyCompilationDiagnostics(
            phase="action_support_gate",
            kept_actions=len(compilation_report.kept_actions),
            downgraded_actions=len(compilation_report.downgraded_actions),
            dropped_actions=len(compilation_report.dropped_actions),
        )
        super().__init__(
            f"Unsupported Final Strategy action {self.action_id!r} at {decision.phase}: "
            f"priority={self.priority}; {self.decision_reason}; "
            f"original_refs={list(self.original_refs)}; "
            f"unresolved_refs={list(self.unresolved_refs)}.",
            phase="action_support_gate",
            diagnostics=diagnostics,
        )


def enforce_action_evidence_support(
    action: dict[str, Any],
    diagnostics: ActionReferenceResolutionDiagnostics,
    context: FinalStrategyCompilationContext,
) -> ActionSupportDecision:
    """Choose a deterministic support decision without creating evidence."""

    action_id = str(action.get("action_id") or "anonymous_action")
    priority = str(action.get("priority") or "P2").upper()
    factual_refs = _unique_strings([
        reference
        for reference in _string_values(action.get("evidence_refs"))
        if context.reference_catalog.is_valid_evidence_ref(reference)
    ])
    if factual_refs:
        return ActionSupportDecision(
            decision="keep",
            action_id=action_id,
            priority=priority,
            reason="At least one catalog-validated factual evidence reference remains.",
            original_refs=diagnostics.original_refs,
            unresolved_refs=diagnostics.unresolved_refs,
            resulting_evidence_refs=tuple(factual_refs),
        )

    related = _unique_strings([
        *_string_values(action.get("related_hypothesis_ids")),
        *_string_values(action.get("hypothesis_ids")),
    ])
    limitations = _string_values(action.get("limitations"))
    research_wording = bool(_RESEARCH_ACTION_PATTERN.search(str(action.get("action") or "")))
    has_uncertainty_signal = bool(related or diagnostics.unresolved_refs or limitations)
    uncertainty_refs = context.uncertainty_evidence_refs(related)
    if research_wording and has_uncertainty_signal and uncertainty_refs:
        limitation = (
            "Downgraded to an evidence-backed investigation because the proposed action "
            "has no direct factual support."
        )
        return ActionSupportDecision(
            decision="downgrade",
            action_id=action_id,
            priority=priority,
            reason=(
                "Research wording and a factual unresolved-hypothesis result permit a "
                "safe investigative action."
            ),
            original_refs=diagnostics.original_refs,
            unresolved_refs=diagnostics.unresolved_refs,
            resulting_evidence_refs=uncertainty_refs,
            limitation=limitation,
        )

    if priority == "P0":
        return ActionSupportDecision(
            decision="fail",
            action_id=action_id,
            priority=priority,
            reason=(
                "Mandatory P0 strategy action has no factual evidence and cannot be "
                "safely downgraded."
            ),
            original_refs=diagnostics.original_refs,
            unresolved_refs=diagnostics.unresolved_refs,
        )

    return ActionSupportDecision(
        decision="drop",
        action_id=action_id,
        priority=priority,
        reason=(
            "Optional action has no factual evidence and no catalog-backed uncertainty "
            "path for safe downgrade."
        ),
        original_refs=diagnostics.original_refs,
        unresolved_refs=diagnostics.unresolved_refs,
    )


def compile_final_strategy_action_support(
    payload: dict[str, Any],
    *,
    original_payload: dict[str, Any],
    context: FinalStrategyCompilationContext,
) -> tuple[dict[str, Any], FinalStrategyCompilationReport]:
    """Gate each canonical top-level action once before strict validation."""

    compiled = deepcopy(payload)
    resolved_locations = _located_actions(compiled)
    original_locations = _located_actions(original_payload)
    resolved_groups = _group_locations(resolved_locations)
    original_groups = _group_locations(original_locations)
    kept: list[ActionSupportDecision] = []
    downgraded: list[ActionSupportDecision] = []
    dropped: list[ActionSupportDecision] = []
    failed: list[ActionSupportDecision] = []
    drop_object_ids: set[int] = set()

    for key, locations in resolved_groups.items():
        merged = _merge_action_copies([location.action for location in locations])
        original_actions = [
            location.action for location in original_groups.get(key, ())
        ]
        original_refs = _unique_strings([
            reference
            for action in original_actions
            for reference in _string_values(action.get("evidence_refs"))
        ])
        unresolved_refs = _unresolved_original_refs(
            original_refs,
            context.reference_catalog,
        )
        decision = enforce_action_evidence_support(
            merged,
            ActionReferenceResolutionDiagnostics(
                original_refs=tuple(original_refs),
                unresolved_refs=tuple(unresolved_refs),
            ),
            context,
        )
        if decision.decision == "keep":
            kept.append(decision)
            _apply_decision(locations, decision)
        elif decision.decision == "downgrade":
            downgraded.append(decision)
            _apply_decision(locations, decision)
        elif decision.decision == "drop":
            dropped.append(decision)
            drop_object_ids.update(id(location.action) for location in locations)
        else:
            failed.append(decision)
            report = FinalStrategyCompilationReport(
                kept_actions=tuple(kept),
                downgraded_actions=tuple(downgraded),
                dropped_actions=tuple(dropped),
                failed_actions=tuple(failed),
            )
            raise UnsupportedFinalStrategyActionError(
                decision,
                compilation_report=report,
            )

    for location in resolved_locations:
        location.container[:] = [
            action for action in location.container if id(action) not in drop_object_ids
        ]
    dropped_ids = {decision.action_id for decision in dropped}
    for section in compiled.get("sections") or []:
        if isinstance(section, dict):
            section["action_ids"] = [
                action_id
                for action_id in _string_values(section.get("action_ids"))
                if action_id not in dropped_ids
            ]
    compiled["sections"] = [
        section
        for section in compiled.get("sections") or []
        if not isinstance(section, dict)
        or section.get("action_ids")
        or section.get("evidence_refs")
    ]

    for location in _located_actions(compiled):
        if not _string_values(location.action.get("evidence_refs")):
            decision = ActionSupportDecision(
                decision="fail",
                action_id=str(location.action.get("action_id") or "anonymous_action"),
                priority=str(location.action.get("priority") or "P2"),
                reason="Action support gate produced an empty evidence_refs list.",
                original_refs=(),
                unresolved_refs=(),
            )
            failed.append(decision)
            report = FinalStrategyCompilationReport(
                kept_actions=tuple(kept), downgraded_actions=tuple(downgraded),
                dropped_actions=tuple(dropped), failed_actions=tuple(failed),
            )
            raise UnsupportedFinalStrategyActionError(decision, compilation_report=report)

    return compiled, FinalStrategyCompilationReport(
        kept_actions=tuple(kept),
        downgraded_actions=tuple(downgraded),
        dropped_actions=tuple(dropped),
        failed_actions=tuple(failed),
    )


@dataclass
class _LocatedAction:
    key: str
    action: dict[str, Any]
    container: list[Any]


def _located_actions(payload: dict[str, Any]) -> list[_LocatedAction]:
    located: list[_LocatedAction] = []
    top = payload.get("actions") or []
    if isinstance(top, list):
        for index, action in enumerate(top):
            if isinstance(action, dict):
                located.append(_LocatedAction(_action_key(action, f"actions[{index}]"), action, top))
    return located


def _action_key(action: dict[str, Any], fallback: str) -> str:
    action_id = action.get("action_id")
    return f"id:{action_id.strip()}" if isinstance(action_id, str) and action_id.strip() else fallback


def _group_locations(values: list[_LocatedAction]) -> dict[str, list[_LocatedAction]]:
    result: dict[str, list[_LocatedAction]] = defaultdict(list)
    for value in values:
        result[value.key].append(value)
    return result


def _merge_action_copies(actions: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(actions[0])
    merged["evidence_refs"] = _unique_strings([
        ref for action in actions for ref in _string_values(action.get("evidence_refs"))
    ])
    merged["related_hypothesis_ids"] = _unique_strings([
        ref for action in actions for ref in _string_values(action.get("related_hypothesis_ids"))
    ])
    merged["hypothesis_ids"] = _unique_strings([
        ref for action in actions for ref in _string_values(action.get("hypothesis_ids"))
    ])
    merged["limitations"] = _unique_strings([
        value for action in actions for value in _string_values(action.get("limitations"))
    ])
    if any(str(action.get("priority") or "").upper() == "P0" for action in actions):
        merged["priority"] = "P0"
    return merged


def _apply_decision(
    locations: list[_LocatedAction],
    decision: ActionSupportDecision,
) -> None:
    for location in locations:
        location.action["evidence_refs"] = list(decision.resulting_evidence_refs)
        if decision.decision == "downgrade":
            location.action["confidence"] = "low"
            location.action["evidence_origin"] = "Hypothesis-to-test"
            limitations = _unique_strings(_string_values(location.action.get("limitations")))
            if decision.limitation:
                _append_unique(limitations, decision.limitation)
            location.action["limitations"] = limitations


def _unresolved_original_refs(
    refs: list[str],
    catalog: ReferenceCatalog,
) -> list[str]:
    unresolved: list[str] = []
    for reference in refs:
        if catalog.is_valid_evidence_ref(reference):
            continue
        resolution = catalog.resolve(reference)
        if not resolution.is_resolved or resolution.entry is None:
            _append_unique(unresolved, reference)
            continue
        entry = resolution.entry
        if entry.support_kind == "policy":
            _append_unique(unresolved, reference)
            continue
        if entry.namespace == "hypothesis":
            valid = any(catalog.is_valid_evidence_ref(ref) for ref in entry.backing_evidence_refs)
        elif entry.namespace in _COMPOSITE_NAMESPACES:
            valid = any(catalog.is_valid_evidence_ref(ref) for ref in entry.backing_evidence_refs)
        else:
            valid = False
        if not valid:
            _append_unique(unresolved, reference)
    return unresolved


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
    "ACTION_SUPPORT_PHASE",
    "ActionReferenceResolutionDiagnostics",
    "ActionSupportDecision",
    "FinalStrategyCompilationContext",
    "FinalStrategyCompilationReport",
    "UnsupportedFinalStrategyActionError",
    "compile_final_strategy_action_support",
    "enforce_action_evidence_support",
]
