from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict

from kaggle_researcher.contracts.base import ContractModel


FinalStrategyCompilationPhase = Literal[
    "draft_parsing",
    "reference_catalog_build",
    "reference_resolution",
    "action_support_gate",
    "post_resolution_schema_validation",
    "final_quality_gate",
]


class SanitizedSchemaValidationError(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location: tuple[str | int, ...] = ()
    error_type: str
    message: str
    action_id: str | None = None


class FinalStrategyCompilationDiagnostics(ContractModel):
    """Bounded, response-free diagnostics for deterministic strategy compilation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: FinalStrategyCompilationPhase
    initial_reference_issues: int = 0
    resolved_references: int = 0
    unresolved_references: int = 0
    kept_actions: int = 0
    downgraded_actions: int = 0
    dropped_actions: int = 0
    schema_validation_errors: tuple[SanitizedSchemaValidationError, ...] = ()


class FinalStrategyCompilationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: FinalStrategyCompilationPhase,
        diagnostics: FinalStrategyCompilationDiagnostics | None = None,
    ) -> None:
        self.phase = phase
        self.diagnostics = diagnostics or FinalStrategyCompilationDiagnostics(phase=phase)
        super().__init__(message)


class UnresolvedFinalStrategyReferenceError(FinalStrategyCompilationError):
    pass


class FinalStrategyRepairError(FinalStrategyCompilationError):
    pass


class FinalStrategySchemaValidationError(FinalStrategyCompilationError):
    def __init__(
        self,
        *,
        errors: list[dict[str, Any]],
        payload: dict[str, Any],
        diagnostics: FinalStrategyCompilationDiagnostics,
    ) -> None:
        sanitized = tuple(_sanitize_schema_error(error, payload) for error in errors[:50])
        diagnostics = diagnostics.model_copy(update={
            "phase": "post_resolution_schema_validation",
            "schema_validation_errors": sanitized,
        })
        action_ids = tuple(dict.fromkeys(item.action_id for item in sanitized if item.action_id))
        self.action_ids = action_ids
        self.initial_reference_issue_count = diagnostics.initial_reference_issues
        self.resolved_reference_count = diagnostics.resolved_references
        self.unresolved_reference_count = diagnostics.unresolved_references
        self.dropped_action_count = diagnostics.dropped_actions
        reason = (
            f"{len(action_ids)} canonical actions do not satisfy the strict schema."
            if action_ids
            else f"{len(sanitized)} strict schema violations remain."
        )
        super().__init__(
            "Final strategy failed during post_resolution_schema_validation: " + reason,
            phase="post_resolution_schema_validation",
            diagnostics=diagnostics,
        )


def _sanitize_schema_error(
    error: dict[str, Any], payload: dict[str, Any]
) -> SanitizedSchemaValidationError:
    location = tuple(
        part if isinstance(part, (str, int)) else str(part)
        for part in error.get("loc", ())
    )
    message = " ".join(str(error.get("msg", "Schema validation failed.")).split())
    return SanitizedSchemaValidationError(
        location=location,
        error_type=str(error.get("type", "validation_error"))[:120],
        message=message[:300],
        action_id=_action_id_at_location(payload, location),
    )


def _action_id_at_location(
    payload: dict[str, Any], location: tuple[str | int, ...]
) -> str | None:
    try:
        if len(location) >= 2 and location[0] == "actions":
            action = (payload.get("actions") or [])[int(location[1])]
            return str(action.get("action_id")) if isinstance(action, dict) and action.get("action_id") else None
        if len(location) >= 4 and location[0] == "sections" and location[2] == "actions":
            section = (payload.get("sections") or [])[int(location[1])]
            action = (section.get("actions") or [])[int(location[3])]
            return str(action.get("action_id")) if isinstance(action, dict) and action.get("action_id") else None
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return None
