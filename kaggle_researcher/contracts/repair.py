from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Iterable, TypeVar

from pydantic import BaseModel, ValidationError

from kaggle_researcher.contracts.errors import BoundaryRepairError, ContractIssue
from kaggle_researcher.contracts.normalization import normalize_contract_payload


T = TypeVar("T", bound=BaseModel)
RepairCallable = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class BoundaryValidationResult(Generic[T]):
    value: T
    repaired: bool
    initial_issues: tuple[ContractIssue, ...] = ()


async def validate_with_one_repair(
    payload: Any,
    *,
    model: type[T],
    repair: RepairCallable,
    contract_name: str,
    allowed_ids: Iterable[str] = (),
    allowed_enums: dict[str, Iterable[str]] | None = None,
) -> BoundaryValidationResult[T]:
    """Normalize, validate, make exactly one bounded repair, then fail hard."""
    normalized = normalize_contract_payload(payload, model.__name__)
    try:
        return BoundaryValidationResult(model.model_validate(normalized), False)
    except ValidationError as initial_error:
        initial_issues = _issues(initial_error)
    repair_input = {
        "contract": contract_name,
        "validation_issues": [issue.as_dict() for issue in initial_issues[:8]],
        "canonical_fields": sorted(model.model_fields),
        "allowed_ids": list(dict.fromkeys(allowed_ids)),
        "allowed_enums": {key: list(values) for key, values in (allowed_enums or {}).items()},
        "original_response": payload,
    }
    repaired = repair(repair_input)
    if inspect.isawaitable(repaired):
        repaired = await repaired
    repaired = normalize_contract_payload(repaired, model.__name__)
    try:
        return BoundaryValidationResult(model.model_validate(repaired), True, initial_issues)
    except ValidationError as final_error:
        raise BoundaryRepairError(
            f"{contract_name} remained invalid after one bounded repair",
            issues=_issues(final_error),
            contract=contract_name,
        ) from final_error


def _issues(error: ValidationError) -> tuple[ContractIssue, ...]:
    return tuple(
        ContractIssue(
            ".".join(str(part) for part in item.get("loc", ())),
            item.get("input"),
            item.get("type", "valid canonical field"),
            item.get("msg", "validation failed"),
        )
        for item in error.errors(include_url=False)
    )


__all__ = ["BoundaryValidationResult", "validate_with_one_repair"]

