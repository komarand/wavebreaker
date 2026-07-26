from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel, Field

from kaggle_researcher.contracts.migration import migrate_eda_task_plan_payload
from kaggle_researcher.contracts.repair import validate_with_one_repair


pytestmark = pytest.mark.contract


class _LlmPayload(BaseModel):
    name: str = Field(min_length=1)


@pytest.mark.asyncio
async def test_llm_contract_gets_one_bounded_repair_with_structured_issues() -> None:
    calls: list[dict[str, object]] = []

    def repair(payload: dict[str, object]) -> dict[str, str]:
        calls.append(payload)
        return {"name": "repaired"}

    result = await validate_with_one_repair(
        {"name": ""}, model=_LlmPayload, repair=repair, contract_name="llm_payload"
    )
    assert result.repaired is True
    assert result.value.name == "repaired"
    assert len(calls) == 1
    assert calls[0]["validation_issues"]


@pytest.mark.asyncio
async def test_llm_contract_never_retries_more_than_once() -> None:
    calls = 0

    def repair(_: dict[str, object]) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"name": ""}

    with pytest.raises(Exception, match="remained invalid after one bounded repair"):
        await validate_with_one_repair(
            {"name": ""}, model=_LlmPayload, repair=repair, contract_name="llm_payload"
        )
    assert calls == 1


def test_internal_migration_runs_only_when_explicitly_selected() -> None:
    migrated = migrate_eda_task_plan_payload({"competition_id": "demo", "eda_tasks": []})
    assert migrated.migrated is True
    assert "schema_version:missing->1.0" in migrated.applied_migrations


def test_internal_producer_schema_failure_is_not_sent_to_repair() -> None:
    ingest = importlib.import_module("kaggle_researcher.contracts.ingest")
    repair_calls = 0

    def forbidden_repair(_: object) -> object:
        nonlocal repair_calls
        repair_calls += 1
        return {"name": "silently repaired"}

    with pytest.raises(Exception) as raised:
        ingest.ingest_internal_contract(
            {"name": ""}, model=_LlmPayload, repair=forbidden_repair, allow_migration=False
        )
    assert repair_calls == 0
    assert raised.value.classification == "internal_contract_error"
