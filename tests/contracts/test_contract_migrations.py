from __future__ import annotations

from copy import deepcopy
from typing import Literal

import pytest
from pydantic import Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.errors import ContractMigrationError
from kaggle_researcher.contracts.ingest import (
    MigrationPolicy,
    RepairPolicy,
    ingest_contract,
)
from kaggle_researcher.contracts.migration import (
    ContractMigration,
    ContractMigrationGraph,
)
from kaggle_researcher.contracts.registry import ContractRegistry


pytestmark = pytest.mark.contract


def test_external_legacy_artifact_uses_explicit_migration_path() -> None:
    raw = {
        "competition_id": "migration-demo",
        "hypotheses": [{
            "id": "hyp-1",
            "category": "validation",
            "claim": "Use stable folds.",
            "confidence": "high",
        }],
    }
    original = deepcopy(raw)

    result = ingest_contract(
        raw,
        expected_family="research_hypotheses",
        source_kind="external_artifact",
        migration_policy=MigrationPolicy.ALLOW,
        repair_policy=RepairPolicy.NORMALIZE,
    )

    assert raw == original
    assert result.original_version == "unversioned"
    assert result.final_version == "1.0"
    assert result.migrations_applied[0] == "research_hypotheses.legacy_unversioned_to_1_0"
    assert any("id->hypothesis_id" in item for item in result.migrations_applied)


def test_migration_graph_rejects_cycles() -> None:
    graph = ContractMigrationGraph()
    graph.register(ContractMigration(
        "example", "1.0", "2.0", lambda value: value, "example.1_to_2"
    ))
    with pytest.raises(ContractMigrationError, match="cycle"):
        graph.register(ContractMigration(
            "example", "2.0", "1.0", lambda value: value,
            "example.2_to_1", allow_downgrade=True,
        ))


class MigratedContract(ContractModel):
    contract_family: Literal["migrated"] = "migrated"
    schema_version: Literal["2.0"] = "2.0"
    required_value: str = Field(min_length=1)


def test_migration_output_must_validate_against_target_schema() -> None:
    registry = ContractRegistry()
    registry.register("migrated", "2.0", MigratedContract, current=True)
    graph = ContractMigrationGraph()
    graph.register(ContractMigration(
        "migrated",
        "unversioned",
        "2.0",
        lambda _: {"contract_family": "migrated", "schema_version": "2.0"},
        "migrated.invalid_target",
    ))

    with pytest.raises(ContractMigrationError, match="failed target schema"):
        ingest_contract(
            {"legacy": True},
            expected_family="migrated",
            source_kind="external_artifact",
            migration_policy=MigrationPolicy.REQUIRE_CURRENT,
            repair_policy=RepairPolicy.FORBID,
            registry=registry,
            migration_graph=graph,
        )


def test_migration_path_selection_is_deterministic() -> None:
    graph = ContractMigrationGraph()

    def step(target: str):
        return lambda payload: {
            **payload,
            "contract_family": "example",
            "schema_version": target,
        }

    graph.register(ContractMigration("example", "1.0", "1.5", step("1.5"), "a"))
    graph.register(ContractMigration("example", "1.0", "1.6", step("1.6"), "b"))
    graph.register(ContractMigration("example", "1.5", "2.0", step("2.0"), "c"))
    graph.register(ContractMigration("example", "1.6", "2.0", step("2.0"), "d"))

    assert [item.migration_id for item in graph.find_path("example", "1.0", "2.0")] == [
        "a", "c",
    ]
