from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field

from kaggle_researcher.contracts.artifacts import (
    load_published_eda_evidence_bundle,
    write_json_atomic,
)
from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.errors import (
    BoundaryRepairError,
    InternalContractValidationError,
    UnknownContractFamilyError,
    UnsupportedSchemaVersionError,
)
from kaggle_researcher.contracts.evidence_manifest import publish_eda_evidence_bundle
from kaggle_researcher.contracts.ingest import (
    MigrationPolicy,
    RepairPolicy,
    ingest_contract,
)
from kaggle_researcher.contracts.migration import ContractMigrationGraph
from kaggle_researcher.contracts.registry import ContractRegistry
from tests.fixtures.evidence_contract import representative_evidence_pack


pytestmark = pytest.mark.contract


class LlmDraftContract(ContractModel):
    contract_family: Literal["llm_draft"] = "llm_draft"
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


def _llm_registry() -> ContractRegistry:
    registry = ContractRegistry()
    registry.register("llm_draft", "1.0", LlmDraftContract, current=True)
    return registry


def test_known_family_version_dispatch_and_hashes_are_deterministic() -> None:
    payload = representative_evidence_pack().model_dump(mode="json")
    first = ingest_contract(
        payload,
        expected_family="eda_evidence_pack",
        source_kind="internal_deterministic",
        migration_policy=MigrationPolicy.FORBID,
        repair_policy=RepairPolicy.FORBID,
    )
    second = ingest_contract(
        payload,
        expected_family="eda_evidence_pack",
        source_kind="internal_deterministic",
        migration_policy=MigrationPolicy.FORBID,
        repair_policy=RepairPolicy.FORBID,
    )

    assert first.final_family == "eda_evidence_pack"
    assert first.final_version == "1.0"
    assert len(first.input_hash) == len(first.output_hash) == 64
    assert first == second


def test_unknown_family_is_typed_error() -> None:
    with pytest.raises(UnknownContractFamilyError):
        ingest_contract(
            {"contract_family": "unknown", "schema_version": "1.0"},
            expected_family=None,
            source_kind="external_artifact",
            migration_policy=MigrationPolicy.FORBID,
            repair_policy=RepairPolicy.FORBID,
        )


def test_unsupported_version_is_typed_error() -> None:
    with pytest.raises(UnsupportedSchemaVersionError):
        ingest_contract(
            {"contract_family": "eda_evidence_pack", "schema_version": "99.0"},
            expected_family="eda_evidence_pack",
            source_kind="external_artifact",
            migration_policy=MigrationPolicy.FORBID,
            repair_policy=RepairPolicy.FORBID,
        )


def test_internal_invalid_contract_never_calls_repair_provider() -> None:
    calls = 0

    def forbidden(_: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    payload = representative_evidence_pack().model_dump(mode="json")
    del payload["run_id"]
    with pytest.raises(InternalContractValidationError):
        ingest_contract(
            payload,
            expected_family="eda_evidence_pack",
            source_kind="internal_deterministic",
            migration_policy=MigrationPolicy.FORBID,
            repair_policy=RepairPolicy.FORBID,
            repair_provider=forbidden,
        )
    assert calls == 0


def test_llm_draft_gets_exactly_one_bounded_repair() -> None:
    calls: list[dict[str, object]] = []

    def repair(request: dict[str, object]) -> dict[str, object]:
        calls.append(request)
        return {
            "contract_family": "llm_draft",
            "schema_version": "1.0",
            "name": "repaired",
            "evidence_refs": ["known-ref"],
        }

    result = ingest_contract(
        {
            "contract_family": "llm_draft",
            "schema_version": "1.0",
            "name": "",
            "evidence_refs": ["known-ref"],
        },
        expected_family="llm_draft",
        source_kind="llm_generated",
        migration_policy=MigrationPolicy.FORBID,
        repair_policy=RepairPolicy.ONE_BOUNDED_REPAIR,
        repair_provider=repair,
        allowed_references=["known-ref"],
        registry=_llm_registry(),
        migration_graph=ContractMigrationGraph(),
    )

    assert result.repair_attempted is result.repair_succeeded is True
    assert result.contract.name == "repaired"
    assert len(calls) == 1
    assert calls[0]["validation_issues"]


def test_repair_cannot_add_unsupported_reference() -> None:
    def repair(_: dict[str, object]) -> dict[str, object]:
        return {
            "contract_family": "llm_draft",
            "schema_version": "1.0",
            "name": "repaired",
            "evidence_refs": ["unsupported-ref"],
        }

    with pytest.raises(BoundaryRepairError, match="unsupported references"):
        ingest_contract(
            {
                "contract_family": "llm_draft",
                "schema_version": "1.0",
                "name": "",
                "evidence_refs": ["known-ref"],
            },
            expected_family="llm_draft",
            source_kind="llm_generated",
            migration_policy=MigrationPolicy.FORBID,
            repair_policy=RepairPolicy.ONE_BOUNDED_REPAIR,
            repair_provider=repair,
            allowed_references=["known-ref"],
            registry=_llm_registry(),
            migration_graph=ContractMigrationGraph(),
        )


def test_bundle_loader_routes_through_ingest(monkeypatch, tmp_path) -> None:
    import kaggle_researcher.contracts.ingest as ingest_module

    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    path = tmp_path / "published_eda_evidence_bundle.json"
    write_json_atomic(path, bundle)
    calls = 0
    original = ingest_module.ingest_contract

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_module, "ingest_contract", counted)
    loaded = load_published_eda_evidence_bundle(path)

    assert calls == 1
    assert loaded.bundle_hash == bundle.bundle_hash
