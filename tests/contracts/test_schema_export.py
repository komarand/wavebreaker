from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.export_schemas import (
    DEFAULT_SCHEMA_ROOT,
    SchemaExportError,
    build_public_schemas,
    export_schemas,
    schema_output_path,
    validate_registered_fixture,
)
from kaggle_researcher.contracts.registry import CONTRACT_REGISTRY, ContractRegistry
from tests.contracts.factories import valid_research_payload
from tests.fixtures.evidence_contract import representative_evidence_pack
from kaggle_researcher.contracts.evidence_manifest import publish_eda_evidence_bundle


pytestmark = pytest.mark.contract


def test_export_is_deterministic_and_committed_schemas_have_no_drift(tmp_path: Path) -> None:
    generated_root = tmp_path / "schemas"
    generated = export_schemas(generated_root)

    committed = {
        path.relative_to(DEFAULT_SCHEMA_ROOT): path.read_text(encoding="utf-8")
        for path in DEFAULT_SCHEMA_ROOT.rglob("*.json")
    }
    actual = {
        path.relative_to(generated_root): path.read_text(encoding="utf-8")
        for path in generated
    }
    assert actual == committed, (
        "Committed contract schemas drifted. Run: "
        ".venv-win\\Scripts\\python.exe -m kaggle_researcher.contracts.export_schemas"
    )


def test_every_registered_family_version_has_one_header_pinned_schema() -> None:
    schemas = build_public_schemas()
    assert len(schemas) == len(CONTRACT_REGISTRY)
    for family, version in CONTRACT_REGISTRY:
        relative = schema_output_path(Path(), family, version)
        schema = schemas[relative]
        properties = schema["properties"]
        assert properties["contract_family"]["const"] == family
        assert properties["schema_version"]["const"] == version
        assert relative.suffix == ".json"
        json.dumps(schema)


def test_representative_fixtures_validate_with_schema_source_models() -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    fixtures = (
        ("research_hypotheses", "1.0", valid_research_payload()),
        ("eda_evidence_pack", "1.0", bundle.evidence_pack.model_dump(mode="json")),
        ("evidence_reference_manifest", "1.0", bundle.evidence_manifest.model_dump(mode="json")),
        ("published_eda_evidence_bundle", "1.0", bundle.model_dump(mode="json")),
    )
    for family, version, fixture in fixtures:
        validate_registered_fixture(family, version, fixture)


def test_duplicate_schema_output_is_rejected() -> None:
    class First(ContractModel):
        contract_family: Literal["final_strategy"] = "final_strategy"
        schema_version: Literal["1.0"] = "1.0"

    class Second(ContractModel):
        contract_family: Literal["final_strategy_result"] = "final_strategy_result"
        schema_version: Literal["1.0"] = "1.0"

    registry = ContractRegistry()
    registry.register("final_strategy", "1.0", First, current=True)
    registry.register("final_strategy_result", "1.0", Second, current=True)
    with pytest.raises(SchemaExportError, match="Duplicate schema output"):
        build_public_schemas(registry)
