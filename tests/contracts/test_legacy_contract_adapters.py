from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.artifacts import (
    load_eda_publication_bundle,
    load_final_strategy,
    load_final_synthesis_context,
)
from kaggle_researcher.contracts.errors import ContractMigrationError
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from tests.fixtures.evidence_contract import stage_bundle, strategy_citing


pytestmark = pytest.mark.contract


def test_legacy_pack_is_adapted_in_memory_without_writing_manifest(tmp_path: Path) -> None:
    _, eda, _, _ = stage_bundle()
    pack_path = tmp_path / "eda_evidence_pack.json"
    original = eda.evidence_pack.model_dump_json(indent=2)
    pack_path.write_text(original, encoding="utf-8")

    bundle, warnings = load_eda_publication_bundle(tmp_path)

    assert bundle.evidence_manifest.origin == "legacy_migration"
    assert any("legacy" in warning.lower() for warning in warnings)
    assert any("origin=legacy_migration" in warning for warning in bundle.warnings)
    assert pack_path.read_text(encoding="utf-8") == original
    assert not (tmp_path / "evidence_reference_manifest.json").exists()
    assert not (tmp_path / "published_eda_evidence_bundle.json").exists()


def test_legacy_final_context_embeds_verified_bundle_without_mutating_file(
    tmp_path: Path,
) -> None:
    research, eda, reasoning, registries = stage_bundle()
    current = build_final_synthesis_context(
        competition_desc="Generic classification.",
        research=research,
        published_eda_bundle=eda.published_bundle,
        reasoning=reasoning,
        registries=registries,
        eda_summary_text="# EDA",
    )
    legacy = current.model_dump(mode="json")
    legacy.pop("contract_family")
    legacy.pop("schema_version")
    legacy.pop("published_eda_bundle")
    legacy["eda_evidence_pack"] = eda.evidence_pack.model_dump(mode="json")
    path = tmp_path / "final_synthesis_context.json"
    original = json.dumps(legacy, ensure_ascii=False, indent=2)
    path.write_text(original, encoding="utf-8")

    migrated = load_final_synthesis_context(path)

    assert migrated.evidence_manifest.origin == "legacy_migration"
    assert any("origin=legacy_migration" in item for item in migrated.limitations)
    assert path.read_text(encoding="utf-8") == original


def test_legacy_final_context_rejects_mixed_competition_snapshots(tmp_path: Path) -> None:
    research, eda, reasoning, registries = stage_bundle()
    current = build_final_synthesis_context(
        competition_desc="Generic classification.",
        research=research,
        published_eda_bundle=eda.published_bundle,
        reasoning=reasoning,
        registries=registries,
        eda_summary_text="# EDA",
    )
    legacy = current.model_dump(mode="json")
    legacy.pop("contract_family")
    legacy.pop("schema_version")
    legacy.pop("published_eda_bundle")
    legacy["eda_evidence_pack"] = eda.evidence_pack.model_dump(mode="json")
    legacy["research_hypotheses"]["competition_id"] = "different-run"
    path = tmp_path / "mixed_context.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ContractMigrationError, match="different competition runs"):
        load_final_synthesis_context(path)


def test_legacy_evidence_bindings_move_to_top_level_catalog(tmp_path: Path) -> None:
    strategy = strategy_citing(["validation_evidence.primary_validation"])
    legacy = strategy.model_dump(mode="json")
    legacy.pop("contract_family")
    legacy.pop("schema_version")
    legacy["actions"][0]["evidence_bindings"] = [{
        "ref": "validation_evidence.primary_validation",
        "resolved_value_preview": {"method": "stratified_kfold"},
        "role": "supporting",
    }]
    path = tmp_path / "final_strategy.json"
    original = json.dumps(legacy, ensure_ascii=False, indent=2)
    path.write_text(original, encoding="utf-8")

    migrated = load_final_strategy(path)

    ref = "validation_evidence.primary_validation"
    assert ref in migrated.evidence_catalog
    assert migrated.actions[0].evidence_bindings == []
    assert any("evidence_bindings" in warning for warning in migrated.warnings)
    assert path.read_text(encoding="utf-8") == original
