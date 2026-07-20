from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.errors import (
    ContractMigrationError,
    UnsupportedSchemaVersionError,
)
from kaggle_researcher.contracts.manifest import (
    ArtifactPointer,
    RunManifest,
    StageStatus,
    artifact_pointer,
    load_run_manifest,
    migrate_run_manifest,
    new_run_manifest,
    validate_artifact_pointer,
    write_run_manifest_atomic,
)
from kaggle_researcher.contracts.manifest import ManifestConfigSnapshot


pytestmark = [pytest.mark.contract, pytest.mark.manifest_migration]
FIXTURES = Path("tests/fixtures/manifests")


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_current_manifest_loads_unchanged_and_is_idempotent(tmp_path: Path) -> None:
    manifest = new_run_manifest(
        run_id=tmp_path.name,
        competition_id="demo",
        config=ManifestConfigSnapshot(output_root=str(tmp_path.parent)),
    )
    first = migrate_run_manifest(manifest.model_dump(mode="json"), run_dir=tmp_path)
    second = migrate_run_manifest(first.value.model_dump(mode="json"), run_dir=tmp_path)

    assert first.migrated is False
    assert second.migrated is False
    assert second.value == manifest


def test_supported_unversioned_manifest_maps_explicit_aliases(tmp_path: Path) -> None:
    result = migrate_run_manifest(
        _fixture("manifest_unversioned_v0.json"), run_dir=tmp_path
    )

    assert result.migrated is True
    assert result.value.schema_version == "1.0"
    assert result.value.status.value == "running"
    assert result.value.stages["research_scout"].status is StageStatus.COMPLETED
    assert any("mapped stage 'scout'" in item for item in result.applied_migrations)


def test_future_version_and_unknown_aliases_fail_without_guessing(tmp_path: Path) -> None:
    payload = _fixture("manifest_unversioned_v0.json")
    payload["schema_version"] = "99.0"
    with pytest.raises(UnsupportedSchemaVersionError):
        migrate_run_manifest(payload, run_dir=tmp_path)

    payload = _fixture("manifest_unversioned_v0.json")
    payload["stages"] = {"mystery": {"status": "success", "outputs": {}}}
    with pytest.raises(ContractMigrationError, match="Unknown legacy stage"):
        migrate_run_manifest(payload, run_dir=tmp_path)

    payload = _fixture("manifest_unversioned_v0.json")
    payload["stages"]["scout"]["status"] = "maybe_done"
    with pytest.raises(ContractMigrationError, match="Unknown stage status"):
        migrate_run_manifest(payload, run_dir=tmp_path)


def test_absolute_inside_path_converts_but_external_and_traversal_fail(tmp_path: Path) -> None:
    inside = tmp_path / "eda" / "eda_evidence_pack.json"
    inside.parent.mkdir(parents=True)
    inside.write_text("{}", encoding="utf-8")
    payload = _fixture("manifest_legacy_absolute_paths.json")
    payload["stages"]["eda"]["outputs"]["evidence_pack"] = str(inside)
    payload["final_outputs"] = {}
    migrated = migrate_run_manifest(payload, run_dir=tmp_path)
    assert migrated.value.stages["eda_engine"].outputs["evidence_pack"].relative_path == "eda/eda_evidence_pack.json"

    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    payload["stages"]["eda"]["outputs"]["evidence_pack"] = str(outside)
    with pytest.raises(ContractMigrationError, match="Unsafe external"):
        migrate_run_manifest(payload, run_dir=tmp_path)

    payload["stages"]["eda"]["outputs"]["evidence_pack"] = "../outside.json"
    with pytest.raises(ContractMigrationError, match="Unsafe"):
        migrate_run_manifest(payload, run_dir=tmp_path)


def test_raw_error_becomes_structured_and_bounded(tmp_path: Path) -> None:
    result = migrate_run_manifest(
        _fixture("manifest_failed_reasoning.json"), run_dir=tmp_path
    )
    error = result.value.stages["reasoning_context"].error
    assert error is not None
    assert error.error_type == "LegacyStageError"
    assert error.message == "provider returned invalid JSON"
    assert error.suggested_rerun_stage == "reasoning_context"


def test_load_migration_preserves_original_backup(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    original = _fixture("manifest_unversioned_v0.json")
    path.write_text(json.dumps(original), encoding="utf-8")

    result = load_run_manifest(path, run_dir=tmp_path)

    assert result.migrated
    backup = tmp_path / "run_manifest.legacy.json"
    assert json.loads(backup.read_text(encoding="utf-8")) == original
    assert RunManifest.model_validate_json(path.read_text(encoding="utf-8"))

    backup.write_text('{"sentinel": true}', encoding="utf-8")
    load_run_manifest(path, run_dir=tmp_path)
    assert json.loads(backup.read_text(encoding="utf-8")) == {"sentinel": True}


def test_pointer_hash_and_missing_file_invalidate_reuse(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    pointer = artifact_pointer(path, run_dir=tmp_path)
    assert validate_artifact_pointer(pointer, run_dir=tmp_path) == (True, None)

    path.write_text('{"value": 2}', encoding="utf-8")
    valid, reason = validate_artifact_pointer(pointer, run_dir=tmp_path)
    assert not valid and "mismatch" in str(reason)
    path.unlink()
    assert validate_artifact_pointer(pointer, run_dir=tmp_path) == (False, "artifact is missing")


def test_atomic_writer_preserves_previous_manifest_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kaggle_researcher.contracts import artifacts

    path = tmp_path / "run_manifest.json"
    original = new_run_manifest(
        run_id="old", competition_id="demo", config=ManifestConfigSnapshot()
    )
    write_run_manifest_atomic(original, path)
    replacement = original.model_copy(update={"run_id": "new"})
    real_replace = artifacts.os.replace

    def fail_replace(source, target):
        if str(source).endswith(".tmp"):
            raise OSError("replace failed")
        return real_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(Exception, match="atomically write"):
        write_run_manifest_atomic(replacement, path)
    assert RunManifest.model_validate_json(path.read_text(encoding="utf-8")).run_id == "old"


def test_production_orchestration_has_no_generic_canonical_context_access() -> None:
    root = Path("kaggle_researcher/orchestration")
    forbidden = {"research", "research_stage", "eda", "eda_stage", "reasoning", "reasoning_stage", "final", "final_stage", "plan_data", "docs", "reasoning_outputs"}
    violations: list[str] = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "context":
                if isinstance(node.slice, ast.Constant) and node.slice.value in forbidden:
                    violations.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "context" and node.func.attr == "get":
                    if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in forbidden:
                        violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_registry_classes_are_defined_only_in_approved_modules() -> None:
    names = {
        "EvidenceRegistry", "ExperimentRegistry", "HypothesisRegistry",
        "RiskRegistry", "SafetyConstraintRegistry", "ValidationRequirementRegistry",
    }
    approved = {
        Path("kaggle_researcher/contracts/evidence.py"),
        Path("kaggle_researcher/contracts/registries.py"),
    }
    definitions: list[tuple[Path, str]] = []
    for path in Path("kaggle_researcher").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions.extend(
            (path, node.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name in names
        )
    assert definitions
    assert all(path in approved for path, _ in definitions)

