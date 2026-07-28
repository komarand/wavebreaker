from __future__ import annotations

import asyncio
import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.eda.metrics import infer_metric_spec
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig
from kaggle_researcher.orchestration.full_run import FULL_RUN_STAGES


pytestmark = [pytest.mark.offline, pytest.mark.pipeline_smoke]

MANIFEST_PATH = Path("tests/fixtures/offline_characterization_manifest.json")


def require_offline_fixture(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.exists():
        raise AssertionError(f"missing offline characterization fixture: {candidate}")
    return candidate


def load_manifest() -> dict[str, Any]:
    path = require_offline_fixture(MANIFEST_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_covers_full_run_stages_and_dataset_cases() -> None:
    manifest = load_manifest()

    assert manifest["schema_version"] == 1
    assert [stage["stage_id"] for stage in manifest["stages"]] == list(FULL_RUN_STAGES)
    assert {case["case_id"] for case in manifest["dataset_cases"]} == {
        "binary_iid_classification",
        "regression",
        "temporal_stability_classification",
        "grouped_classification",
        "panel_entity_time",
        "ranking_query_grouped",
        "multi_output_submission",
        "multi_table_relational",
        "malformed_safe_partial_inventory",
        "unknown_custom_metric",
    }

    references = [manifest["full_pipeline"], *manifest["stages"]]
    for reference in references:
        for fixture_path in reference["fixture_paths"]:
            require_offline_fixture(fixture_path)
        for node_id in _node_ids(reference):
            _assert_pytest_node_exists(node_id)

    for case in manifest["dataset_cases"]:
        require_offline_fixture(case["fixture_path"])
        _assert_pytest_node_exists(case["pytest_node_id"])


def test_missing_fixture_failure_names_the_fixture() -> None:
    missing = Path("tests/fixtures/removed_offline_characterization_fixture.json")

    with pytest.raises(AssertionError, match=re.escape(str(missing))):
        require_offline_fixture(missing)


def test_external_network_attempt_is_rejected_by_offline_guard() -> None:
    with socket.socket() as candidate:
        with pytest.raises(AssertionError, match="external network access is forbidden"):
            candidate.connect(("192.0.2.1", 443))


def test_malformed_inventory_fixture_is_reported_without_crash() -> None:
    fixture = require_offline_fixture("tests/fixtures/eda/malformed_inventory_tiny")

    inventory = build_file_inventory(fixture)
    by_name = {item.name: item for item in inventory.files}

    assert by_name["train_base.csv"].can_read is True
    assert by_name["notes.txt"].can_read is False
    assert "Unsupported dataset file extension" in (by_name["notes.txt"].read_error or "")
    assert inventory.warnings == [
        "notes.txt: Unsupported dataset file extension: .txt"
    ]


def test_unknown_metric_fixture_requires_custom_implementation() -> None:
    fixture = require_offline_fixture(
        "tests/fixtures/offline_characterization_unknown_metric.json"
    )
    case = json.loads(fixture.read_text(encoding="utf-8"))

    metric = infer_metric_spec(case["metric_name"], case["task_type"])

    assert metric.family.value == case["expected_family"]
    assert metric.supports_local_eval is case["local_metric_available"]
    assert metric.needs_custom_implementation is case["needs_custom_implementation"]


def test_panel_entity_time_fixture_runs_offline(tmp_path: Path) -> None:
    fixture = require_offline_fixture("tests/fixtures/eda/panel_entity_time_tiny")
    result = asyncio.run(
        run_eda(
            EdaRunConfig(
                competition_id="panel_entity_time_tiny",
                hypotheses_path=fixture / "research_hypotheses.json",
                task_plan_path=fixture / "eda_task_plan.json",
                local_dataset_path=fixture,
                output_dir=tmp_path / "runs",
                download_dataset=False,
                profile_sample_rows=1000,
            )
        )
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert payload["competition_id"] == "panel_entity_time_tiny"
    assert payload["inferred_schema"]["candidate_time_columns"]
    assert payload["metric_evidence"]["metric_family"] == "temporal_stability"
    assert payload["validation_evidence"]["primary_validation"]["method"] == "temporal_holdout"
    assert result.evidence_manifest_path is not None
    assert result.published_bundle_path is not None


def test_repository_prerequisites_for_offline_execution_exist() -> None:
    assert require_offline_fixture("AGENTS.md").is_file()
    archive = require_offline_fixture("docs/archive")
    assert any(path.is_file() for path in archive.rglob("*.md"))

    task_root = Path("docs/tasks/v6")
    if task_root.exists():
        for path in task_root.rglob("*.md"):
            content = path.read_text(encoding="utf-8")
            assert "docs/archive/" not in content


def _node_ids(reference: dict[str, Any]) -> list[str]:
    if "pytest_node_id" in reference:
        return [reference["pytest_node_id"]]
    return list(reference["pytest_node_ids"])


def _assert_pytest_node_exists(node_id: str) -> None:
    path_text, separator, test_name = node_id.partition("::")
    assert separator and test_name, f"invalid pytest node id: {node_id}"
    path = require_offline_fixture(path_text)
    source = path.read_text(encoding="utf-8")
    pattern = rf"^(?:async\s+)?def\s+{re.escape(test_name)}\s*\("
    assert re.search(pattern, source, flags=re.MULTILINE), (
        f"pytest node does not exist: {node_id}"
    )
