from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.artifacts import load_eda_task_plan, load_research_hypotheses
from kaggle_researcher.contracts.errors import UnsupportedSchemaVersionError
from kaggle_researcher.contracts.migration import (
    migrate_eda_task_plan_payload,
    migrate_research_hypotheses_payload,
)
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def test_current_versions_are_accepted(tmp_path: Path) -> None:
    research_path = tmp_path / "research.json"
    plan_path = tmp_path / "plan.json"
    research_path.write_text(json.dumps(valid_research_payload()), encoding="utf-8")
    plan_path.write_text(json.dumps(valid_task_plan_payload()), encoding="utf-8")
    hypotheses, research_migration = load_research_hypotheses(research_path)
    plan, plan_migration = load_eda_task_plan(plan_path, hypotheses=hypotheses)
    assert hypotheses.schema_version == plan.schema_version == "1.0"
    assert not research_migration.migrated
    assert not plan_migration.migrated


@pytest.mark.parametrize("version", ["2.0", "99.0", "0.9"])
def test_unsupported_versions_are_rejected(version: str) -> None:
    research = valid_research_payload()
    research["schema_version"] = version
    with pytest.raises(UnsupportedSchemaVersionError, match="Unsupported research_hypotheses"):
        migrate_research_hypotheses_payload(research)

    plan = valid_task_plan_payload()
    plan["schema_version"] = version
    with pytest.raises(UnsupportedSchemaVersionError, match="Unsupported eda_task_plan"):
        migrate_eda_task_plan_payload(plan)


def test_explicit_legacy_migration_preserves_semantic_ids_task_and_metric() -> None:
    research = valid_research_payload()
    research.pop("schema_version")
    research.pop("contract_family")
    plan = valid_task_plan_payload()
    plan.pop("schema_version")
    plan.pop("contract_family")
    original_hypothesis_ids = [item["hypothesis_id"] for item in research["hypotheses"]]
    original_task_ids = [item["task_id"] for item in plan["eda_tasks"]]

    migrated_research = migrate_research_hypotheses_payload(research)
    migrated_plan = migrate_eda_task_plan_payload(plan)
    assert migrated_research.migrated and migrated_plan.migrated
    assert [item["hypothesis_id"] for item in migrated_research.value["hypotheses"]] == original_hypothesis_ids
    assert [item["task_id"] for item in migrated_plan.value["eda_tasks"]] == original_task_ids
    assert migrated_plan.value["task_type"] == "binary_classification"
    assert migrated_plan.value["metric"] == {"name": "roc_auc"}
    assert len(migrated_research.value["hypotheses"]) == len(research["hypotheses"])

