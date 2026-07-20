from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.eda_task_plan import (
    EdaTaskPlan,
    ResearchArtifactContractError,
    load_eda_task_plan,
    migrate_eda_task_plan_payload,
    validate_research_artifact_bundle,
)
from kaggle_researcher.contracts.research_hypotheses import ResearchHypotheses
from kaggle_researcher.eda.schemas import EdaTaskPlan as EdaEngineTaskPlan


pytestmark = pytest.mark.contract


def _hypotheses() -> ResearchHypotheses:
    return ResearchHypotheses.model_validate({
        "schema_version": "1.0", "competition_id": "demo", "hypotheses": [{
            "hypothesis_id": "schema_001", "category": "schema", "claim": "Infer roles.",
            "priority": "P0", "confidence_before_eda": "medium",
        }],
    })


def _legacy_plan() -> dict[str, object]:
    return {
        "competition_id": "demo",
        "eda_tasks": [{
            "id": "eda_schema_001", "module": "schema_inferer", "priority": "P0",
            "hypothesis_ids": ["schema_001"], "blocking": True,
        }],
        "hypothesis_index": {"schema_001": {"category": "dataset_schema", "supporting_source_count": 3}},
    }


def test_shared_eda_task_plan_migrates_reported_legacy_shape() -> None:
    migration = migrate_eda_task_plan_payload(_legacy_plan())
    plan = EdaTaskPlan.model_validate(migration.canonical_payload)

    assert EdaEngineTaskPlan is EdaTaskPlan
    assert plan.eda_tasks[0].task_id == "eda_schema_001"
    assert plan.hypothesis_index == {"schema_001": ["eda_schema_001"]}
    assert any("id->task_id" in item for item in migration.applied_migrations)
    assert any("object->list" in item for item in migration.applied_migrations)
    validate_research_artifact_bundle(_hypotheses(), plan)


def test_task_plan_rejects_conflicts_invalid_index_and_unknown_references() -> None:
    conflicting = _legacy_plan()
    task = conflicting["eda_tasks"][0]
    assert isinstance(task, dict)
    task["task_id"] = "different"
    with pytest.raises(ResearchArtifactContractError, match="conflicting"):
        migrate_eda_task_plan_payload(conflicting)

    invalid_index = _legacy_plan()
    invalid_index["hypothesis_index"] = {"schema_001": "not-a-list"}
    with pytest.raises(ResearchArtifactContractError, match="hypothesis_index"):
        migrate_eda_task_plan_payload(invalid_index)

    plan = EdaTaskPlan.model_validate(migrate_eda_task_plan_payload(_legacy_plan()).canonical_payload)
    plan.eda_tasks[0].related_hypothesis_ids = ["unknown"]
    with pytest.raises(ResearchArtifactContractError, match="unknown hypotheses"):
        validate_research_artifact_bundle(_hypotheses(), plan)


def test_dedicated_loader_migrates_legacy_plan_before_bundle_validation(tmp_path: Path) -> None:
    path = tmp_path / "eda_task_plan.json"
    path.write_text(json.dumps(_legacy_plan()), encoding="utf-8")

    plan, migration = load_eda_task_plan(path, hypotheses=_hypotheses())

    assert migration.migrated is True
    assert plan.hypothesis_index["schema_001"] == ["eda_schema_001"]
