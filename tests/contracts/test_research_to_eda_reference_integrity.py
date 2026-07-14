from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def _result(research: dict, plan: dict):
    return validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(research), EdaTaskPlan.model_validate(plan)
    )


def _codes(result) -> set[str]:
    return {issue.code for issue in result.errors}


def test_duplicate_hypothesis_ids_are_detected() -> None:
    research = valid_research_payload()
    research["hypotheses"].append(deepcopy(research["hypotheses"][0]))
    assert "duplicate_hypothesis_id" in _codes(_result(research, valid_task_plan_payload()))


def test_unknown_hypothesis_reference_is_detected() -> None:
    plan = valid_task_plan_payload()
    plan["eda_tasks"][0]["related_hypothesis_ids"] = ["schema_missing"]
    assert "unknown_hypothesis_reference" in _codes(_result(valid_research_payload(), plan))


def test_unknown_index_key_and_unknown_task_are_detected() -> None:
    plan = valid_task_plan_payload()
    plan["hypothesis_index"]["schema_missing"] = ["inventory"]
    codes = _codes(_result(valid_research_payload(), plan))
    assert "unknown_hypothesis_index_key" in codes

    plan["hypothesis_index"]["schema_missing"] = ["task_missing"]
    with pytest.raises(ValidationError, match="references unknown tasks"):
        EdaTaskPlan.model_validate(plan)


def test_both_one_way_mapping_directions_are_detected() -> None:
    plan = valid_task_plan_payload()
    plan["hypothesis_index"]["schema_core"].remove("schema")
    assert "one_way_task_hypothesis_mapping" in _codes(_result(valid_research_payload(), plan))

    plan = valid_task_plan_payload()
    plan["hypothesis_index"]["metric_core"].append("schema")
    assert "one_way_hypothesis_task_mapping" in _codes(_result(valid_research_payload(), plan))


def test_duplicate_related_and_index_mappings_are_detected_without_repair() -> None:
    plan = EdaTaskPlan.model_validate(valid_task_plan_payload())
    plan.eda_tasks[0].related_hypothesis_ids.append(plan.eda_tasks[0].related_hypothesis_ids[0])
    plan.hypothesis_index["schema_core"].append("inventory")
    result = validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(valid_research_payload()), plan
    )
    assert {
        "duplicate_related_hypothesis_id", "duplicate_hypothesis_index_mapping"
    } <= _codes(result)
    assert len(plan.eda_tasks[0].related_hypothesis_ids) == 2


def test_hypothesis_driven_task_cannot_drop_all_references() -> None:
    plan = valid_task_plan_payload()
    plan["eda_tasks"][3]["related_hypothesis_ids"] = []
    plan["hypothesis_index"]["metric_core"] = []
    result = _result(valid_research_payload(), plan)
    assert "hypothesis_driven_task_without_hypothesis" in _codes(result)


def test_many_to_many_references_are_valid_when_bidirectional() -> None:
    research = valid_research_payload()
    plan = valid_task_plan_payload()
    plan["eda_tasks"][4]["related_hypothesis_ids"].append("leak_core")
    plan["hypothesis_index"]["leak_core"].append("validation")
    assert _result(research, plan).valid
