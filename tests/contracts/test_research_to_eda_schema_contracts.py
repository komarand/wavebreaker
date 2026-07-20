from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.eda import EdaTask, EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypothesis, ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def test_valid_boundary_models_are_accepted() -> None:
    hypotheses = ResearchHypotheses.model_validate(valid_research_payload())
    plan = EdaTaskPlan.model_validate(valid_task_plan_payload())
    assert validate_research_to_eda_contract(hypotheses, plan).valid


@pytest.mark.parametrize(
    "field",
    [
        "hypothesis_id", "category", "claim", "confidence_before_eda",
    ],
)
def test_research_hypothesis_rejects_missing_required_field(field: str) -> None:
    payload = valid_research_payload()["hypotheses"][0].copy()
    payload.pop(field)
    with pytest.raises(ValidationError):
        ResearchHypothesis.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "credit_specific"),
        ("priority", "urgent"),
        ("confidence_before_eda", "certain"),
        ("claim", ""),
        ("hypothesis_id", ""),
        ("expected_eda_checks", "schema_inferer.detect_target"),
    ],
)
def test_research_hypothesis_rejects_invalid_values(field: str, value: object) -> None:
    payload = valid_research_payload()["hypotheses"][0].copy()
    payload[field] = value
    with pytest.raises(ValidationError):
        ResearchHypothesis.model_validate(payload)


def test_unknown_fields_are_forbidden_recursively() -> None:
    research = valid_research_payload()
    research["unknown"] = True
    with pytest.raises(ValidationError):
        ResearchHypotheses.model_validate(research)

    plan = valid_task_plan_payload()
    plan["eda_tasks"][0]["unknown"] = True
    with pytest.raises(ValidationError):
        EdaTaskPlan.model_validate(plan)


def test_task_schema_rejects_bad_priority_empty_id_and_non_mapping_params() -> None:
    base = valid_task_plan_payload()["eda_tasks"][0]
    for field, value in (("priority", "P9"), ("task_id", ""), ("params", [])):
        payload = deepcopy(base)
        payload[field] = value
        with pytest.raises(ValidationError):
            EdaTask.model_validate(payload)


def test_duplicate_task_ids_are_rejected_by_canonical_schema() -> None:
    payload = valid_task_plan_payload()
    payload["eda_tasks"].append(deepcopy(payload["eda_tasks"][0]))
    with pytest.raises(ValidationError, match="duplicate task_id"):
        EdaTaskPlan.model_validate(payload)


def test_mutable_defaults_are_not_shared() -> None:
    first = ResearchHypotheses(competition_id="a")
    second = ResearchHypotheses(competition_id="b")
    first.hypotheses.append(ResearchHypothesis.model_validate(valid_research_payload()["hypotheses"][0]))
    assert second.hypotheses == []

    first_plan = EdaTaskPlan(competition_id="a")
    second_plan = EdaTaskPlan(competition_id="b")
    first_plan.dataset["x"] = 1
    assert second_plan.dataset == {}


def test_json_schema_keeps_strict_additional_properties_and_version_literal() -> None:
    for model in (ResearchHypotheses, ResearchHypothesis, EdaTaskPlan, EdaTask):
        assert model.model_json_schema()["additionalProperties"] is False
    assert ResearchHypotheses.model_json_schema()["properties"]["schema_version"]["const"] == "1.0"
    assert EdaTaskPlan.model_json_schema()["properties"]["schema_version"]["const"] == "1.0"

