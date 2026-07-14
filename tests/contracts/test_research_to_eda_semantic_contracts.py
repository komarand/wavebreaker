from __future__ import annotations

from copy import deepcopy

import pytest

from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import (
    CATEGORY_PREFIXES,
    MODULE_DEPENDENCIES,
    STABLE_ERROR_CODES,
    validate_research_to_eda_contract,
)
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def _validate(research: dict, plan: dict):
    return validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(research),
        EdaTaskPlan.model_validate(plan),
    )


def _codes(result) -> set[str]:
    return {issue.code for issue in result.errors}


def test_competition_identity_mismatch_is_blocking() -> None:
    plan = valid_task_plan_payload(competition_id="other")
    result = _validate(valid_research_payload(), plan)
    assert "competition_id_mismatch" in _codes(result)


def test_required_categories_must_be_p0_not_merely_present() -> None:
    research = valid_research_payload()
    research["hypotheses"][0]["priority"] = "P1"
    result = _validate(research, valid_task_plan_payload())
    assert "missing_p0_hypothesis" in _codes(result)
    assert result.errors[0].severity == "error"


@pytest.mark.parametrize("category", ["schema", "metric", "validation", "leakage"])
def test_missing_each_p0_category_is_rejected(category: str) -> None:
    research = valid_research_payload()
    research["hypotheses"] = [item for item in research["hypotheses"] if item["category"] != category]
    result = _validate(research, valid_task_plan_payload())
    assert any(
        issue.code == "missing_p0_hypothesis" and category in issue.related_ids
        for issue in result.errors
    )


def test_duplicate_optional_category_is_allowed() -> None:
    research = valid_research_payload()
    duplicate = deepcopy(research["hypotheses"][0])
    duplicate["hypothesis_id"] = "schema_secondary"
    duplicate["priority"] = "P1"
    research["hypotheses"].append(duplicate)
    assert _validate(research, valid_task_plan_payload()).valid


def test_prefix_mismatch_is_a_stable_warning_policy() -> None:
    research = valid_research_payload()
    research["hypotheses"][0]["hypothesis_id"] = "metric_wrong_category"
    plan = valid_task_plan_payload()
    for task in plan["eda_tasks"]:
        task["related_hypothesis_ids"] = [
            "metric_wrong_category" if value == "schema_core" else value
            for value in task["related_hypothesis_ids"]
        ]
    plan["hypothesis_index"]["metric_wrong_category"] = plan["hypothesis_index"].pop("schema_core")
    result = _validate(research, plan)
    assert result.valid
    assert {issue.code for issue in result.warnings} == {"hypothesis_id_category_prefix_mismatch"}


def test_id_prefix_registry_matches_public_categories() -> None:
    assert CATEGORY_PREFIXES["validation"] == ("val_", "validation_")
    assert CATEGORY_PREFIXES["data_quality"] == ("data_quality_", "dq_")


def test_unknown_check_and_category_mismatch_are_reported() -> None:
    research = valid_research_payload()
    research["hypotheses"][0]["expected_eda_checks"] = ["schema_inferer.not_real"]
    result = _validate(research, valid_task_plan_payload())
    assert {"unknown_eda_check", "hypothesis_check_category_mismatch"} <= _codes(result)


def test_p0_cannot_depend_only_on_optional_module() -> None:
    research = valid_research_payload()
    research["hypotheses"][0]["expected_eda_checks"] = ["relationship_inferer.relationships"]
    research["hypotheses"][0]["category"] = "relationship"
    research["hypotheses"][0]["hypothesis_id"] = "relationship_core"
    result = _validate(research, valid_task_plan_payload())
    assert "p0_depends_only_on_optional_module" in _codes(result)


def test_unknown_module_and_blocking_conflict_are_reported() -> None:
    plan = valid_task_plan_payload()
    plan["eda_tasks"][0]["module"] = "future_module"
    result = _validate(valid_research_payload(), plan)
    assert "unknown_eda_module" in _codes(result)

    plan = valid_task_plan_payload()
    plan["eda_tasks"][0]["blocking"] = False
    result = _validate(valid_research_payload(), plan)
    assert "blocking_task_conflict" in _codes(result)


def test_module_dependency_order_and_missing_planned_dependency_are_reported() -> None:
    plan = valid_task_plan_payload()
    plan["recommended_module_sequence"] = list(reversed(plan["recommended_module_sequence"]))
    assert "module_dependency_order_violation" in _codes(_validate(valid_research_payload(), plan))

    plan = valid_task_plan_payload()
    plan["recommended_module_sequence"].remove("table_profiler")
    result = _validate(valid_research_payload(), plan)
    assert "module_dependency_missing_from_sequence" in _codes(result)


def test_dependency_graph_is_the_generic_engine_graph() -> None:
    assert MODULE_DEPENDENCIES["schema_inferer"] == ("file_inventory",)
    assert MODULE_DEPENDENCIES["validation_analyzer"] == ("metric_analyzer",)
    assert MODULE_DEPENDENCIES["leakage_checker"] == ("validation_analyzer",)
    assert MODULE_DEPENDENCIES["feature_probe"] == ("relationship_inferer",)


def test_machine_readable_issue_code_registry_is_frozen() -> None:
    assert "competition_id_mismatch" in STABLE_ERROR_CODES
    assert "unknown_eda_check" in STABLE_ERROR_CODES
    assert "forced_temporal_without_evidence" in STABLE_ERROR_CODES
    assert "premature_eda_factual_claim" in STABLE_ERROR_CODES


@pytest.mark.parametrize(
    "claim",
    [
        "EDA confirmed leakage.",
        "Leakage was found in a feature.",
        "The dataset contains 10,000 rows.",
        "Baseline achieved 0.91 AUC.",
    ],
)
def test_premature_factual_claims_are_rejected(claim: str) -> None:
    research = valid_research_payload()
    research["hypotheses"][0]["claim"] = claim
    assert "premature_eda_factual_claim" in _codes(_validate(research, valid_task_plan_payload()))


@pytest.mark.parametrize(
    "claim",
    [
        "Check whether leakage exists.",
        "Hypothesis: a date column may be diagnostic.",
        "Sources suggest temporal validation may require investigation.",
        "Potential risk of train/test overlap should be measured.",
    ],
)
def test_hypothetical_wording_is_allowed(claim: str) -> None:
    research = valid_research_payload()
    research["hypotheses"][0]["claim"] = claim
    assert _validate(research, valid_task_plan_payload()).valid
