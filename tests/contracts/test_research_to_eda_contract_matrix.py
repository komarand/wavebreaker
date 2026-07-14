from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def _case(name: str) -> tuple[dict, dict]:
    research = valid_research_payload()
    plan = valid_task_plan_payload()
    if name == "regression":
        plan["task_type"], plan["metric"] = "regression", {"name": "rmse"}
    elif name == "grouped_binary":
        research["hypotheses"][2]["expected_eda_checks"] = ["validation_analyzer.group_policy"]
        research["hypotheses"][3]["expected_eda_checks"] = ["leakage_checker.group_overlap"]
        plan["dataset"]["group_column"] = "customer_id"
    elif name == "ranking":
        plan["task_type"], plan["metric"] = "ranking", {"name": "ndcg"}
        research["hypotheses"][2]["expected_eda_checks"] = ["validation_analyzer.ranking_validation"]
        research["hypotheses"][3]["expected_eda_checks"] = ["leakage_checker.ranking_query_overlap"]
    elif name == "temporal_stability":
        plan["metric"] = {"name": "gini_stability"}
        research["hypotheses"][2]["expected_eda_checks"] = [
            "validation_analyzer.temporal_cv_feasibility"
        ]
    elif name == "competition_id_mismatch":
        plan["competition_id"] = "other"
    elif name == "duplicate_hypothesis_ids":
        research["hypotheses"].append(deepcopy(research["hypotheses"][0]))
    elif name == "dangling_hypothesis_reference":
        plan["eda_tasks"][0]["related_hypothesis_ids"] = ["schema_missing"]
    elif name == "unknown_module":
        plan["eda_tasks"][0]["module"] = "future_module"
    elif name == "invalid_expected_check":
        research["hypotheses"][0]["expected_eda_checks"] = ["schema_inferer.future_check"]
    elif name == "metric_task_mismatch":
        plan["task_type"], plan["metric"] = "regression", {"name": "f1"}
    elif name == "forced_temporal_from_date_only":
        plan["dataset"]["date_column"] = "event_date"
        plan["eda_tasks"][4]["params"] = {"validation_policy": "temporal"}
    elif name == "missing_p0_hypothesis":
        research["hypotheses"][0]["priority"] = "P1"
    elif name == "premature_eda_claim":
        research["hypotheses"][0]["claim"] = "EDA confirmed an exact schema."
    return research, plan


@pytest.mark.parametrize(
    ("fixture_name", "expected_valid", "expected_codes"),
    [
        ("iid_binary", True, set()),
        ("regression", True, set()),
        ("grouped_binary", True, set()),
        ("ranking", True, set()),
        ("temporal_stability", True, set()),
        ("competition_id_mismatch", False, {"competition_id_mismatch"}),
        ("duplicate_hypothesis_ids", False, {"duplicate_hypothesis_id"}),
        ("dangling_hypothesis_reference", False, {"unknown_hypothesis_reference"}),
        ("unknown_module", False, {"unknown_eda_module"}),
        ("invalid_expected_check", False, {"unknown_eda_check"}),
        ("metric_task_mismatch", False, {"metric_task_type_mismatch"}),
        ("forced_temporal_from_date_only", False, {"forced_temporal_without_evidence"}),
        ("missing_p0_hypothesis", False, {"missing_p0_hypothesis"}),
        ("premature_eda_claim", False, {"premature_eda_factual_claim"}),
    ],
)
def test_research_to_eda_contract_matrix(
    fixture_name: str, expected_valid: bool, expected_codes: set[str]
) -> None:
    research, plan = _case(fixture_name)
    result = validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(research), EdaTaskPlan.model_validate(plan)
    )
    assert result.valid is expected_valid
    assert expected_codes <= {issue.code for issue in result.errors}


def test_validator_is_deterministic_and_does_not_mutate_inputs() -> None:
    research, plan = _case("iid_binary")
    models = (
        ResearchHypotheses.model_validate(research), EdaTaskPlan.model_validate(plan)
    )
    before = tuple(model.model_dump(mode="json") for model in models)
    first = validate_research_to_eda_contract(*models)
    second = validate_research_to_eda_contract(*models)
    assert first == second
    assert tuple(model.model_dump(mode="json") for model in models) == before


@pytest.mark.parametrize(
    ("kind", "name", "expected_valid", "expected_code"),
    [
        ("valid", "iid_binary", True, None),
        ("invalid", "competition_id_mismatch", False, "competition_id_mismatch"),
    ],
)
def test_static_file_fixture_matrix(
    kind: str, name: str, expected_valid: bool, expected_code: str | None
) -> None:
    root = Path(__file__).parents[1] / "fixtures" / "contracts" / "research_to_eda" / kind / name
    research = ResearchHypotheses.model_validate(json.loads(
        (root / "research_hypotheses.json").read_text(encoding="utf-8")
    ))
    plan = EdaTaskPlan.model_validate(json.loads(
        (root / "eda_task_plan.json").read_text(encoding="utf-8")
    ))
    result = validate_research_to_eda_contract(research, plan)
    assert result.valid is expected_valid
    if expected_code:
        assert expected_code in {issue.code for issue in result.errors}


@pytest.mark.parametrize("suffix", ["alpha", "beta_2", "x9", "long_stable_identifier"])
def test_parameterized_id_fuzz_roundtrip(suffix: str) -> None:
    research, plan = _case("iid_binary")
    old, new = "schema_core", f"schema_{suffix}"
    research["hypotheses"][0]["hypothesis_id"] = new
    for task in plan["eda_tasks"]:
        task["related_hypothesis_ids"] = [new if item == old else item for item in task["related_hypothesis_ids"]]
    plan["hypothesis_index"][new] = plan["hypothesis_index"].pop(old)
    result = validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(research), EdaTaskPlan.model_validate(plan)
    )
    assert result.valid
