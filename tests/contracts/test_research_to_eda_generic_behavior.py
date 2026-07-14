from __future__ import annotations

import inspect

import pytest

import kaggle_researcher.contracts.research_to_eda as boundary
from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypotheses
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def _valid(research: dict, plan: dict) -> bool:
    return boundary.validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(research), EdaTaskPlan.model_validate(plan)
    ).valid


@pytest.mark.parametrize(
    ("task_type", "metric"),
    [
        ("binary_classification", "roc_auc"),
        ("multiclass_classification", "log_loss"),
        ("regression", "rmse"),
        ("forecasting_tabular", "mae"),
        ("survival", "concordance_index"),
    ],
)
def test_generic_tabular_contracts_do_not_need_competition_specific_columns(
    task_type: str, metric: str
) -> None:
    plan = valid_task_plan_payload(task_type=task_type, metric_name=metric)
    plan["dataset"].update({"id_column": "row_id", "target_column": "target_value"})
    assert _valid(valid_research_payload(), plan)


def test_grouped_binary_contract_is_generic() -> None:
    research = valid_research_payload()
    research["hypotheses"][2]["expected_eda_checks"] = [
        "validation_analyzer.group_cv_feasibility"
    ]
    research["hypotheses"][3]["expected_eda_checks"] = ["leakage_checker.group_overlap"]
    plan = valid_task_plan_payload()
    plan["dataset"]["group_column"] = "customer_id"
    plan["eda_tasks"][4]["params"] = {"candidate_policy": "stratified_group_kfold"}
    assert _valid(research, plan)


def test_ranking_contract_is_generic() -> None:
    research = valid_research_payload()
    research["hypotheses"][2]["expected_eda_checks"] = [
        "validation_analyzer.ranking_validation"
    ]
    research["hypotheses"][3]["expected_eda_checks"] = [
        "leakage_checker.ranking_query_overlap"
    ]
    plan = valid_task_plan_payload(task_type="ranking", metric_name="ndcg")
    plan["dataset"].update({"query_column": "query_id", "target_column": "relevance"})
    assert _valid(research, plan)


def test_temporal_stability_contract_is_generic() -> None:
    research = valid_research_payload()
    research["hypotheses"][2]["expected_eda_checks"] = [
        "validation_analyzer.period_distribution",
        "validation_analyzer.temporal_cv_feasibility",
    ]
    plan = valid_task_plan_payload(metric_name="gini_stability")
    plan["dataset"].update({"id_column": "entity_id", "time_column": "period"})
    assert _valid(research, plan)


def test_boundary_source_has_no_home_credit_global_column_requirements() -> None:
    source = inspect.getsource(boundary)
    for forbidden in ("case_id", "WEEK_NUM", "date_decision"):
        assert forbidden not in source

