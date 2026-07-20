from __future__ import annotations

import pytest

from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from tests.contracts.factories import valid_research_payload, valid_task_plan_payload


pytestmark = [pytest.mark.contract, pytest.mark.offline, pytest.mark.unit]


def _validate(task_type: str, metric: str, *, research: dict | None = None, plan: dict | None = None):
    research = research or valid_research_payload()
    plan = plan or valid_task_plan_payload(task_type=task_type, metric_name=metric)
    return validate_research_to_eda_contract(
        ResearchHypotheses.model_validate(research), EdaTaskPlan.model_validate(plan)
    )


@pytest.mark.parametrize(
    ("task_type", "metric"),
    [
        ("binary_classification", "roc_auc"),
        ("multiclass_classification", "log_loss"),
        ("binary_classification", "f1"),
        ("regression", "rmse"),
        ("regression", "mae"),
        ("forecasting_tabular", "rmse"),
        ("survival", "concordance_index"),
    ],
)
def test_supported_task_metric_pairs(task_type: str, metric: str) -> None:
    assert _validate(task_type, metric).valid


@pytest.mark.parametrize(
    ("task_type", "metric"),
    [
        ("binary_classification", "rmse"),
        ("regression", "f1"),
        ("ranking", "roc_auc"),
    ],
)
def test_obvious_task_metric_contradictions(task_type: str, metric: str) -> None:
    result = _validate(task_type, metric)
    assert "metric_task_type_mismatch" in {issue.code for issue in result.errors}


@pytest.mark.parametrize(
    ("metric", "field", "wrong"),
    [
        ("roc_auc", "requires_probabilities", False),
        ("roc_auc", "rank_based", False),
        ("log_loss", "requires_calibration", False),
        ("f1", "threshold_search_needed", False),
        ("rmse", "threshold_search_needed", True),
    ],
)
def test_explicit_metric_semantics_cannot_contradict_registry(
    metric: str, field: str, wrong: bool
) -> None:
    task_type = "regression" if metric == "rmse" else "binary_classification"
    plan = valid_task_plan_payload(task_type=task_type, metric_name=metric)
    plan["metric"][field] = wrong
    result = _validate(task_type, metric, plan=plan)
    assert "metric_semantics_mismatch" in {issue.code for issue in result.errors}


def test_ranking_requires_ranking_and_query_group_checks() -> None:
    result = _validate("ranking", "ndcg")
    codes = {issue.code for issue in result.errors}
    assert {"ranking_validation_check_missing", "ranking_group_check_missing"} <= codes

    research = valid_research_payload()
    research["hypotheses"][2]["expected_eda_checks"] = [
        "validation_analyzer.ranking_validation"
    ]
    research["hypotheses"][3]["expected_eda_checks"] = [
        "leakage_checker.ranking_query_overlap"
    ]
    assert _validate("ranking", "ndcg", research=research).valid


def test_temporal_stability_requires_time_check_but_not_specific_columns() -> None:
    result = _validate("binary_classification", "gini_stability")
    assert "temporal_metric_check_missing" in {issue.code for issue in result.errors}
    research = valid_research_payload()
    research["hypotheses"][2]["expected_eda_checks"] = [
        "validation_analyzer.temporal_cv_feasibility"
    ]
    assert _validate("binary_classification", "gini_stability", research=research).valid


def test_unknown_metric_degrades_with_warning_and_explicit_limitation() -> None:
    research = valid_research_payload()
    research["hypotheses"][1]["limitations"] = ["Local scoring implementation is unknown."]
    result = _validate("custom", "my_competition_metric", research=research)
    assert result.valid
    assert "unknown_metric" in {issue.code for issue in result.warnings}


def test_unknown_metric_cannot_claim_local_implementation() -> None:
    research = valid_research_payload()
    research["hypotheses"][1]["limitations"] = ["Unknown local implementation."]
    plan = valid_task_plan_payload(task_type="custom", metric_name="mystery")
    plan["metric"]["local_metric_available"] = True
    result = _validate("custom", "mystery", research=research, plan=plan)
    assert "unknown_metric_claims_local_implementation" in {
        issue.code for issue in result.errors
    }


def test_ordinary_iid_task_is_not_forced_temporal_by_a_date_column() -> None:
    plan = valid_task_plan_payload()
    plan["dataset"]["date_column"] = "event_date"
    assert _validate("binary_classification", "roc_auc", plan=plan).valid
    plan["eda_tasks"][4]["params"] = {"validation_policy": "temporal"}
    result = _validate("binary_classification", "roc_auc", plan=plan)
    assert "forced_temporal_without_evidence" in {issue.code for issue in result.errors}

