from __future__ import annotations

import json
from pathlib import Path

from kaggle_researcher.eda.modules.hypothesis_evaluator import evaluate_hypotheses
from kaggle_researcher.eda.schemas import (
    HypothesisResult,
    LeakageCheckResult,
    MetricEvidence,
    ResearchHypotheses,
    ResearchHypothesis,
    ValidationEvidence,
)


FIXTURE_HYPOTHESES_PATH = Path("tests/fixtures/eda/home_credit_tiny/research_hypotheses.json")


def test_all_fixture_hypotheses_are_evaluated() -> None:
    hypotheses = _fixture_hypotheses()

    results = evaluate_hypotheses(hypotheses, _fixture_evidence_pack())

    assert len(results) == len(hypotheses)
    assert {result.hypothesis_id for result in results} == {
        hypothesis.hypothesis_id for hypothesis in hypotheses
    }


def test_confirmed_results_include_evidence_refs() -> None:
    hypotheses = _fixture_hypotheses()

    results = evaluate_hypotheses(hypotheses, _fixture_evidence_pack())

    confirmed_or_rejected = [
        result
        for result in results
        if result.status in {"confirmed", "rejected"}
    ]
    assert confirmed_or_rejected
    assert all(result.evidence_refs for result in confirmed_or_rejected)


def test_binary_iid_validation_hypothesis_rejects_date_only_temporal_claim() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="val_iid_001",
        category="validation",
        claim="Temporal CV is required because an event_date column exists.",
        priority="P0",
        confidence_before_eda="medium",
    )
    evidence_pack = {
        "validation_evidence": ValidationEvidence(
            primary_validation={"method": "stratified_kfold"},
            diagnostic_validations=[
                {"method": "temporal_holdout", "split_column": "event_date"}
            ],
            rejected_validations=[{"method": "temporal_holdout_as_default"}],
        )
    }

    result = evaluate_hypotheses([hypothesis], evidence_pack)[0]

    assert result.status == "rejected"
    assert result.evidence_refs == [
        "validation_evidence.diagnostic_validations",
        "validation_evidence.rejected_validations",
    ]


def test_temporal_diagnostic_feasibility_claim_is_only_partially_confirmed() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="val_diag_001",
        category="validation",
        claim="Temporal validation should be feasible because a date column exists.",
        priority="P0",
        confidence_before_eda="medium",
    )
    evidence_pack = {
        "validation_evidence": ValidationEvidence(
            primary_validation={"method": "stratified_kfold"},
            diagnostic_validations=[
                {"method": "temporal_holdout", "split_column": "event_date"}
            ],
            rejected_validations=[{"method": "temporal_holdout_as_default"}],
        )
    }

    result = evaluate_hypotheses([hypothesis], evidence_pack)[0]

    assert result.status == "partially_confirmed"
    assert result.limitations == [
        "Time column alone is insufficient for primary temporal validation."
    ]


def test_skipped_module_produces_skipped_result_with_limitation() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="metric_skip_001",
        category="metric",
        claim="Metric evidence should be available.",
        expected_eda_checks=["metric_analyzer.basic"],
        priority="P0",
        confidence_before_eda="medium",
    )

    result = evaluate_hypotheses(
        [hypothesis],
        evidence_pack_partial={},
        module_statuses={"metric_analyzer": "skipped"},
    )[0]

    assert result.status == "skipped"
    assert result.limitations


def test_missing_evidence_produces_not_testable_with_limitation() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="schema_missing_001",
        category="schema",
        claim="Schema should identify roles.",
        priority="P0",
        confidence_before_eda="medium",
    )

    result = evaluate_hypotheses([hypothesis], evidence_pack_partial={})[0]

    assert result.status == "not_testable"
    assert result.limitations


def test_unknown_category_is_not_testable() -> None:
    hypothesis = ResearchHypothesis.model_construct(
        hypothesis_id="unknown_001",
        category="mystery",
        claim="Mystery category should be evaluated.",
        expected_eda_checks=[],
        priority="P0",
        confidence_before_eda="medium",
        rationale=None,
        source_refs=[],
    )

    result = evaluate_hypotheses([hypothesis], evidence_pack_partial={})[0]

    assert result.status == "not_testable"
    assert result.limitations == ["Unsupported hypothesis category: mystery."]


def _fixture_hypotheses() -> list[ResearchHypothesis]:
    payload = json.loads(FIXTURE_HYPOTHESES_PATH.read_text(encoding="utf-8"))
    return ResearchHypotheses(**payload).hypotheses


def _fixture_evidence_pack() -> dict:
    return {
        "inferred_schema": {
            "train_base_table": "train_base.csv",
            "test_base_table": "test_base.csv",
            "target_column": "target",
            "primary_id_column": "case_id",
        },
        "metric_evidence": MetricEvidence(
            metric_name="gini_stability",
            requires_probabilities=True,
            rank_based=True,
            prediction_output_type="probability",
        ),
        "validation_evidence": ValidationEvidence(
            primary_validation={"method": "temporal_holdout", "split_column": "WEEK_NUM"}
        ),
        "leakage_evidence": [
            LeakageCheckResult(
                check_id="id_overlap",
                status="passed",
                severity="low",
                finding="No train/test ID overlap.",
            ),
            LeakageCheckResult(
                check_id="target_in_test",
                status="passed",
                severity="low",
                finding="Target absent from test.",
            ),
        ],
    }
