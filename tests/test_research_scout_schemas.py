from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kaggle_researcher.eda.schemas import EdaTaskPlan, ResearchHypotheses
from kaggle_researcher.research_scout import build_research_scout_summary
from kaggle_researcher.research_scout.schemas import (
    EdaTaskPlanDraft,
    ResearchScoutOutput,
    ScoutEdaTask,
    ScoutHypothesis,
    ScoutLimitation,
    ScoutStructuredFinding,
)
from kaggle_researcher.research_scout.prompts import RESEARCH_SCOUT_SYSTEM_PROMPT


def test_research_scout_output_validates_and_serializes_to_eda_inputs(tmp_path) -> None:
    output = _scout_output()

    research_payload = output.to_research_hypotheses_payload()
    task_plan_payload = output.to_eda_task_plan_payload()
    summary = output.to_summary_markdown()
    paths = output.write_outputs(tmp_path)

    ResearchHypotheses(**research_payload)
    EdaTaskPlan(**task_plan_payload)
    assert "schema_001" in summary
    assert paths["research_hypotheses"].is_file()
    assert paths["eda_task_plan"].is_file()
    assert paths["research_scout_summary"].is_file()
    ResearchHypotheses(**json.loads(paths["research_hypotheses"].read_text(encoding="utf-8")))
    EdaTaskPlan(**json.loads(paths["eda_task_plan"].read_text(encoding="utf-8")))


def test_generated_payload_contains_required_generic_fields() -> None:
    output = _scout_output()
    research_payload = output.to_research_hypotheses_payload()
    task_plan_payload = output.to_eda_task_plan_payload()

    hypothesis = research_payload["hypotheses"][0]
    assert {
        "hypothesis_id",
        "category",
        "claim",
        "rationale",
        "expected_eda_checks",
        "priority",
        "confidence_before_eda",
        "source_refs",
    } <= set(hypothesis)
    assert {
        "competition_id",
        "task_type",
        "metric",
        "eda_tasks",
        "hypothesis_index",
        "recommended_module_sequence",
        "recommended_human_checklist",
        "blocking_tasks",
    } <= set(task_plan_payload)


def test_invalid_hypothesis_category_or_status_fails_validation() -> None:
    with pytest.raises(ValidationError):
        ScoutHypothesis(
            hypothesis_id="mystery_001",
            category="mystery",
            claim="Invalid category.",
            rationale="Invalid category should fail.",
            priority="P0",
            confidence_before_eda="medium",
        )

    with pytest.raises(ValidationError):
        ScoutHypothesis(
            hypothesis_id="schema_001",
            category="schema",
            claim="Invalid status.",
            rationale="Invalid status should fail.",
            priority="P0",
            confidence_before_eda="medium",
            status="confirmed",
        )


def test_hypothesis_id_must_use_stable_category_prefix() -> None:
    with pytest.raises(ValidationError):
        ScoutHypothesis(
            hypothesis_id="validation_001",
            category="metric",
            claim="Metric should be resolved.",
            rationale="Metric checks need stable ids.",
            priority="P0",
            confidence_before_eda="medium",
        )


def test_prompts_do_not_assume_temporal_or_home_credit_defaults() -> None:
    prompt = RESEARCH_SCOUT_SYSTEM_PROMPT.lower()

    assert "do not assume temporal validation by default" in prompt
    assert "do not assume home credit column names" in prompt


def test_legacy_research_scout_imports_still_work() -> None:
    assert callable(build_research_scout_summary)


def _scout_output() -> ResearchScoutOutput:
    hypotheses = [
        ScoutHypothesis(
            hypothesis_id="schema_001",
            category="schema",
            claim="Infer generic train/test/schema roles from available files.",
            rationale="Schema roles are required before any downstream EDA evidence.",
            expected_eda_checks=["schema_inferer.roles"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["doc_schema"],
        ),
        ScoutHypothesis(
            hypothesis_id="metric_001",
            category="metric",
            claim="Resolve the official metric and prediction semantics.",
            rationale="Metric semantics determine validation and prediction format.",
            expected_eda_checks=["metric_analyzer.registry"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["doc_metric"],
        ),
        ScoutHypothesis(
            hypothesis_id="val_001",
            category="validation",
            claim="Select a primary validation policy from task and data evidence.",
            rationale="Validation must match task structure without assuming time splits.",
            expected_eda_checks=["validation_analyzer.policy"],
            priority="P0",
            confidence_before_eda="medium",
        ),
        ScoutHypothesis(
            hypothesis_id="leak_001",
            category="leakage",
            claim="Check generic train/test leakage risks before modeling.",
            rationale="Leakage can invalidate all local validation results.",
            expected_eda_checks=["leakage_checker.basic"],
            priority="P0",
            confidence_before_eda="medium",
        ),
        ScoutHypothesis(
            hypothesis_id="drift_001",
            category="drift",
            claim="Measure train/test drift as diagnostic evidence when test data exists.",
            rationale="Drift can inform risk assessment without overriding validation by default.",
            expected_eda_checks=["drift_analyzer.generic"],
            priority="P1",
            confidence_before_eda="low",
        ),
    ]
    tasks = [
        ScoutEdaTask(
            task_id="schema_001",
            module="schema_inferer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["schema_001"],
        ),
        ScoutEdaTask(
            task_id="metric_001",
            module="metric_analyzer",
            priority="P0",
            related_hypothesis_ids=["metric_001"],
        ),
        ScoutEdaTask(
            task_id="validation_001",
            module="validation_analyzer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["val_001"],
        ),
        ScoutEdaTask(
            task_id="leakage_001",
            module="leakage_checker",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["leak_001"],
        ),
        ScoutEdaTask(
            task_id="drift_001",
            module="drift_analyzer",
            priority="P1",
            related_hypothesis_ids=["drift_001"],
        ),
    ]
    task_plan = EdaTaskPlanDraft(
        competition_id="generic_tabular",
        task_type="binary_classification",
        metric={"name": "roc_auc"},
        dataset={"download_required": False},
        eda_tasks=tasks,
        hypothesis_index={
            "schema_001": ["schema_001"],
            "metric_001": ["metric_001"],
            "val_001": ["validation_001"],
            "leak_001": ["leakage_001"],
            "drift_001": ["drift_001"],
        },
        recommended_module_sequence=[
            "file_inventory",
            "schema_inferer",
            "table_profiler",
            "metric_analyzer",
            "validation_analyzer",
            "leakage_checker",
            "drift_analyzer",
        ],
        recommended_human_checklist=[
            "Confirm the metric and submission format from competition docs."
        ],
        blocking_tasks=["schema_inferer", "validation_analyzer", "leakage_checker"],
    )
    return ResearchScoutOutput(
        competition_id="generic_tabular",
        task_type="binary_classification",
        metric={"name": "roc_auc"},
        dataset={"download_required": False},
        hypotheses=hypotheses,
        eda_task_plan=task_plan,
        structured_findings=[
            ScoutStructuredFinding(
                finding_id="finding_001",
                category="metric",
                finding="Sources mention ROC AUC as the metric.",
                evidence_refs=["metric_001"],
                source_refs=["doc_metric"],
            )
        ],
        scout_limitations=[
            ScoutLimitation(
                limitation_id="lim_001",
                description="Scout has not inspected local train/test data.",
                affected_outputs=["validation"],
            )
        ],
        models_used={"research_scout": "mock"},
    )
