from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggle_researcher.eda.schemas import (
    DatasetFile,
    EdaEvidencePack,
    EdaRunConfig,
    EdaTask,
    EdaTaskPlan,
    FileInventoryResult,
    HypothesisResult,
    LeakageCheckResult,
    RecommendedNextAction,
    ResearchHypotheses,
    ResearchHypothesis,
    competition_ids_match,
)


def test_invalid_confidence_status_and_priority_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        ResearchHypothesis(
            hypothesis_id="schema_001",
            category="schema",
            claim="Dataset schema should identify train/test tables.",
            expected_eda_checks=["schema_inferer.detect_roles"],
            priority="P9",
            confidence_before_eda="medium",
        )

    with pytest.raises(ValidationError):
        HypothesisResult(
            hypothesis_id="val_001",
            category="validation",
            status="probably",
            confidence_after_eda="medium",
            finding="Temporal validation may be feasible.",
            evidence_refs=["validation_evidence.oot_holdout"],
            impact_on_strategy="Use validation evidence before choosing CV.",
        )

    with pytest.raises(ValidationError):
        LeakageCheckResult(
            check_id="leak_001",
            status="warning",
            severity="severe",
            finding="Target-like column name detected.",
        )

    with pytest.raises(ValidationError):
        ResearchHypothesis(
            hypothesis_id="metric_001",
            category="metric",
            claim="Metric is rank-based.",
            expected_eda_checks=["metric_analyzer.basic"],
            priority="P0",
            confidence_before_eda="certain",
        )


def test_mutable_defaults_are_independent_per_instance() -> None:
    first = ResearchHypotheses(competition_id="comp-a")
    second = ResearchHypotheses(competition_id="comp-b")

    first.scout_limitations.append("source coverage is limited")
    first.models_used["scout"] = "mock-model"

    assert second.scout_limitations == []
    assert second.models_used == {}

    first_inventory = FileInventoryResult(dataset_path="dataset-a")
    second_inventory = FileInventoryResult(dataset_path="dataset-b")
    first_inventory.files.append(
        DatasetFile(
            path="train.csv",
            name="train.csv",
            extension=".csv",
            size_bytes=100,
            size_mb=0.0001,
            role_hint="train",
            table_hint="base",
            can_read=True,
        )
    )

    assert second_inventory.files == []


def test_minimal_valid_eda_evidence_pack_validates() -> None:
    pack = EdaEvidencePack(
        competition_id="fixture_competition",
        created_at="2026-07-06T12:00:00+03:00",
        run_id="fixture_competition_20260706_120000",
        hypothesis_results=[
            HypothesisResult(
                hypothesis_id="schema_001",
                category="schema",
                status="confirmed",
                confidence_after_eda="high",
                finding="Train and test base tables are present.",
                evidence_refs=["file_inventory.files"],
                impact_on_strategy="Use detected base tables as the initial modeling surface.",
            )
        ],
        recommended_next_actions=[
            RecommendedNextAction(
                priority="P0",
                action="Verify schema roles before modeling.",
                why="The file inventory provides table role evidence.",
                evidence_refs=["file_inventory.files"],
            )
        ],
    )

    assert pack.schema_version == "1.0"
    assert pack.relationship_evidence == {}
    assert pack.feature_probe_evidence == []
    assert pack.hypothesis_results[0].evidence_refs == ["file_inventory.files"]


def test_empty_evidence_ref_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendedNextAction(
            priority="P0",
            action="Use out-of-time validation.",
            why="Temporal validation appears feasible.",
            evidence_refs=["validation_evidence.oot_holdout", "   "],
        )

    with pytest.raises(ValidationError):
        HypothesisResult(
            hypothesis_id="metric_001",
            category="metric",
            status="confirmed",
            confidence_after_eda="high",
            finding="Metric is rank-based.",
            evidence_refs=[""],
            impact_on_strategy="Use probabilities or ranks, not hard labels.",
        )


def test_eda_run_config_and_task_plan_validate_contract_fields() -> None:
    config = EdaRunConfig(
        competition_id="fixture_competition",
        hypotheses_path=Path("research_hypotheses.json"),
        task_plan_path=Path("eda_task_plan.json"),
        local_dataset_path=Path("dataset"),
        download_dataset=False,
        modules=["file_inventory"],
        skip_modules=["baseline_runner"],
    )

    assert config.download_dataset is False
    assert config.profile_sample_rows == 200_000

    task_plan = EdaTaskPlan(
        competition_id="fixture_competition",
        eda_tasks=[
            EdaTask(
                task_id="schema_001",
                module="schema_inferer",
                priority="P0",
                blocking=True,
                related_hypothesis_ids=["schema_001"],
            )
        ],
    )
    hypotheses = ResearchHypotheses(competition_id="fixture_competition")

    assert competition_ids_match(hypotheses, task_plan)
