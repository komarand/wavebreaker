from __future__ import annotations

import asyncio
from pathlib import Path

from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import (
    EdaEvidencePack,
    EdaRunConfig,
    HypothesisResult,
    RecommendedNextAction,
)
from kaggle_researcher.eda.summary import build_eda_summary


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_build_eda_summary_contains_required_sections() -> None:
    summary = build_eda_summary(_pack())

    for heading in (
        "## Dataset",
        "## Schema",
        "## Metric",
        "## Validation",
        "## Leakage",
        "## Relationships",
        "## Drift",
        "## Baseline",
        "## Feature probes",
        "## Hypothesis results",
        "## Recommended next actions",
        "## Warnings",
        "## Limitations",
    ):
        assert heading in summary

    assert "[refs: validation_evidence.primary_validation]" in summary
    assert "Temporal validation is diagnostic, not the selected primary validation." in summary
    assert "Temporal validation is required" not in summary
    assert "Use selected validation." in summary


def test_build_eda_summary_marks_absent_p1_modules_as_skipped() -> None:
    summary = build_eda_summary(
        _pack(
            relationship_evidence={},
            drift_evidence={},
            baseline_evidence={"status": "skipped", "reason": "Baseline disabled."},
            feature_probe_evidence=[],
            artifacts={
                "module_statuses": {
                    "relationship_inferer": "skipped",
                    "drift_analyzer": "skipped",
                    "feature_probe": "skipped",
                },
                "module_status_details": {
                    "relationship_inferer": {"status": "skipped"},
                    "drift_analyzer": {"status": "skipped"},
                    "feature_probe": {"status": "skipped"},
                },
            },
        )
    )

    assert "## Relationships\n- `skipped`." in summary
    assert "## Drift\n- `skipped`." in summary
    assert "## Baseline\n- Status: `skipped`" in summary
    assert "## Feature probes\n- `skipped`." in summary


def test_build_eda_summary_renders_feature_family_and_role_candidates() -> None:
    summary = build_eda_summary(
        _pack(
            feature_probe_evidence=[
                {
                    "feature_family": "base_numeric_features",
                    "status": "medium_potential",
                    "finding": "Numeric features vary.",
                }
            ],
        )
    )

    assert "`base_numeric_features`: `medium_potential`" in summary
    assert "unknown_family" not in summary


def test_orchestrator_writes_generic_summary(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs"))
    summary = result.summary_path.read_text(encoding="utf-8")

    assert "## Dataset" in summary
    assert "## Validation" in summary
    assert "Primary validation: `stratified_kfold`" in summary
    assert "Temporal validation is required" not in summary
    assert "## Recommended next actions" in summary


async def _run_fixture(competition_id: str, output_dir: Path):
    fixture_dir = FIXTURE_ROOT / competition_id
    return await run_eda(
        EdaRunConfig(
            competition_id=competition_id,
            hypotheses_path=fixture_dir / "research_hypotheses.json",
            task_plan_path=fixture_dir / "eda_task_plan.json",
            local_dataset_path=fixture_dir,
            output_dir=output_dir,
            download_dataset=False,
            profile_sample_rows=1000,
        )
    )


def _pack(
    *,
    relationship_evidence: dict | None = None,
    drift_evidence: dict | None = None,
    baseline_evidence: dict | None = None,
    feature_probe_evidence: list[dict] | None = None,
    artifacts: dict | None = None,
) -> EdaEvidencePack:
    return EdaEvidencePack(
        competition_id="summary_tiny",
        created_at="2026-07-08T12:00:00+03:00",
        run_id="summary_tiny_20260708_120000",
        dataset={"dataset_path": "tests/fixtures/eda/summary_tiny", "source": "local"},
        file_inventory={"files": [{"name": "train.csv"}, {"name": "test.csv"}]},
        inferred_schema={
            "target_column": "target",
            "primary_id_column": "row_id",
            "candidate_date_columns": ["event_date"],
            "tables": [{"table_name": "train_base"}],
            "global_roles": {
                "target_column_candidates": [{"name": "target"}, {"name": "label"}],
                "role_inference_warnings": ["Ambiguous target column candidates."],
            },
        },
        table_profiles=[{"table_name": "train_base", "sampled": True}],
        metric_evidence={
            "metric_name": "roc_auc",
            "task_type": "binary_classification",
            "metric_family": "classification",
            "prediction_output_type": "probability",
            "local_metric_available": True,
        },
        validation_evidence={
            "primary_validation": {"method": "stratified_kfold", "reason": "IID classification."},
            "diagnostic_validations": [{"method": "temporal_holdout", "status": "diagnostic"}],
            "evidence_refs": ["validation_evidence.primary_validation"],
            "reasoning_summary": "Selected stratified_kfold for ordinary iid classification.",
        },
        leakage_evidence=[
            {
                "check_id": "target_in_test",
                "status": "passed",
                "severity": "low",
                "finding": "Target is absent from test.",
                "evidence": {"target_column": "target"},
            }
        ],
        relationship_evidence=relationship_evidence
        if relationship_evidence is not None
        else {"status": "completed", "relationships": [{"left": "train", "right": "test"}]},
        drift_evidence=drift_evidence
        if drift_evidence is not None
        else {"status": "completed", "severity": "low", "auc": 0.52},
        baseline_evidence=baseline_evidence
        if baseline_evidence is not None
        else {"status": "completed", "metric_name": "roc_auc", "mean_score": 0.61},
        feature_probe_evidence=feature_probe_evidence
        if feature_probe_evidence is not None
        else [{"family": "numeric", "status": "medium_potential", "finding": "Numeric features vary."}],
        hypothesis_results=[
            HypothesisResult(
                hypothesis_id="val_001",
                category="validation",
                status="confirmed",
                confidence_after_eda="medium",
                finding="Validation policy selected.",
                evidence_refs=["validation_evidence.primary_validation"],
                impact_on_strategy="Use selected validation.",
            )
        ],
        recommended_next_actions=[
            RecommendedNextAction(
                priority="P0",
                action="Use selected validation.",
                why="The primary validation policy is explicit.",
                evidence_refs=["validation_evidence.primary_validation"],
            )
        ],
        warnings=["source coverage limited"],
        limitations=["tiny fixture"],
        artifacts=artifacts if artifacts is not None else {},
    )
