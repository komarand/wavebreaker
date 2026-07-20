from __future__ import annotations

import json
from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.baseline_ablations import (
    _add_comparisons,
    _best_ablation_summary,
    _feature_block_findings,
    _has_effective_feature_change,
    classify_ablation_materiality,
    run_baseline_ablations,
)
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.risk_register import build_eda_risk_register
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaTaskPlan, InferredSchema, MetricEvidence, ValidationEvidence
from kaggle_researcher.eda.summary import build_eda_summary


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_baseline_ablations_binary_fixture_runs_fold_safe_blocks(tmp_path: Path) -> None:
    reader, schema, validation, metric = _binary_ablation_fixture(tmp_path)

    evidence = run_baseline_ablations(
        schema,
        validation,
        metric,
        [],
        reader,
        tmp_path / "ablations",
        max_rows=1000,
    )

    assert evidence["status"] == "completed"
    completed = [item for item in evidence["ablations"] if item["status"] == "completed"]
    assert len(completed) >= 2
    assert evidence["fold_policy"]["same_folds_across_ablations"] is True
    for ablation in completed:
        assert "target" not in ablation["feature_columns"]
        assert "id" not in ablation["feature_columns"]
        assert ablation["preprocessing_policy"]["fit_scope"] == "inside_cv_folds"
        assert ablation["preprocessing_policy"]["uses_target_encoding"] is False
        assert ablation["preprocessing_policy"]["uses_test_labels"] is False
        assert ablation["delta_direction"] == "positive_is_better"


def test_baseline_ablations_skip_empty_categorical_blocks(tmp_path: Path) -> None:
    (tmp_path / "train_base.csv").write_text(
        "id,target,num_a,num_b\n"
        "1,0,0.1,1.0\n2,1,0.9,2.0\n3,0,0.2,1.1\n4,1,0.8,2.2\n"
        "5,0,0.3,1.2\n6,1,0.7,2.4\n",
        encoding="utf-8",
    )
    reader = DatasetReader(tmp_path)
    schema = InferredSchema(target_column="target", primary_id_column="id", train_base_table="train_base.csv", confidence="high")
    validation = ValidationEvidence(primary_validation={"method": "stratified_kfold"})
    metric = MetricEvidence(metric_name="accuracy", task_type="binary_classification", metric_family="threshold_classification", local_metric_available=True)

    evidence = run_baseline_ablations(schema, validation, metric, [], reader, tmp_path / "ablations")

    categorical = next(item for item in evidence["ablations"] if item["ablation_id"] == "abl_002_numeric_low_card_cat")
    assert categorical["status"] == "skipped"
    assert "low_cardinality_categorical" in categorical["reason"]


def test_missingness_and_high_cardinality_blocks_are_recorded(tmp_path: Path) -> None:
    reader, schema, validation, metric = _binary_ablation_fixture(tmp_path)

    evidence = run_baseline_ablations(schema, validation, metric, [], reader, tmp_path / "ablations")

    missingness = next(item for item in evidence["ablations"] if item["ablation_id"] == "abl_003_numeric_low_card_cat_missingness")
    high_card = next(item for item in evidence["ablations"] if item["ablation_id"] == "abl_004_add_high_cardinality")
    assert missingness["generated_feature_columns"]
    assert any(name.startswith("__missing_") for name in missingness["generated_feature_columns"])
    assert high_card["status"] == "completed"
    assert high_card["preprocessing_policy"]["uses_target_encoding"] is False
    assert high_card["warnings"]
    assert any(item["feature_block"] == "missingness_indicators" for item in evidence["feature_block_findings"])
    assert any(item["feature_block"] == "high_cardinality_categorical" for item in evidence["feature_block_findings"])


def test_lower_is_better_metric_deltas_are_positive_for_improvement(tmp_path: Path) -> None:
    schema, validation, metric, reader = _fixture_inputs("regression_tiny")

    evidence = run_baseline_ablations(schema, validation, metric, [], reader, tmp_path / "ablations")

    assert evidence["status"] == "completed"
    reference_value = evidence["baseline_reference"]["metric_value"]
    for ablation in evidence["ablations"]:
        if ablation["status"] != "completed":
            continue
        assert ablation["delta_direction"] == "positive_is_better"
        if ablation["metric_value"] < reference_value:
            assert ablation["delta_vs_reference"] > 0


def test_predictive_low_cardinality_categorical_is_helped(tmp_path: Path) -> None:
    reader, schema, validation, metric = _predictive_categorical_fixture(tmp_path)

    evidence = run_baseline_ablations(schema, validation, metric, [], reader, tmp_path / "ablations")

    finding = next(item for item in evidence["feature_block_findings"] if item["feature_block"] == "low_cardinality_categorical")
    assert finding["status"] == "helped"
    assert finding["delta_metric"] > 0


def test_paired_fold_comparison_uses_oriented_deltas_and_best_prior() -> None:
    metric = MetricEvidence(metric_name="accuracy", metric_family="threshold_classification", greater_is_better=True)
    reference = _synthetic_ablation("abl_001_safe_numeric", [0.70, 0.70, 0.70, 0.70, 0.70], 1)
    stronger = _synthetic_ablation("abl_002_numeric_low_card_cat", [0.80, 0.80, 0.80, 0.80, 0.80], 2)
    composite = _synthetic_ablation("abl_006_all_safe_features", [0.802, 0.802, 0.802, 0.802, 0.802], 4, kind="composite_configuration")
    completed = [reference, stronger, composite]

    _add_comparisons(completed, metric)

    assert composite["delta_vs_reference"] == 0.102
    assert composite["delta_vs_previous"] == 0.002
    assert composite["delta_vs_best_prior"] == 0.002
    assert composite["comparison_reference_ids"]["best_prior_ablation_id"] == stronger["ablation_id"]
    assert composite["fold_comparison"]["fold_deltas"] == [0.002] * 5
    assert composite["fold_wins"] + composite["fold_losses"] + composite["fold_ties"] == 5
    assert composite["materiality"] == "small"


def test_lower_is_better_paired_fold_comparison_is_positive_for_improvement() -> None:
    metric = MetricEvidence(metric_name="rmse", metric_family="regression", greater_is_better=False)
    reference = _synthetic_ablation("abl_001_safe_numeric", [1.0, 1.0, 1.0, 1.0, 1.0], 1)
    candidate = _synthetic_ablation("abl_002_numeric_low_card_cat", [0.98, 0.97, 0.99, 0.98, 0.97], 2)
    _add_comparisons([reference, candidate], metric)

    assert candidate["delta_vs_reference"] > 0
    assert candidate["fold_comparison"]["fold_wins"] == 5
    assert candidate["materiality"] == "material"


def test_stability_materiality_and_configuration_findings_distinguish_noise() -> None:
    metric = MetricEvidence(metric_name="accuracy", metric_family="threshold_classification", greater_is_better=True)
    reference = _synthetic_ablation("abl_001_safe_numeric", [0.70] * 5, 1)
    stable = _synthetic_ablation("abl_002_numeric_low_card_cat", [0.71] * 5, 2)
    noisy = _synthetic_ablation("abl_003_numeric_low_card_cat_missingness", [0.718, 0.708, 0.718, 0.708, 0.718], 3)
    degraded = _synthetic_ablation("abl_004_add_high_cardinality", [0.69] * 5, 4)
    composite = _synthetic_ablation("abl_006_all_safe_features", [0.7105] * 5, 6, kind="composite_configuration")
    completed = [reference, stable, noisy, degraded, composite]
    _add_comparisons(completed, metric)
    for item in completed:
        item["is_best_overall"] = item is composite

    findings = _feature_block_findings(completed, metric)
    stable_finding = next(item for item in findings if item.get("feature_block") == "low_cardinality_categorical")
    noisy_finding = next(item for item in findings if item.get("feature_block") == "missingness_indicators")
    degraded_finding = next(item for item in findings if item.get("feature_block") == "high_cardinality_categorical")
    configuration = next(item for item in findings if item.get("finding_type") == "configuration")

    assert stable_finding["status"] == "helped"
    assert stable_finding["materiality"] == "material"
    assert stable_finding["stability"] == "stable"
    assert noisy_finding["status"] == "unstable"
    assert degraded_finding["status"] == "hurt"
    assert configuration["configuration"] == "all_safe_features"
    assert not any(item.get("feature_block") == "all_safe_features" for item in findings)


def test_best_ablation_marks_simpler_competitor_when_marginal_gain_is_small() -> None:
    metric = MetricEvidence(metric_name="accuracy", metric_family="threshold_classification", greater_is_better=True)
    reference = _synthetic_ablation("abl_001_safe_numeric", [0.70] * 5, 1)
    simpler = _synthetic_ablation("abl_002_numeric_low_card_cat", [0.80] * 5, 2)
    best = _synthetic_ablation("abl_006_all_safe_features", [0.802] * 5, 4, kind="composite_configuration")
    completed = [reference, simpler, best]
    _add_comparisons(completed, metric)
    summary = _best_ablation_summary(best, completed, metric)

    assert summary["simpler_competitive_ablation_id"] == simpler["ablation_id"]
    assert summary["is_materially_better_than_simpler_prior"] is False


def test_materiality_thresholds_are_metric_aware() -> None:
    assert classify_ablation_materiality(metric_name="accuracy", metric_family="threshold_classification", greater_is_better=True, delta=0.001, reference_metric=0.7, fold_delta_std=0.0) == "negligible"
    assert classify_ablation_materiality(metric_name="rmse", metric_family="regression", greater_is_better=False, delta=0.003, reference_metric=1.0, fold_delta_std=0.0) == "small"


def test_no_effective_feature_change_is_detected_without_false_neutral_result() -> None:
    previous = {"raw_feature_columns": ["num"], "generated_feature_columns": ["__missing_num"]}
    candidate = {"raw_feature_columns": ["num"], "generated_feature_columns": ["__missing_num"]}

    assert _has_effective_feature_change(candidate, previous) is False
    assert candidate["effective_feature_change"] is False
    assert candidate["added_raw_columns"] == []


def _synthetic_ablation(
    ablation_id: str,
    scores: list[float],
    n_features: int,
    *,
    kind: str = "atomic_increment",
) -> dict[str, object]:
    return {
        "ablation_id": ablation_id,
        "status": "completed",
        "ablation_kind": kind,
        "metric_value": round(sum(scores) / len(scores), 6),
        "fold_results": [{"fold": index, "metric_value": value} for index, value in enumerate(scores)],
        "complexity_assessment": {
            "n_effective_features": n_features,
            "effective_feature_change": True,
        },
    }


def test_baseline_ablation_summary_hints_and_risks_are_concise() -> None:
    evidence = {
        "status": "completed",
        "metric_name": "accuracy",
        "baseline_reference": {"ablation_id": "abl_001_safe_numeric", "metric_value": 0.6},
        "best_ablation": {
            "ablation_id": "abl_002_numeric_low_card_cat",
            "metric_value": 0.8,
            "simpler_competitive_ablation_id": "abl_001_safe_numeric",
            "simpler_competitive_metric_value": 0.798,
            "delta_vs_simpler_competitive": 0.002,
            "materiality_vs_simpler_competitive": "small",
            "stability_vs_best_prior": "mixed",
        },
        "feature_block_findings": [
            {"finding_type": "feature_block", "feature_block": "low_cardinality_categorical", "status": "helped", "delta_metric": 0.2, "materiality": "material", "stability": "stable"},
            {"finding_type": "feature_block", "feature_block": "high_cardinality_categorical", "status": "unstable", "delta_metric": 0.0, "materiality": "small", "stability": "unstable"},
            {"finding_type": "feature_block", "feature_block": "missingness_indicators", "status": "helped", "delta_metric": 0.03, "materiality": "material", "stability": "stable"},
            {"finding_type": "configuration", "configuration": "all_safe_features", "status": "competitive", "materiality_vs_best_prior": "negligible", "stability_vs_best_prior": "mixed", "recommendation": "Prefer the simpler competitive configuration initially."},
        ],
        "limitations": ["Baseline ablations are lightweight sanity checks, not hyperparameter tuning."],
    }
    summary = build_eda_summary(
        EdaEvidencePack(
            competition_id="abl_summary",
            created_at="2026-07-11T12:00:00+03:00",
            run_id="abl_summary_run",
            baseline_ablation_evidence=evidence,
        )
    )
    hints = build_eda_strategy_hints({"baseline_ablation_evidence": evidence})
    risks = build_eda_risk_register(
        inferred_schema={},
        metric_evidence={},
        validation_evidence={},
        target_diagnostics={},
        leakage_evidence=[],
        drift_evidence={},
        relationship_evidence={},
        feature_probe_evidence=[],
        feature_diagnostics={},
        baseline_evidence={"status": "completed", "preprocessing_policy": {"safety_checks": {"fits_preprocessing_inside_folds": True}}},
        baseline_ablation_evidence=evidence,
        notebook_static_analysis={},
    )

    assert "## Baseline ablations" in summary
    assert "Feature block findings" in summary
    assert "Simpler competitive ablation" in summary
    assert "Best-vs-simpler delta" in summary
    assert "Best-vs-prior fold stability" in summary
    assert "fold_results" not in summary
    flattened = [item for group in hints.values() for item in group]
    assert any("Prioritize the feature block" in item["action"] for item in flattened)
    assert any("Retest the feature block" in item["action"] for item in flattened)
    assert any(risk["risk_type"] == "missingness" and "model-relevant" in risk["title"] for risk in risks)
    assert any(risk["risk_type"] == "high_cardinality" for risk in risks)


def _fixture_inputs(
    fixture_name: str,
) -> tuple[InferredSchema, ValidationEvidence, MetricEvidence, DatasetReader]:
    fixture_dir = FIXTURE_ROOT / fixture_name
    inventory = build_file_inventory(fixture_dir)
    reader = DatasetReader(fixture_dir)
    schema = infer_schema(inventory, reader)
    profiles = profile_tables(inventory, schema, reader)
    task_plan = EdaTaskPlan(
        **json.loads((fixture_dir / "eda_task_plan.json").read_text(encoding="utf-8"))
    )
    metric = analyze_metric(task_plan, schema, profiles)
    validation = analyze_validation(schema, profiles, metric, reader)
    return schema, validation, metric, reader


def _binary_ablation_fixture(tmp_path: Path) -> tuple[DatasetReader, InferredSchema, ValidationEvidence, MetricEvidence]:
    rows = ["id,target,num,cat,missing_feature,rare_code,text_note"]
    for index in range(30):
        target = index % 2
        cat = "yes" if target else "no"
        missing = "" if target and index % 4 == 1 else str(index % 3)
        rare = f"code_{index:03d}"
        text = "positive text value" if target else "negative text value"
        rows.append(f"{index},{target},{index % 5},{cat},{missing},{rare},{text}")
    (tmp_path / "train_base.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return (
        DatasetReader(tmp_path),
        InferredSchema(target_column="target", primary_id_column="id", train_base_table="train_base.csv", confidence="high"),
        ValidationEvidence(primary_validation={"method": "stratified_kfold"}),
        MetricEvidence(metric_name="accuracy", task_type="binary_classification", metric_family="threshold_classification", local_metric_available=True),
    )


def _predictive_categorical_fixture(tmp_path: Path) -> tuple[DatasetReader, InferredSchema, ValidationEvidence, MetricEvidence]:
    rows = ["id,target,num_noise,cat_signal"]
    for index in range(18):
        target = index % 2
        rows.append(f"{index},{target},0,{ 'B' if target else 'A'}")
    (tmp_path / "train_base.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return (
        DatasetReader(tmp_path),
        InferredSchema(target_column="target", primary_id_column="id", train_base_table="train_base.csv", confidence="high"),
        ValidationEvidence(primary_validation={"method": "stratified_kfold"}),
        MetricEvidence(metric_name="accuracy", task_type="binary_classification", metric_family="threshold_classification", local_metric_available=True),
    )
