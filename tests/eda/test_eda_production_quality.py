from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.drift_analyzer import analyze_drift
from kaggle_researcher.eda.modules.feature_diagnostics import diagnose_features
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.relationship_inferer import infer_relationships
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaTaskPlan
from kaggle_researcher.eda.summary import build_eda_summary


def test_id_artifact_is_excluded_from_feature_drift_severity(tmp_path: Path) -> None:
    _write_id_only_drift_fixture(tmp_path)
    payload = _evidence(tmp_path)

    drift = analyze_drift(payload["schema"], payload["validation"], payload["reader"])

    assert "row_id" not in drift["safe_feature_columns"]
    assert {"column": "row_id", "reason": "primary_id"} in drift["excluded_columns"]
    assert drift["id_artifact_drift"]["status"] == "computed"
    assert drift["feature_drift_severity"] != "high"
    assert "row_id" not in drift["numeric_psi"]["columns"]


def test_zero_coverage_relationship_is_rejected_not_one_to_one(tmp_path: Path) -> None:
    _write_tabular_fixture(tmp_path)
    (tmp_path / "train_extra.csv").write_text(
        "row_id,extra_value\n100,1\n101,2\n",
        encoding="utf-8",
    )
    payload = _evidence(tmp_path)

    relationships = infer_relationships(payload["schema"], payload["inventory"], payload["reader"])

    assert not any(
        item.get("table") == "train_extra.csv" and item.get("relationship_type") == "one_to_one"
        for item in relationships["relationships"]
    )
    assert any(
        item["right_table"] == "train_extra.csv" and "zero join coverage" in item["reason"]
        for item in relationships["rejected_relationships"]
    )


def test_feature_diagnostics_and_strategy_hints_are_generic_and_safe(tmp_path: Path) -> None:
    _write_tabular_fixture(tmp_path)
    payload = _evidence(tmp_path)
    drift = analyze_drift(payload["schema"], payload["validation"], payload["reader"])

    diagnostics = diagnose_features(
        payload["schema"],
        payload["profiles"],
        payload["metric"],
        drift,
        payload["reader"],
    )
    hints = build_eda_strategy_hints(
        {
            "validation_evidence": payload["validation"],
            "metric_evidence": payload["metric"],
            "drift_evidence": drift,
            "feature_diagnostics": diagnostics,
            "leakage_evidence": [],
            "baseline_evidence": {},
        }
    )

    safe_columns = set(diagnostics["safe_feature_columns"])
    assert "target" not in safe_columns
    assert "row_id" not in safe_columns
    assert diagnostics["numeric_feature_diagnostics"]["high_missingness"]
    assert diagnostics["categorical_feature_diagnostics"]["unseen_category_risks"]
    cat_high = next(
        item
        for item in diagnostics["categorical_feature_diagnostics"]["columns"]
        if item["column"] == "cat_high"
    )
    assert cat_high["target_association_reliability"] == "not_reliable"
    assert cat_high not in diagnostics["categorical_feature_diagnostics"][
        "high_target_association_candidates"
    ]
    count_zero = next(
        item
        for item in diagnostics["numeric_feature_diagnostics"]["columns"]
        if item["column"] == "count_zero"
    )
    assert count_zero["feature_numeric_kind"] == "count_zero_inflated"
    assert count_zero not in diagnostics["numeric_feature_diagnostics"]["outlier_heavy"]
    assert all(
        item["feature_numeric_kind"] == "continuous"
        for item in diagnostics["numeric_feature_diagnostics"]["outlier_heavy"]
    )
    assert diagnostics["missingness_diagnostics"]["recommended_indicators"]
    assert diagnostics["date_time_diagnostics"]["temporal_validation_signal"] == "diagnostic_only"
    assert hints["feature_engineering"]
    assert any("Do not use primary IDs" in item["action"] for item in hints["do_not_do"])


def test_summary_renders_final_roles_drift_artifacts_diagnostics_and_hints(tmp_path: Path) -> None:
    _write_tabular_fixture(tmp_path)
    payload = _evidence(tmp_path)
    drift = analyze_drift(payload["schema"], payload["validation"], payload["reader"])
    diagnostics = diagnose_features(
        payload["schema"],
        payload["profiles"],
        payload["metric"],
        drift,
        payload["reader"],
    )
    hints = build_eda_strategy_hints(
        {
            "validation_evidence": payload["validation"],
            "metric_evidence": payload["metric"],
            "drift_evidence": drift,
            "feature_diagnostics": diagnostics,
            "leakage_evidence": [],
            "baseline_evidence": {},
        }
    )
    pack = EdaEvidencePack(
        competition_id="generic_quality",
        created_at="2026-07-08T12:00:00+03:00",
        run_id="generic_quality_run",
        file_inventory={
            "files": [],
            "reconciled_table_roles": {
                "train.csv": "train_base",
                "test.csv": "test_base",
                "submission_template.csv": "sample_submission",
            },
        },
        inferred_schema=payload["schema"].model_dump(mode="json"),
        table_profiles=[profile.model_dump(mode="json") for profile in payload["profiles"]],
        metric_evidence=payload["metric"].model_dump(mode="json"),
        validation_evidence=payload["validation"].model_dump(mode="json"),
        drift_evidence=drift,
        feature_diagnostics=diagnostics,
        eda_strategy_hints=hints,
        warnings=[],
        limitations=[],
    )

    summary = build_eda_summary(pack)

    assert "Final table roles" in summary
    assert "sample_submission" in summary
    assert "Excluded drift artifacts" in summary
    assert "Feature diagnostics" in summary
    assert "Strategy hints" in summary


def _evidence(tmp_path: Path) -> dict:
    inventory = build_file_inventory(tmp_path)
    reader = DatasetReader(tmp_path)
    schema = infer_schema(inventory, reader, task_type_hint="binary_classification", metric_hint="roc_auc")
    profiles = profile_tables(inventory, schema, reader, sample_rows=1000)
    metric = analyze_metric(
        EdaTaskPlan(competition_id="generic_quality", task_type="binary_classification", metric={"name": "roc_auc"}),
        schema,
        profiles,
    )
    validation = analyze_validation(schema, profiles, metric, reader)
    return {
        "inventory": inventory,
        "reader": reader,
        "schema": schema,
        "profiles": profiles,
        "metric": metric,
        "validation": validation,
    }


def _write_tabular_fixture(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "row_id,target,stable_num,missing_num,outlier_num,count_zero,low_info,cat_low,cat_high,event_date\n"
        "1,0,10,,1,0,1,a,h001,2024-01-01\n"
        "2,1,11,5,2,0,1,b,h002,2024-01-02\n"
        "3,0,10,,3,1,1,a,h003,2024-01-03\n"
        "4,1,11,7,100,0,1,b,h004,2024-01-04\n",
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text(
        "row_id,stable_num,missing_num,outlier_num,count_zero,low_info,cat_low,cat_high,event_date\n"
        "100,10,,2,0,1,a,h900,2024-01-05\n"
        "101,11,,3,1,1,c,h901,2024-01-06\n",
        encoding="utf-8",
    )
    (tmp_path / "submission_template.csv").write_text(
        "row_id,target\n100,0\n101,0\n",
        encoding="utf-8",
    )


def _write_id_only_drift_fixture(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "row_id,target,stable_num,stable_cat\n"
        "1,0,10,a\n"
        "2,1,11,b\n"
        "3,0,10,a\n"
        "4,1,11,b\n",
        encoding="utf-8",
    )
    (tmp_path / "test.csv").write_text(
        "row_id,stable_num,stable_cat\n"
        "100,10,a\n"
        "101,11,b\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        "row_id,target\n100,0\n101,0\n",
        encoding="utf-8",
    )
