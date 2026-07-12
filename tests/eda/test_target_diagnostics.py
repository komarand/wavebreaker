from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.target_diagnostics import diagnose_target
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaTaskPlan, InferredSchema
from kaggle_researcher.eda.summary import build_eda_summary


def test_binary_balanced_target_reports_distribution_and_stratification(tmp_path: Path) -> None:
    rows = [
        f"{idx},{1 if idx < 40 else 0},{idx % 7},seg{idx % 3}"
        for idx in range(100)
    ]
    diagnostics = _diagnostics(tmp_path, rows, task_type="binary_classification", metric_name="roc_auc")

    assert diagnostics["status"] == "completed"
    assert diagnostics["distribution"]["target_type"] == "binary"
    assert diagnostics["imbalance"]["severity"] in {"none", "mild"}
    assert any(
        item["implication"] == "stratification_required"
        for item in diagnostics["validation_implications"]
    )
    assert not any(
        item["pattern"] == "very_rare_target_class"
        for item in diagnostics["suspicious_patterns"]
    )


def test_binary_severe_imbalance_recommends_minority_checks(tmp_path: Path) -> None:
    rows = [
        f"{idx},{1 if idx < 2 else 0},{idx % 5},seg{idx % 4}"
        for idx in range(100)
    ]
    diagnostics = _diagnostics(tmp_path, rows, task_type="binary_classification", metric_name="accuracy")

    assert diagnostics["imbalance"]["severity"] in {"severe", "extreme"}
    assert any(
        "minority-class" in action["action"]
        for action in diagnostics["recommended_actions"]
    )
    assert any(
        item["implication"] == "fold_class_count_checks_required"
        for item in diagnostics["validation_implications"]
    )


def test_multiclass_rare_class_lists_rare_class(tmp_path: Path) -> None:
    labels = ["rare"] * 3 + ["a"] * 45 + ["b"] * 52
    rows = [f"{idx},{label},{idx % 11},seg{idx % 5}" for idx, label in enumerate(labels)]
    diagnostics = _diagnostics(tmp_path, rows, task_type="multiclass_classification", metric_name="accuracy")

    rare_classes = diagnostics["distribution"]["rare_classes"]
    assert any(item["class"] == "rare" for item in rare_classes)
    assert any(
        item["implication"] == "rare_class_fold_coverage_check"
        for item in diagnostics["validation_implications"]
    )


def test_regression_heavy_tail_emits_transform_hypothesis(tmp_path: Path) -> None:
    targets = [float(idx) for idx in range(1, 80)] + [5000.0, 10000.0]
    rows = [f"{idx},{target},{idx % 13},seg{idx % 4}" for idx, target in enumerate(targets)]
    diagnostics = _diagnostics(tmp_path, rows, task_type="regression", metric_name="rmse")

    assert diagnostics["distribution"]["target_type"] == "regression"
    assert diagnostics["distribution"]["heavy_tail"] is True
    hints = diagnostics["distribution"]["target_transform_hints"]
    assert any(item["hint"] == "test_log_or_boxcox_like_transform" for item in hints)
    assert "hypothesis" in hints[0]["why"]


def test_missing_target_values_emit_suspicious_pattern(tmp_path: Path) -> None:
    rows = [
        f"{idx},{'' if idx < 5 else idx % 2},{idx % 9},seg{idx % 3}"
        for idx in range(50)
    ]
    diagnostics = _diagnostics(tmp_path, rows, task_type="binary_classification", metric_name="roc_auc")

    assert diagnostics["distribution"]["missing_target_count"] == 5
    assert any(
        item["pattern"] == "missing_target_values"
        for item in diagnostics["suspicious_patterns"]
    )


def test_target_by_missingness_recommends_indicators(tmp_path: Path) -> None:
    rows = []
    for idx in range(80):
        target = 1 if idx < 20 else 0
        missing_feature = "" if idx < 20 else idx
        rows.append(f"{idx},{target},{missing_feature},seg{idx % 4}")
    diagnostics = _diagnostics(tmp_path, rows, task_type="binary_classification", metric_name="roc_auc")

    missingness = diagnostics["target_by_missingness"]
    assert missingness
    assert missingness[0]["column"] == "feature"
    assert missingness[0]["absolute_difference"] >= 0.5
    assert any("missingness indicators" in action["action"] for action in diagnostics["recommended_actions"])


def test_high_cardinality_categorical_target_signal_is_caution(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 7},code_{idx}"
        for idx in range(60)
    ]
    diagnostics = _diagnostics(tmp_path, rows, task_type="binary_classification", metric_name="roc_auc")

    categorical = diagnostics["target_by_feature"]["categorical"]
    assert categorical
    assert categorical[0]["reliability"] in {"caution_high_cardinality", "caution_sparse_categories"}


def test_time_target_pattern_is_diagnostic_not_primary_override(tmp_path: Path) -> None:
    rows = []
    for idx in range(80):
        month = "2024-01" if idx < 40 else "2024-02"
        target = 0 if idx < 40 else 1
        rows.append(f"{idx},{target},{idx % 5},seg{idx % 3},{month}-{(idx % 28) + 1:02d}")
    diagnostics, payload = _diagnostics_with_payload(
        tmp_path,
        rows,
        task_type="binary_classification",
        metric_name="roc_auc",
        header="row_id,target,feature,category,event_date",
    )

    assert diagnostics["target_by_time"]["status"] == "computed"
    assert any(
        item["implication"] == "temporal_target_pattern_diagnostic"
        for item in diagnostics["validation_implications"]
    )
    assert payload["validation"].primary_validation["method"] == "stratified_kfold"


def test_no_target_available_returns_not_testable(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        ["1,10,a", "2,20,b"],
        header="row_id,feature,category",
        sample_header="row_id,prediction",
    )
    reader = DatasetReader(tmp_path)
    diagnostics = diagnose_target(
        InferredSchema(
            train_base_table="train.csv",
            test_base_table="test.csv",
            target_column=None,
            confidence="medium",
        ),
        {"task_type": "binary_classification", "metric_name": "roc_auc"},
        {},
        {},
        [],
        reader,
    )

    assert diagnostics["status"] == "not_testable"
    assert any("Target column is not inferred" in item for item in diagnostics["limitations"])


def test_summary_renders_concise_target_diagnostics(tmp_path: Path) -> None:
    rows = [
        f"{idx},{1 if idx < 30 else 0},{idx % 7},seg{idx % 3}"
        for idx in range(80)
    ]
    diagnostics = _diagnostics(tmp_path, rows, task_type="binary_classification", metric_name="roc_auc")
    summary = build_eda_summary(
        EdaEvidencePack(
            competition_id="target_summary",
            created_at="2026-07-11T12:00:00+03:00",
            run_id="target_summary_run",
            target_diagnostics=diagnostics,
        )
    )

    assert "## Target diagnostics" in summary
    assert "Target type: `binary`" in summary
    assert "Class balance" in summary
    assert "top_categories" not in summary


def test_target_strategy_hints_dedup_with_recommended_actions(tmp_path: Path) -> None:
    rows = []
    for idx in range(80):
        target = 1 if idx < 20 else 0
        missing_feature = "" if idx < 20 else idx
        rows.append(f"{idx},{target},{missing_feature},seg{idx % 4}")
    diagnostics, payload = _diagnostics_with_payload(tmp_path, rows, task_type="binary_classification", metric_name="f1")
    hints = build_eda_strategy_hints(
        {
            "validation_evidence": payload["validation"],
            "metric_evidence": payload["metric"],
            "feature_diagnostics": {},
            "target_diagnostics": diagnostics,
            "leakage_evidence": [],
        }
    )
    actions = build_recommended_next_actions(
        {
            "validation_evidence": payload["validation"],
            "metric_evidence": payload["metric"],
            "eda_strategy_hints": hints,
            "target_diagnostics": diagnostics,
        }
    )

    assert sum("threshold" in action.action.lower() for action in actions) == 1
    assert sum("missingness indicators" in action.action.lower() for action in actions) == 1


def _diagnostics(
    tmp_path: Path,
    rows: list[str],
    *,
    task_type: str,
    metric_name: str,
    header: str = "row_id,target,feature,category",
) -> dict:
    diagnostics, _ = _diagnostics_with_payload(
        tmp_path,
        rows,
        task_type=task_type,
        metric_name=metric_name,
        header=header,
    )
    return diagnostics


def _diagnostics_with_payload(
    tmp_path: Path,
    rows: list[str],
    *,
    task_type: str,
    metric_name: str,
    header: str = "row_id,target,feature,category",
) -> tuple[dict, dict]:
    _write_dataset(tmp_path, rows, header=header)
    reader = DatasetReader(tmp_path)
    inventory = build_file_inventory(tmp_path)
    schema = infer_schema(inventory, reader, task_type_hint=task_type, metric_hint=metric_name)
    profiles = profile_tables(inventory, schema, reader, sample_rows=1000)
    metric = analyze_metric(EdaTaskPlan(competition_id="target_diag", task_type=task_type, metric={"name": metric_name}), schema, profiles)
    validation = analyze_validation(schema, profiles, metric, reader)
    diagnostics = diagnose_target(schema, metric, validation, {}, profiles, reader, max_rows=1000)
    return diagnostics, {
        "schema": schema,
        "profiles": profiles,
        "metric": metric,
        "validation": validation,
    }


def _write_dataset(
    tmp_path: Path,
    rows: list[str],
    *,
    header: str,
    sample_header: str = "row_id,target",
) -> None:
    (tmp_path / "train.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    train_columns = header.split(",")
    test_header = ",".join(column for column in train_columns if column != "target")
    test_rows = []
    for index, row in enumerate(rows[:10], start=1000):
        values = dict(zip(train_columns, row.split(","), strict=False))
        values["row_id"] = str(index)
        test_rows.append(",".join(values[column] for column in test_header.split(",")))
    (tmp_path / "test.csv").write_text(
        test_header + "\n" + "\n".join(test_rows) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        sample_header + "\n1000,0\n",
        encoding="utf-8",
    )
