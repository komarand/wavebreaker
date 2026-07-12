from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.feature_diagnostics import diagnose_features
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.interaction_diagnostics import diagnose_interactions
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.schemas import MetricEvidence


def test_role_safe_candidates_and_numeric_interaction_hypothesis(tmp_path: Path) -> None:
    rows = ["id,target,prediction,x,y,category"]
    for index in range(80):
        x = (index % 5 + 1) * (1 if index % 4 in {0, 1} else -1)
        y = (index % 7 + 1) * (1 if index % 4 in {0, 2} else -1)
        rows.append(f"{index},{int(x * y > 0)},0,{x},{y},{'a' if index % 2 else 'b'}")
    result = _diagnose(tmp_path, rows)

    assert result["status"] == "completed"
    excluded = {item["column"]: item["reason"] for item in result["excluded_columns"]}
    assert excluded["id"] == "primary_id"
    assert "target" not in result["candidate_columns"]
    pair = next(item for item in result["numeric_numeric"] if {item["left_column"], item["right_column"]} == {"x", "y"})
    assert pair["materiality"] == "material"
    assert pair["interaction_reliability"] == "reliable"
    assert result["interaction_hypotheses"]


def test_numeric_redundancy_and_deterministic_candidate_caps(tmp_path: Path) -> None:
    rows = ["id,target,a,b,c,d"]
    for index in range(60):
        rows.append(f"{index},{index % 2},{index},{index + 0.001},{index % 3},{index % 5}")
    redundancy = _diagnose(tmp_path, rows)
    first = _diagnose(tmp_path, rows, max_numeric_columns=2)
    second = _diagnose(tmp_path, rows, max_numeric_columns=2)

    assert first["candidate_selection"] == second["candidate_selection"]
    assert len(first["candidate_selection"]["numeric_columns"]) == 2
    assert first["candidate_selection"]["excluded_due_to_caps"]
    assert redundancy["redundancy_groups"]


def test_categorical_cross_and_missingness_are_cautious_experiment_hypotheses(tmp_path: Path) -> None:
    rows = ["id,target,cat_a,cat_b,missing_feature"]
    for index in range(80):
        a = "a" if index % 2 else "b"
        b = "x" if index % 4 < 2 else "y"
        target = int((a, b) in {("a", "x"), ("b", "y")} or index < 40)
        missing = "" if index < 40 else str(index)
        rows.append(f"{index},{target},{a},{b},{missing}")
    result = _diagnose(tmp_path, rows)

    cross = next(item for item in result["categorical_categorical"] if {item["left_column"], item["right_column"]} == {"cat_a", "cat_b"})
    assert cross["materiality"] in {"material", "small"}
    assert cross["suggested_experiment"]
    assert result["missingness_interactions"]
    assert any(item["interaction_type"] == "missingness" for item in result["interaction_hypotheses"])


def test_no_target_keeps_unsupervised_redundancy_without_target_hypotheses(tmp_path: Path) -> None:
    rows = ["id,a,b"] + [f"{index},{index},{index + 0.001}" for index in range(50)]
    (tmp_path / "train_base.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    reader = DatasetReader(tmp_path)
    inventory = build_file_inventory(tmp_path)
    schema = infer_schema(inventory, reader)
    schema.target_column = None
    profiles = profile_tables(inventory, schema, reader)
    metric = MetricEvidence(metric_name="unknown", task_type="unknown")

    result = diagnose_interactions(schema, profiles, metric, reader, min_group_rows=10)

    assert result["status"] == "completed"
    assert result["target_column"] is None
    assert result["redundancy_groups"]
    assert result["interaction_hypotheses"] == []


def _diagnose(tmp_path: Path, rows: list[str], *, max_numeric_columns: int = 30) -> dict:
    (tmp_path / "train_base.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    reader = DatasetReader(tmp_path)
    inventory = build_file_inventory(tmp_path)
    schema = infer_schema(inventory, reader)
    schema.target_column = "target"
    profiles = profile_tables(inventory, schema, reader)
    metric = MetricEvidence(metric_name="accuracy", task_type="binary_classification", metric_family="threshold_classification", greater_is_better=True)
    features = diagnose_features(schema, profiles, metric, {}, reader)
    return diagnose_interactions(
        schema, profiles, metric, reader, feature_diagnostics=features,
        max_rows=500, max_numeric_columns=max_numeric_columns, min_group_rows=10,
    )
