from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.schemas import EdaTaskPlan, ResearchHypotheses, competition_ids_match


FIXTURE_ROOT = Path("tests/fixtures/eda")
FIXTURE_DIR = FIXTURE_ROOT / "home_credit_tiny"
GENERIC_FIXTURES = ("iid_binary_tiny", "regression_tiny")
CSV_FILES = [
    "train_base.csv",
    "test_base.csv",
    "sample_submission.csv",
    "train_static_0.csv",
    "test_static_0.csv",
]


def test_home_credit_tiny_fixture_files_exist() -> None:
    expected_files = [
        *CSV_FILES,
        "research_hypotheses.json",
        "eda_task_plan.json",
    ]

    for filename in expected_files:
        assert (FIXTURE_DIR / filename).is_file()


def test_home_credit_tiny_input_json_validates_against_eda_schemas() -> None:
    hypotheses = ResearchHypotheses.model_validate_json(
        (FIXTURE_DIR / "research_hypotheses.json").read_text(encoding="utf-8")
    )
    task_plan = EdaTaskPlan.model_validate_json(
        (FIXTURE_DIR / "eda_task_plan.json").read_text(encoding="utf-8")
    )

    assert competition_ids_match(hypotheses, task_plan)
    assert {hypothesis.hypothesis_id for hypothesis in hypotheses.hypotheses} >= {
        "schema_001",
        "metric_001",
        "val_001",
        "leak_001",
    }
    assert [task.module for task in task_plan.eda_tasks] == [
        "file_inventory",
        "schema_inferer",
        "table_profiler",
        "metric_analyzer",
        "validation_analyzer",
        "leakage_checker",
    ]
    assert set(task_plan.blocking_tasks) == {
        "file_inventory",
        "schema_inferer",
        "validation_analyzer",
        "leakage_checker",
    }


def test_dataset_reader_can_read_every_fixture_table() -> None:
    reader = DatasetReader(FIXTURE_DIR)

    for filename in CSV_FILES:
        schema = reader.read_schema(filename)
        head = reader.file_head(filename, n_rows=3)
        count = reader.count_rows(filename)

        assert schema
        assert head.height > 0
        assert count is not None
        assert count > 0


def test_home_credit_tiny_fixture_core_data_invariants() -> None:
    reader = DatasetReader(FIXTURE_DIR)

    train_base = reader.file_head("train_base.csv", n_rows=100)
    test_base = reader.file_head("test_base.csv", n_rows=100)
    sample_submission = reader.file_head("sample_submission.csv", n_rows=100)

    assert "target" in train_base.columns
    assert "target" not in test_base.columns
    assert "case_id" in train_base.columns
    assert "case_id" in test_base.columns
    assert "case_id" in sample_submission.columns
    assert "WEEK_NUM" in train_base.columns
    assert "WEEK_NUM" in test_base.columns
    assert set(train_base["case_id"]).isdisjoint(set(test_base["case_id"]))
    assert sample_submission.columns == ["case_id", "score"]

    target_by_week = (
        train_base.group_by("WEEK_NUM")
        .agg(pl.col("target").mean().alias("target_rate"))
        .sort("WEEK_NUM")
    )
    assert target_by_week["target_rate"].to_list() == [0.0, 0.5, 0.5, 1.0]


def test_home_credit_tiny_json_files_are_plain_json_objects() -> None:
    for filename in ("research_hypotheses.json", "eda_task_plan.json"):
        payload = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))

        assert isinstance(payload, dict)
        assert payload["competition_id"] == "home_credit_tiny"


def test_generic_offline_fixtures_exist_and_validate() -> None:
    for fixture_name in GENERIC_FIXTURES:
        fixture_dir = FIXTURE_ROOT / fixture_name
        for filename in (
            "train_base.csv",
            "test_base.csv",
            "sample_submission.csv",
            "research_hypotheses.json",
            "eda_task_plan.json",
        ):
            assert (fixture_dir / filename).is_file()

        hypotheses = ResearchHypotheses.model_validate_json(
            (fixture_dir / "research_hypotheses.json").read_text(encoding="utf-8")
        )
        task_plan = EdaTaskPlan.model_validate_json(
            (fixture_dir / "eda_task_plan.json").read_text(encoding="utf-8")
        )
        assert competition_ids_match(hypotheses, task_plan)

        reader = DatasetReader(fixture_dir)
        assert reader.count_rows("train_base.csv") > 0
        assert reader.count_rows("test_base.csv") > 0
