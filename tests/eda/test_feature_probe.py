from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.feature_probe import probe_feature_families
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.relationship_inferer import infer_relationships
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.presets import HOME_CREDIT_CRMS_PRESET
from kaggle_researcher.eda.schemas import EdaTaskPlan


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_secondary_aggregations_are_not_testable_before_relationship_evidence() -> None:
    schema, profiles, metric = _fixture_context(
        "home_credit_tiny",
        preset=HOME_CREDIT_CRMS_PRESET,
    )

    probes = probe_feature_families(schema, profiles, {}, [], {}, metric)
    secondary = _probe(probes, "secondary_table_aggregations")

    assert secondary["status"] == "not_testable"
    assert "relationship inference" in secondary["recommendation"]
    assert secondary["evidence"]["secondary_tables"]


def test_secondary_aggregations_gain_potential_with_relationship_evidence() -> None:
    fixture_dir = FIXTURE_ROOT / "home_credit_tiny"
    inventory = build_file_inventory(fixture_dir, preset=HOME_CREDIT_CRMS_PRESET)
    reader = DatasetReader(fixture_dir)
    schema = infer_schema(inventory, reader, preset=HOME_CREDIT_CRMS_PRESET)
    profiles = profile_tables(inventory, schema, reader)
    relationship_evidence = infer_relationships(schema, inventory, reader)
    task_plan = EdaTaskPlan(
        **json.loads((fixture_dir / "eda_task_plan.json").read_text(encoding="utf-8"))
    )
    metric = analyze_metric(task_plan, schema, profiles).model_dump(mode="json")

    probes = probe_feature_families(
        schema,
        profiles,
        relationship_evidence,
        [],
        {},
        metric,
    )
    secondary = _probe(probes, "secondary_table_aggregations")

    assert secondary["status"] in {"medium_potential", "high_potential"}
    assert secondary["evidence"]["usable_relationships"]
    assert any(item["join_key"] == "case_id" for item in secondary["evidence"]["usable_relationships"])


def test_target_encoding_is_high_leakage_risk_without_safe_oof_policy() -> None:
    schema, profiles, metric = _fixture_context("iid_binary_tiny")

    probes = probe_feature_families(schema, profiles, {}, [], {}, metric)
    target_encoding = _probe(probes, "target_encoding_or_woe")

    assert target_encoding["status"] == "unsafe"
    assert target_encoding["leakage_risk"] == "high"
    assert target_encoding["evidence"]["categorical_columns"]


def test_regression_outlier_fixture_recommends_target_transform_for_skewed_target() -> None:
    schema, profiles, metric = _fixture_context("regression_outliers_tiny")

    probes = probe_feature_families(schema, profiles, {}, [], {}, metric)
    transform = _probe(probes, "regression_target_transform")

    assert transform["status"] in {"medium_potential", "high_potential"}
    assert transform["evidence"]["target_column"] == "target"
    assert transform["evidence"]["skew_proxy"] >= 4.0
    assert "target transforms" in transform["recommendation"]


def test_ranking_group_features_are_only_relevant_for_ranking_tasks() -> None:
    rank_schema, rank_profiles, rank_metric = _fixture_context("ranking_tiny")
    binary_schema, binary_profiles, binary_metric = _fixture_context("iid_binary_tiny")

    rank_probe = _probe(
        probe_feature_families(rank_schema, rank_profiles, {}, [], {}, rank_metric),
        "ranking_group_features",
    )
    binary_probe = _probe(
        probe_feature_families(binary_schema, binary_profiles, {}, [], {}, binary_metric),
        "ranking_group_features",
    )

    assert rank_probe["status"] == "high_potential"
    assert "query_id" in rank_probe["evidence"]["group_columns"]
    assert binary_probe["status"] == "not_testable"


def _fixture_context(
    fixture_name: str,
    *,
    preset: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    fixture_dir = FIXTURE_ROOT / fixture_name
    inventory = build_file_inventory(fixture_dir, preset=preset)
    reader = DatasetReader(fixture_dir)
    schema = infer_schema(inventory, reader, preset=preset)
    profiles = profile_tables(inventory, schema, reader)
    task_plan = EdaTaskPlan(
        **json.loads((fixture_dir / "eda_task_plan.json").read_text(encoding="utf-8"))
    )
    metric = analyze_metric(task_plan, schema, profiles).model_dump(mode="json")
    return schema, profiles, metric


def _probe(probes: list[dict[str, Any]], feature_family: str) -> dict[str, Any]:
    return next(item for item in probes if item["feature_family"] == feature_family)
