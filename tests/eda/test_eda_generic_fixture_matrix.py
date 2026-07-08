from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig


FIXTURE_ROOT = Path("tests/fixtures/eda")
GENERIC_FIXTURES = (
    "iid_binary_tiny",
    "regression_outliers_tiny",
    "grouped_binary_tiny",
    "multiclass_tiny",
    "ranking_tiny",
    "leakage_target_in_test_tiny",
    "messy_schema_tiny",
)
TEMPORAL_PRIMARY_TOKENS = ("temporal", "oot", "expanding")


def test_home_credit_fixture_still_runs(tmp_path: Path) -> None:
    payload, result = _run_fixture("home_credit_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "home_credit_tiny")
    assert payload["validation_evidence"]["primary_validation"]["method"] == "temporal_holdout"


def test_iid_binary_selects_stratified_kfold_even_with_date_column(tmp_path: Path) -> None:
    payload, result = _run_fixture("iid_binary_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "iid_binary_tiny")
    primary = payload["validation_evidence"]["primary_validation"]["method"]
    assert "stratified_kfold" in primary
    assert not _contains_any(primary, TEMPORAL_PRIMARY_TOKENS)
    assert payload["metric_evidence"]["requires_probabilities"] is True
    assert payload["metric_evidence"]["prediction_output_type"] == "probability"
    assert _has_action(payload, "StratifiedKFold")
    assert _has_action(payload, "probabilities")

    train_profile = _profile(payload, "train_base.csv")
    assert _column(train_profile, "income")["max"] == 1200000
    assert _column(train_profile, "noisy_score")["min"] == -9.4
    _assert_json_has_no_non_finite_numbers(payload)


def test_regression_outliers_selects_kfold_and_profiles_outliers(tmp_path: Path) -> None:
    payload, result = _run_fixture("regression_outliers_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "regression_outliers_tiny")
    primary = payload["validation_evidence"]["primary_validation"]["method"]
    assert "kfold" in primary
    assert not _contains_any(primary, TEMPORAL_PRIMARY_TOKENS)
    assert payload["metric_evidence"]["metric_family"] == "regression_error"
    assert _has_action(payload, "regression loss")

    train_profile = _profile(payload, "train_base.csv")
    assert _column(train_profile, "target")["max"] == 2500000
    assert _column(train_profile, "sqft")["max"] == 18000
    assert _column(train_profile, "previous_price")["missing_count"] > 0
    _assert_json_has_no_non_finite_numbers(payload)


def test_grouped_binary_selects_group_aware_validation(tmp_path: Path) -> None:
    payload, result = _run_fixture("grouped_binary_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "grouped_binary_tiny")
    primary = payload["validation_evidence"]["primary_validation"]
    assert primary["method"] in {"group_kfold", "stratified_group_kfold"}
    assert primary["group_column"] == "customer_id"
    assert _has_action(payload, "grouped validation")

    group_overlap = _leakage_check(payload, "group_overlap")
    assert group_overlap["status"] == "warning"
    assert group_overlap["severity"] in {"medium", "high"}
    assoc = _leakage_check(payload, "numeric_target_association")
    checked = assoc.get("evidence", {}).get("checked_columns", [])
    assert "row_id" not in checked
    assert "customer_id" not in checked


def test_multiclass_logloss_runs_and_uses_probabilistic_metric(tmp_path: Path) -> None:
    payload, result = _run_fixture("multiclass_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "multiclass_tiny")
    assert payload["metric_evidence"]["metric_family"] == "probabilistic_classification"
    assert payload["metric_evidence"]["requires_probabilities"] is True
    assert payload["validation_evidence"]["primary_validation"]["method"] == "stratified_kfold"
    assert payload["inferred_schema"]["global_roles"]["prediction_columns"] == [
        "class_0",
        "class_1",
        "class_2",
    ]


def test_ranking_metric_selects_query_group_validation(tmp_path: Path) -> None:
    payload, result = _run_fixture("ranking_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "ranking_tiny")
    assert payload["metric_evidence"]["metric_family"] == "ranking"
    assert payload["metric_evidence"]["requires_query_groups"] is True
    primary = payload["validation_evidence"]["primary_validation"]
    assert primary["method"] in {"ranking_group_cv", "group_kfold"}
    assert primary.get("group_column") == "query_id"
    assert _has_action(payload, "query/group-aware validation")
    assert result.module_statuses["baseline_runner"] == "skipped"
    assert _leakage_check(payload, "ranking_query_overlap")["status"] != "failed"


def test_target_in_test_is_critical_leakage(tmp_path: Path) -> None:
    payload, result = _run_fixture("leakage_target_in_test_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "leakage_target_in_test_tiny")
    target_check = _leakage_check(payload, "target_in_test")
    assert target_check["status"] == "failed"
    assert target_check["severity"] in {"high", "critical"}
    assert "target" in target_check["finding"].lower()
    assert "test" in target_check["finding"].lower()
    assert _has_action(payload, "leakage")
    assert _has_p0_action(payload)
    leak_results = [
        item
        for item in payload["hypothesis_results"]
        if item["hypothesis_id"] in {"leak_001", "leak_002"}
    ]
    assert {item["status"] for item in leak_results} <= {"confirmed", "partially_confirmed"}


def test_messy_schema_degrades_gracefully(tmp_path: Path) -> None:
    payload, result = _run_fixture("messy_schema_tiny", tmp_path)

    _assert_core_artifacts(payload, result, "messy_schema_tiny")
    schema = payload["inferred_schema"]
    assert schema["train_base_table"] == "train.csv"
    assert schema["test_base_table"] == "test.csv"
    assert schema["target_column"] == "label"
    assert schema["primary_id_column"] == "record_key"
    assert payload["validation_evidence"]["primary_validation"]["method"] == "stratified_kfold"
    assert payload["hypothesis_results"]


def test_all_generic_fixtures_have_valid_input_json_and_files() -> None:
    for fixture_name in GENERIC_FIXTURES:
        fixture_dir = FIXTURE_ROOT / fixture_name
        assert fixture_dir.is_dir()
        assert (fixture_dir / "sample_submission.csv").is_file()
        assert (fixture_dir / "research_hypotheses.json").is_file()
        assert (fixture_dir / "eda_task_plan.json").is_file()
        assert (fixture_dir / "train_base.csv").is_file() or (fixture_dir / "train.csv").is_file()
        assert (fixture_dir / "test_base.csv").is_file() or (fixture_dir / "test.csv").is_file()

        hypotheses = json.loads((fixture_dir / "research_hypotheses.json").read_text(encoding="utf-8"))
        task_plan = json.loads((fixture_dir / "eda_task_plan.json").read_text(encoding="utf-8"))
        assert hypotheses["competition_id"] == task_plan["competition_id"] == fixture_name
        ids = {item["hypothesis_id"] for item in hypotheses["hypotheses"]}
        assert {"schema_001", "metric_001", "val_001", "leak_001"} <= ids


def _run_fixture(fixture_name: str, tmp_path: Path) -> tuple[dict[str, Any], Any]:
    fixture_dir = FIXTURE_ROOT / fixture_name
    result = asyncio.run(
        run_eda(
            EdaRunConfig(
                competition_id=fixture_name,
                hypotheses_path=fixture_dir / "research_hypotheses.json",
                task_plan_path=fixture_dir / "eda_task_plan.json",
                local_dataset_path=fixture_dir,
                output_dir=tmp_path / "runs",
                download_dataset=False,
                profile_sample_rows=1000,
            )
        )
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))
    return payload, result


def _assert_core_artifacts(payload: dict[str, Any], result: Any, fixture_name: str) -> None:
    assert result.evidence_pack_path.is_file()
    assert result.summary_path.is_file()
    assert (result.output_dir / "module_statuses.json").is_file()
    assert payload["competition_id"] == fixture_name

    for key in (
        "file_inventory",
        "inferred_schema",
        "table_profiles",
        "metric_evidence",
        "validation_evidence",
        "leakage_evidence",
        "hypothesis_results",
        "recommended_next_actions",
    ):
        assert key in payload
        assert payload[key] not in (None, {}, [])

    fixture_dir = FIXTURE_ROOT / fixture_name
    input_hypotheses = json.loads(
        (fixture_dir / "research_hypotheses.json").read_text(encoding="utf-8")
    )["hypotheses"]
    expected_ids = {item["hypothesis_id"] for item in input_hypotheses}
    result_ids = {item["hypothesis_id"] for item in payload["hypothesis_results"]}
    assert result_ids == expected_ids
    assert all(action["evidence_refs"] for action in payload["recommended_next_actions"])


def _profile(payload: dict[str, Any], path: str) -> dict[str, Any]:
    return next(profile for profile in payload["table_profiles"] if profile["path"] == path)


def _column(profile: dict[str, Any], name: str) -> dict[str, Any]:
    return next(column for column in profile["columns"] if column["name"] == name)


def _leakage_check(payload: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(item for item in payload["leakage_evidence"] if item["check_id"] == check_id)


def _has_action(payload: dict[str, Any], text: str) -> bool:
    needle = text.lower()
    return any(needle in action["action"].lower() for action in payload["recommended_next_actions"])


def _has_p0_action(payload: dict[str, Any]) -> bool:
    return any(action["priority"] == "P0" for action in payload["recommended_next_actions"])


def _contains_any(value: str, tokens: tuple[str, ...]) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in tokens)


def _assert_json_has_no_non_finite_numbers(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_has_no_non_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_has_no_non_finite_numbers(item)
    elif isinstance(value, float):
        assert math.isfinite(value)
