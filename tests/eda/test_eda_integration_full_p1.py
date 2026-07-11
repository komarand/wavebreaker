from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.quality import (
    validate_evidence_pack,
    validate_hypothesis_results,
    validate_no_unsupported_summary_claims,
)
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaRunConfig, ResearchHypotheses


FIXTURE_ROOT = Path("tests/fixtures/eda")
P1_MODULES = ["relationship_inferer", "drift_analyzer", "feature_probe"]
TEMPORAL_PRIMARY_TOKENS = ("temporal", "oot", "expanding")


def test_home_credit_full_p1_offline_integration(tmp_path: Path) -> None:
    payload, result = _run_full_p1_fixture("home_credit_tiny", tmp_path)

    _assert_common_full_p1_outputs(payload, result)
    assert payload["validation_evidence"]["primary_validation"]["method"] == "temporal_holdout"
    assert payload["relationship_evidence"]["relationships"]
    assert payload["drift_evidence"]["status"] == "completed"
    assert payload["feature_probe_evidence"]


@pytest.mark.parametrize(
    ("fixture_name", "expected_method"),
    [
        ("iid_binary_tiny", "stratified_kfold"),
        ("regression_tiny", "kfold"),
    ],
)
def test_iid_and_regression_full_p1_generic_validation(
    fixture_name: str,
    expected_method: str,
    tmp_path: Path,
) -> None:
    payload, result = _run_full_p1_fixture(fixture_name, tmp_path)

    _assert_common_full_p1_outputs(payload, result)
    primary_method = payload["validation_evidence"]["primary_validation"]["method"]
    assert primary_method == expected_method
    assert not _contains_any(primary_method, TEMPORAL_PRIMARY_TOKENS)
    assert payload["drift_evidence"]["status"] in {"completed", "not_testable", "skipped"}

    if fixture_name == "iid_binary_tiny":
        assert payload["metric_evidence"]["task_type"] == "binary_classification"
        assert payload["metric_evidence"]["requires_probabilities"] is True
    else:
        assert payload["metric_evidence"]["task_type"] == "regression"
        assert payload["metric_evidence"]["metric_family"] == "regression_error"
        assert payload["metric_evidence"]["metric_name"] == "rmse"


def test_grouped_binary_full_p1_selects_group_aware_validation(tmp_path: Path) -> None:
    payload, result = _run_full_p1_fixture("grouped_binary_tiny", tmp_path)

    _assert_common_full_p1_outputs(payload, result)
    primary = payload["validation_evidence"]["primary_validation"]
    assert primary["method"] in {"group_kfold", "stratified_group_kfold"}
    assert primary["group_column"] == "customer_id"
    assert payload["metric_evidence"]["task_type"] == "binary_classification"


def _run_full_p1_fixture(fixture_name: str, tmp_path: Path) -> tuple[dict[str, Any], Any]:
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
                modules=P1_MODULES,
            )
        )
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))
    return payload, result


def _assert_common_full_p1_outputs(payload: dict[str, Any], result: Any) -> None:
    pack = EdaEvidencePack(**payload)
    hypotheses = _load_hypotheses(payload["competition_id"])
    summary = result.summary_path.read_text(encoding="utf-8")

    assert result.evidence_pack_path.is_file()
    assert result.summary_path.is_file()
    assert (result.output_dir / "module_statuses.json").is_file()
    assert result.module_statuses["baseline_runner"] == "skipped"
    assert result.module_statuses["relationship_inferer"] == "completed"
    assert result.module_statuses["feature_probe"] == "completed"

    assert payload["recommended_next_actions"]
    assert {item["hypothesis_id"] for item in payload["hypothesis_results"]} == {
        item.hypothesis_id for item in hypotheses.hypotheses
    }
    assert all(action["evidence_refs"] for action in payload["recommended_next_actions"])

    quality_warnings = [
        *validate_evidence_pack(pack),
        *validate_hypothesis_results(pack, hypotheses),
        *validate_no_unsupported_summary_claims(summary, pack),
    ]
    assert not _critical_quality_warnings(quality_warnings)


def _load_hypotheses(competition_id: str) -> ResearchHypotheses:
    path = FIXTURE_ROOT / competition_id / "research_hypotheses.json"
    return ResearchHypotheses(**json.loads(path.read_text(encoding="utf-8")))


def _critical_quality_warnings(warnings: list[str]) -> list[str]:
    critical_tokens = (
        "broken evidence_ref",
        "missing hypothesis result",
        "unexpected hypothesis result",
        "has no evidence_refs",
        "overclaims",
        "temporal validation is required",
    )
    return [
        warning
        for warning in warnings
        if any(token in warning.lower() for token in critical_tokens)
    ]


def _contains_any(value: str, tokens: tuple[str, ...]) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in tokens)
