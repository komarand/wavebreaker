from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kaggle_researcher.eda import orchestrator
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_p1_failure_writes_status_placeholder_and_evidence_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_drift(*args, **kwargs):
        raise RuntimeError("synthetic drift failure token=super-secret-token")

    monkeypatch.setattr(orchestrator, "analyze_drift", broken_drift)

    result = asyncio.run(
        _run_fixture(
            "iid_binary_tiny",
            tmp_path / "runs",
            modules=["drift_analyzer"],
        )
    )

    payload = _read_json(result.evidence_pack_path)
    status_payload = _read_json(result.output_dir / "module_statuses.json")

    assert result.evidence_pack_path.is_file()
    assert result.module_statuses["drift_analyzer"] == "failed"
    assert payload["drift_evidence"]["status"] == "failed"
    assert payload["hypothesis_results"]

    drift_status = status_payload["drift_analyzer"]
    assert drift_status["module"] == "drift_analyzer"
    assert drift_status["status"] == "failed"
    assert drift_status["started_at"]
    assert drift_status["finished_at"]
    assert isinstance(drift_status["duration_sec"], float)
    assert "super-secret-token" not in drift_status["error_message"]
    assert "token=[REDACTED]" in drift_status["error_message"]
    assert "super-secret-token" not in payload["drift_evidence"]["error_message"]


def test_blocking_failure_writes_partial_statuses_and_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_schema(*args, **kwargs):
        raise RuntimeError("schema broke password=hunter2 " + ("x" * 1000))

    monkeypatch.setattr(orchestrator, "infer_schema", broken_schema)

    with pytest.raises(RuntimeError, match="schema broke"):
        asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs"))

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "module_statuses.json").is_file()
    assert (run_dir / "eda_evidence_pack_partial.json").is_file()
    assert (run_dir / "eda_evidence_pack.json").is_file()
    assert (run_dir / "file_inventory.json").is_file()

    status_payload = _read_json(run_dir / "module_statuses.json")
    partial_payload = _read_json(run_dir / "eda_evidence_pack_partial.json")

    assert status_payload["file_inventory"]["status"] == "success"
    schema_status = status_payload["schema_inferer"]
    assert schema_status["status"] == "failed"
    assert "hunter2" not in schema_status["error_message"]
    assert "password=[REDACTED]" in schema_status["error_message"]
    assert len(schema_status["error_message"]) <= 500

    assert partial_payload["artifacts"]["partial"] is True
    assert partial_payload["hypothesis_results"] == []
    assert partial_payload["recommended_next_actions"] == []
    assert any("failed before completion" in item for item in partial_payload["warnings"])
    assert any("Partial EDA evidence pack" in item for item in partial_payload["limitations"])


async def _run_fixture(
    competition_id: str,
    output_dir: Path,
    *,
    modules: list[str] | None = None,
):
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
            modules=modules,
        )
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
