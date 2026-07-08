from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kaggle_researcher.eda import orchestrator
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig


FIXTURE_ROOT = Path("tests/fixtures/eda")


def test_mvp_run_keeps_p1_placeholders_without_p1_flags(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("home_credit_tiny", tmp_path / "runs"))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["relationship_inferer"] == "skipped"
    assert result.module_statuses["drift_analyzer"] == "skipped"
    assert payload["relationship_evidence"] == {}
    assert payload["drift_evidence"] == {}


def test_p1_run_writes_relationship_and_drift_evidence(tmp_path: Path) -> None:
    result = asyncio.run(
        _run_fixture("home_credit_tiny", tmp_path / "runs", enable_p1_modules=True)
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["relationship_inferer"] == "completed"
    assert result.module_statuses["drift_analyzer"] == "completed"
    assert (result.output_dir / "relationship_evidence.json").is_file()
    assert (result.output_dir / "drift_evidence.json").is_file()
    assert payload["relationship_evidence"]["relationships"]
    assert payload["drift_evidence"]["status"] == "completed"
    assert "P1 Evidence" in result.summary_path.read_text(encoding="utf-8")


def test_baseline_does_not_run_without_enable_baseline(tmp_path: Path) -> None:
    result = asyncio.run(
        _run_fixture("iid_binary_tiny", tmp_path / "runs", enable_p1_modules=True)
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["baseline_runner"] == "skipped"
    assert payload["baseline_evidence"]["status"] == "skipped"
    assert not (result.output_dir / "artifacts" / "baseline" / "baseline_oof_predictions.csv").exists()


def test_enable_baseline_runs_baseline_module(tmp_path: Path) -> None:
    result = asyncio.run(
        _run_fixture(
            "iid_binary_tiny",
            tmp_path / "runs",
            enable_p1_modules=True,
            enable_baseline=True,
        )
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["baseline_runner"] == "completed"
    assert payload["baseline_evidence"]["status"] == "completed"
    assert (result.output_dir / "artifacts" / "baseline" / "baseline_oof_predictions.csv").is_file()


def test_p1_failure_is_recorded_and_run_continues(tmp_path: Path, monkeypatch) -> None:
    def broken_drift(*args, **kwargs):
        raise RuntimeError("synthetic drift failure")

    monkeypatch.setattr(orchestrator, "analyze_drift", broken_drift)

    result = asyncio.run(
        _run_fixture(
            "iid_binary_tiny",
            tmp_path / "runs",
            modules=["drift_analyzer"],
        )
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["drift_analyzer"] == "failed"
    assert payload["drift_evidence"]["status"] == "failed"
    assert "synthetic drift failure" in payload["drift_evidence"]["error_message"]
    assert payload["hypothesis_results"]
    assert any("drift_analyzer failed" in warning for warning in payload["warnings"])


async def _run_fixture(
    competition_id: str,
    output_dir: Path,
    *,
    enable_p1_modules: bool = False,
    enable_baseline: bool = False,
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
            enable_p1_modules=enable_p1_modules,
            enable_baseline=enable_baseline,
            modules=modules,
        )
    )
