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
    assert result.module_statuses["interaction_diagnostics"] == "skipped"
    assert payload["relationship_evidence"] == {}
    assert payload["drift_evidence"] == {}
    assert payload["interaction_diagnostics"]["status"] == "skipped"
    assert payload["slice_diagnostics"]["status"] == "skipped"


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
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "## Relationships" in summary
    assert "## Drift" in summary


def test_baseline_does_not_run_without_enable_baseline(tmp_path: Path) -> None:
    result = asyncio.run(
        _run_fixture("iid_binary_tiny", tmp_path / "runs", enable_p1_modules=True)
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["baseline_runner"] == "skipped"
    assert result.module_statuses["baseline_ablation_runner"] == "skipped"
    assert payload["baseline_evidence"]["status"] == "skipped"
    assert payload["baseline_ablation_evidence"]["status"] == "skipped"
    assert result.module_statuses["slice_diagnostics"] == "skipped"
    assert payload["slice_diagnostics"]["reason"] == "missing_oof_predictions"
    assert "metric_value" not in payload["baseline_evidence"]
    assert "preprocessing_policy" not in payload["baseline_evidence"]
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
    assert payload["baseline_evidence"]["preprocessing_policy"]["fit_scope"] == "inside_cv_folds"
    assert (result.output_dir / "artifacts" / "baseline" / "baseline_oof_predictions.csv").is_file()


def test_enable_baseline_ablations_runs_baseline_and_ablations(tmp_path: Path) -> None:
    result = asyncio.run(
        _run_fixture(
            "iid_binary_tiny",
            tmp_path / "runs",
            enable_baseline_ablations=True,
        )
    )
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["baseline_runner"] == "completed"
    assert result.module_statuses["baseline_ablation_runner"] == "completed"
    assert payload["baseline_evidence"]["status"] == "completed"
    assert payload["baseline_ablation_evidence"]["status"] == "completed"
    assert payload["baseline_ablation_evidence"]["fold_policy"]["same_folds_across_ablations"] is True
    assert payload["baseline_ablation_evidence"]["ablations"]
    assert (result.output_dir / "baseline_ablation_evidence.json").is_file()


def test_enable_interaction_diagnostics_writes_optional_evidence(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs", enable_interaction_diagnostics=True))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["interaction_diagnostics"] == "completed"
    assert payload["interaction_diagnostics"]["status"] == "completed"
    assert (result.output_dir / "interaction_diagnostics.json").is_file()
    assert "## Interaction diagnostics" in result.summary_path.read_text(encoding="utf-8")


def test_enable_visual_diagnostics_writes_plot_manifest(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs", enable_visual_diagnostics=True))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["visual_diagnostics"] == "completed"
    assert payload["visual_diagnostics"]["generated_plots"]
    assert (result.output_dir / "plots_manifest.json").is_file()
    assert "## Visual diagnostics" in result.summary_path.read_text(encoding="utf-8")


def test_slice_diagnostics_skips_cleanly_without_baseline_oof(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs", enable_slice_diagnostics=True))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["slice_diagnostics"] == "skipped"
    assert payload["slice_diagnostics"]["reason"] == "missing_oof_predictions"


def test_slice_diagnostics_consumes_fold_safe_baseline_oof(tmp_path: Path) -> None:
    result = asyncio.run(_run_fixture("iid_binary_tiny", tmp_path / "runs", enable_baseline=True, enable_slice_diagnostics=True))
    payload = json.loads(result.evidence_pack_path.read_text(encoding="utf-8"))

    assert result.module_statuses["slice_diagnostics"] == "completed"
    assert payload["slice_diagnostics"]["slices"]
    assert (result.output_dir / "slice_diagnostics.json").is_file()


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
    enable_baseline_ablations: bool = False,
    enable_interaction_diagnostics: bool = False,
    enable_visual_diagnostics: bool = False,
    enable_slice_diagnostics: bool = False,
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
            enable_baseline_ablations=enable_baseline_ablations,
            enable_interaction_diagnostics=enable_interaction_diagnostics,
            enable_visual_diagnostics=enable_visual_diagnostics,
            enable_slice_diagnostics=enable_slice_diagnostics,
            modules=modules,
        )
    )
