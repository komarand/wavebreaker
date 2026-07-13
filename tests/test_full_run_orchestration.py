from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.main import build_full_run_parser
from kaggle_researcher.orchestration import full_run
from kaggle_researcher.orchestration.full_run import (
    FULL_RUN_STAGES,
    FullRunConfig,
    invalidated_stage_ids,
    run_full_research,
    _write_manifest,
)


def test_full_run_parser_accepts_required_flags() -> None:
    args = build_full_run_parser().parse_args([
        "--competition-id", "demo", "--no-download-dataset", "--enable-slice-diagnostics",
        "--force-rerun-stage", "experiment_planner", "--force-rerun-stage", "final_strategy",
    ])

    assert args.competition_id == "demo"
    assert args.download_dataset is False
    assert args.enable_slice_diagnostics is True
    assert args.force_rerun_stage == ["experiment_planner", "final_strategy"]


@pytest.mark.asyncio
async def test_full_run_tracks_canonical_stage_order_and_resume(monkeypatch, tmp_path: Path) -> None:
    observed: list[str] = []

    async def fake_stage(stage_id: str, config: FullRunConfig, run_dir: Path, context: dict[str, object]) -> None:
        observed.append(stage_id)
        if stage_id == "final_strategy":
            final = run_dir / "final"; final.mkdir(parents=True, exist_ok=True)
            (final / "final_strategy.json").write_text("{}", encoding="utf-8")
        if stage_id == "final_report":
            (run_dir / "final" / "final_report.md").write_text("# Report\n", encoding="utf-8")

    monkeypatch.setattr(full_run, "_run_stage", fake_stage)
    monkeypatch.setattr(full_run, "_validate_artifacts", lambda run_dir: None)
    config = FullRunConfig(competition_id="demo", output_root=tmp_path, disable_progress=True)

    result = await run_full_research(config)

    assert observed == list(FULL_RUN_STAGES)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["stages"]["experiment_planner"]["outputs"] == {}
    assert (result.run_dir / "run_summary.md").is_file()

    observed.clear()
    resumed = await run_full_research(FullRunConfig(
        competition_id="demo", output_root=tmp_path, resume_run_dir=result.run_dir, disable_progress=True,
    ))
    assert resumed.status == "reused"
    assert observed == []


@pytest.mark.asyncio
async def test_full_run_rejects_unknown_forced_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown full-run stage IDs"):
        await run_full_research(FullRunConfig(
            competition_id="demo", output_root=tmp_path, force_rerun_stages={"not-a-stage"}, disable_progress=True,
        ))


@pytest.mark.contract
@pytest.mark.parametrize(("forced", "expected"), [
    ("research_scout", set(FULL_RUN_STAGES) - {"input_validation"}),
    ("eda_engine", set(FULL_RUN_STAGES) - {"input_validation", "research_scout"}),
    ("reasoning_context", {"reasoning_context", "experiment_planner", "skeptical_reviewer", "final_strategy", "final_report", "artifact_validation"}),
    ("experiment_planner", {"experiment_planner", "skeptical_reviewer", "final_strategy", "final_report", "artifact_validation"}),
])
def test_forced_stage_invalidates_transitive_dependents(forced: str, expected: set[str]) -> None:
    config = FullRunConfig(competition_id="demo", force_rerun_stages={forced})

    assert invalidated_stage_ids(config) == expected


@pytest.mark.contract
def test_atomic_manifest_failure_preserves_previous_valid_json(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    original = {"schema_version": "1.0", "status": "completed"}
    path.write_text(json.dumps(original), encoding="utf-8")
    original_replace = Path.replace

    def fail_temporary_replace(self: Path, target: Path):
        if self.suffix == ".tmp":
            raise OSError("simulated replace interruption")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_temporary_replace)
    with pytest.raises(OSError, match="replace interruption"):
        _write_manifest(path, {"schema_version": "1.0", "status": "running"})

    assert json.loads(path.read_text(encoding="utf-8")) == original


@pytest.mark.contract
@pytest.mark.asyncio
async def test_resume_after_reasoning_failure_reuses_semantically_valid_scout_and_eda(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []
    fail_reasoning = True

    async def fake_stage(stage_id: str, config: FullRunConfig, run_dir: Path, context: dict[str, object]) -> None:
        nonlocal fail_reasoning
        observed.append(stage_id)
        if stage_id == "research_scout":
            research = run_dir / "research"; research.mkdir(parents=True, exist_ok=True)
            (research / "research_hypotheses.json").write_text(json.dumps({
                "schema_version": "1.0", "competition_id": "demo", "hypotheses": [{
                    "hypothesis_id": "val_001", "category": "validation",
                    "claim": "Use honest validation.", "confidence_before_eda": "medium",
                }],
            }), encoding="utf-8")
            (research / "eda_task_plan.json").write_text(json.dumps({
                "schema_version": "1.0", "competition_id": "demo",
                "eda_tasks": [{"task_id": "val-task", "module": "validation_analyzer", "priority": "P0", "related_hypothesis_ids": ["val_001"]}],
                "hypothesis_index": {"val_001": ["val-task"]},
            }), encoding="utf-8")
        elif stage_id == "eda_engine":
            eda = run_dir / "eda"; eda.mkdir(parents=True, exist_ok=True)
            (eda / "eda_evidence_pack.json").write_text(json.dumps({
                "competition_id": "demo", "created_at": "2026-07-13T00:00:00Z", "run_id": "run-001",
            }), encoding="utf-8")
            (eda / "eda_summary.md").write_text("# EDA\n", encoding="utf-8")
        elif stage_id == "reasoning_context" and fail_reasoning:
            fail_reasoning = False
            raise RuntimeError("simulated reasoning interruption")
        elif stage_id == "final_strategy":
            final = run_dir / "final"; final.mkdir(parents=True, exist_ok=True)
            (final / "final_strategy.json").write_text("{}", encoding="utf-8")
        elif stage_id == "final_report":
            (run_dir / "final" / "final_report.md").write_text("# Report\n", encoding="utf-8")

    monkeypatch.setattr(full_run, "_run_stage", fake_stage)
    monkeypatch.setattr(full_run, "_validate_artifacts", lambda run_dir: None)
    config = FullRunConfig(competition_id="demo", output_root=tmp_path, disable_progress=True)
    with pytest.raises(RuntimeError, match="simulated reasoning interruption"):
        await run_full_research(config)

    run_dir = next(tmp_path.iterdir())
    observed.clear()
    result = await run_full_research(FullRunConfig(
        competition_id="demo",
        output_root=tmp_path,
        resume_run_dir=run_dir,
        disable_progress=True,
    ))

    assert result.status == "completed"
    assert "research_scout" not in observed
    assert "eda_engine" not in observed
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stages"]["research_scout"]["status"] == "reused"
    assert manifest["stages"]["eda_engine"]["status"] == "reused"
