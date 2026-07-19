from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_researcher.main import build_full_run_parser
from kaggle_researcher.orchestration import full_run
from kaggle_researcher.orchestration.full_run import (
    FULL_RUN_STAGES,
    FullRunConfig,
    invalidated_stage_ids,
    run_full_research,
)


def test_full_run_parser_accepts_required_flags() -> None:
    args = build_full_run_parser().parse_args([
        "--competition-id", "demo", "--no-download-dataset", "--enable-slice-diagnostics",
        "--force-rerun-stage", "experiment_planner", "--force-rerun-stage", "final_strategy",
        "--require-valid-final-synthesis",
    ])

    assert args.competition_id == "demo"
    assert args.download_dataset is False
    assert args.enable_slice_diagnostics is True
    assert args.force_rerun_stage == ["experiment_planner", "final_strategy"]
    assert args.require_valid_final_synthesis is True


def test_full_run_result_semantics_distinguish_degraded_and_strict(tmp_path: Path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    (final / "final_synthesis_diagnostics.json").write_text("{}", encoding="utf-8")

    degraded = full_run._build_full_run_result(
        tmp_path,
        tmp_path / "run_manifest.json",
        final / "final_strategy.json",
        final / "final_report.md",
        "completed",
        "degraded_fallback",
        True,
        require_valid_final_synthesis=False,
    )
    strict = full_run._build_full_run_result(
        tmp_path,
        tmp_path / "run_manifest.json",
        final / "final_strategy.json",
        final / "final_report.md",
        "completed",
        "degraded_fallback",
        True,
        require_valid_final_synthesis=True,
    )

    assert degraded.workflow_status == "completed_with_degradation"
    assert degraded.final_synthesis_stage_status == "degraded_fallback"
    assert degraded.degraded_stages == ["final_synthesis"]
    assert strict.workflow_status == "failed"
    assert strict.final_synthesis_stage_status == "failed"
    assert strict.status == "failed"


@pytest.mark.asyncio
async def test_full_run_tracks_canonical_stage_order_and_resume(monkeypatch, tmp_path: Path) -> None:
    observed: list[str] = []

    async def fake_stage(stage_id: str, state) -> None:
        observed.append(stage_id)
        run_dir = state.run_dir
        if stage_id == "final_strategy":
            final = run_dir / "final"; final.mkdir(parents=True, exist_ok=True)
            (final / "final_strategy.json").write_text("{}", encoding="utf-8")
            (final / "final_synthesis_diagnostics.json").write_text(
                json.dumps({"schema_version": "1.0", "competition_id": "demo"}),
                encoding="utf-8",
            )
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
    assert (
        manifest["stages"]["final_strategy"]["outputs"]["diagnostics"]
        ["relative_path"]
        == "final/final_synthesis_diagnostics.json"
    )
    assert (
        manifest["final_outputs"]["final_synthesis_diagnostics"]["relative_path"]
        == "final/final_synthesis_diagnostics.json"
    )
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


@pytest.mark.asyncio
async def test_failed_strict_final_stage_preserves_diagnostics_pointer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def fake_stage(stage_id: str, state) -> None:
        if stage_id != "final_strategy":
            return
        final = state.run_dir / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "final_synthesis_diagnostics.json").write_text(
            json.dumps({"schema_version": "1.0", "competition_id": "demo"}),
            encoding="utf-8",
        )
        raise RuntimeError("strict final validation failed")

    monkeypatch.setattr(full_run, "_run_stage", fake_stage)

    with pytest.raises(RuntimeError, match="strict final validation failed"):
        await run_full_research(FullRunConfig(
            competition_id="demo",
            output_root=tmp_path,
            disable_progress=True,
        ))

    run_dir = next(tmp_path.iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    diagnostics = manifest["stages"]["final_strategy"]["outputs"]["diagnostics"]
    assert diagnostics["relative_path"] == "final/final_synthesis_diagnostics.json"
    assert manifest["stages"]["final_strategy"]["status"] == "failed"


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
    from kaggle_researcher.contracts import artifacts
    original_replace = artifacts.os.replace

    def fail_temporary_replace(source, target):
        if str(source).endswith(".tmp"):
            raise OSError("simulated replace interruption")
        return original_replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_temporary_replace)
    with pytest.raises(Exception, match="atomically write"):
        artifacts.write_json_atomic(path, {"schema_version": "1.0", "status": "running"})

    assert json.loads(path.read_text(encoding="utf-8")) == original


@pytest.mark.contract
@pytest.mark.asyncio
async def test_resume_after_reasoning_failure_reuses_semantically_valid_scout_and_eda(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []
    fail_reasoning = True

    async def fake_stage(stage_id: str, state) -> None:
        nonlocal fail_reasoning
        observed.append(stage_id)
        run_dir = state.run_dir
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
            (research / "plan.json").write_text(json.dumps({
                "task_type": "binary_classification", "metric": "roc_auc", "domain": "generic",
            }), encoding="utf-8")
            (research / "retrieved_documents.json").write_text("[]", encoding="utf-8")
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


@pytest.mark.contract
@pytest.mark.asyncio
async def test_force_final_strategy_reuses_completed_upstream_contracts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "existing-run"
    research = run_dir / "research"; research.mkdir(parents=True)
    eda = run_dir / "eda"; eda.mkdir()
    reasoning = run_dir / "reasoning"; reasoning.mkdir()
    final = run_dir / "final"; final.mkdir()
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
    (research / "plan.json").write_text(json.dumps({
        "task_type": "binary_classification", "metric": "roc_auc", "domain": "generic",
    }), encoding="utf-8")
    (research / "retrieved_documents.json").write_text("[]", encoding="utf-8")
    (eda / "eda_evidence_pack.json").write_text(json.dumps({
        "competition_id": "demo", "created_at": "2026-07-13T00:00:00Z", "run_id": "run-001",
        "validation_evidence": {"primary_validation": {"method": "stratified_kfold"}},
    }), encoding="utf-8")
    (eda / "eda_summary.md").write_text("# EDA\n", encoding="utf-8")
    reasoning_payloads = {
        "metric_result.json": {"confidence": "medium", "metric_explanation": "Ranking.", "needs_calibration": False, "rank_averaging_useful": True, "threshold_search_needed": False, "surrogate_loss_suggestion": "Binary objective."},
        "validation_result.json": {"confidence": "medium", "recommended_cv": "StratifiedKFold", "validation_risk": "medium", "likely_split": "iid", "reasoning": "Preserve balance.", "primary_validation": {"method": "stratified_kfold"}},
        "leakage_result.json": {"confidence": "low", "risk_level": "medium", "possible_issues": [], "recommended_checks": []},
        "leaderboard_audit.json": {"confidence": "medium", "shake_up_risk": "medium", "submission_selection_rule": "Use CV.", "public_lb_trust": "low", "warnings": []},
        "experiment_plan.json": [{"experiment_id": "exp_001", "source_hypothesis_ids": ["eda_hypothesis_001"], "priority": "P1", "experiment": "Test encoding", "why": "Test EDA hypothesis.", "cost": "low", "expected_gain": "diagnostic", "risk": "variance", "evidence_ids": []}],
        "skeptical_review.json": {"confidence": "medium", "reviewed_experiment_ids": ["exp_001"], "approved_experiment_ids": ["exp_001"], "rejected_experiment_ids": []},
    }
    for filename, payload in reasoning_payloads.items():
        (reasoning / filename).write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "schema_version": "1.0", "run_id": run_dir.name, "competition_id": "demo",
        "status": "completed", "started_at": None, "finished_at": None, "config": {},
        "stages": {stage: {"status": "completed", "attempt": 1, "outputs": {}, "error": None} for stage in FULL_RUN_STAGES},
        "final_outputs": {},
    }
    manifest["stages"]["research_scout"]["outputs"] = {
        "research_hypotheses": "research/research_hypotheses.json",
        "eda_task_plan": "research/eda_task_plan.json",
        "plan": "research/plan.json",
        "retrieved_documents": "research/retrieved_documents.json",
    }
    manifest["stages"]["eda_engine"]["outputs"] = {
        "evidence_pack": "eda/eda_evidence_pack.json", "summary": "eda/eda_summary.md",
    }
    manifest["stages"]["reasoning_context"]["outputs"] = {
        "metric": "reasoning/metric_result.json", "validation": "reasoning/validation_result.json",
        "leakage": "reasoning/leakage_result.json", "leaderboard": "reasoning/leaderboard_audit.json",
    }
    manifest["stages"]["experiment_planner"]["outputs"] = {"experiment_plan": "reasoning/experiment_plan.json"}
    manifest["stages"]["skeptical_reviewer"]["outputs"] = {"review": "reasoning/skeptical_review.json"}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    observed: list[str] = []

    async def fake_stage(stage_id: str, state) -> None:
        observed.append(stage_id)
        if stage_id == "final_strategy":
            assert state.research_result is not None
            assert state.eda_result is not None
            assert state.reasoning_result is not None
            assert state.reasoning_result.experiments is not None
            assert state.reasoning_result.review is not None
            (final / "final_strategy.json").write_text(json.dumps({
                "competition_id": "demo", "actions": [{
                    "priority": "P1", "action": "Run approved encoding experiment.",
                    "reason": "It traces to the EDA hypothesis.",
                    "evidence_refs": ["validation_evidence.primary_validation"],
                    "related_hypothesis_ids": ["eda_hypothesis_001"],
                    "experiment_ids": ["exp_001"],
                }],
            }), encoding="utf-8")
        elif stage_id == "final_report":
            (final / "final_report.md").write_text("# Final report\n", encoding="utf-8")

    monkeypatch.setattr(full_run, "_run_stage", fake_stage)
    monkeypatch.setattr(full_run, "load_config", lambda: SimpleNamespace(
        deepseek_api_key="test-key", deepseek_v4_pro="test-model"
    ))
    result = await run_full_research(FullRunConfig(
        competition_id="demo",
        output_root=tmp_path,
        resume_run_dir=run_dir,
        force_rerun_stages={"final_strategy"},
        disable_progress=True,
    ))

    assert result.status == "completed"
    assert observed == ["input_validation", "final_strategy", "final_report", "artifact_validation"]
    updated = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    for stage in ("research_scout", "eda_engine", "reasoning_context", "experiment_planner", "skeptical_reviewer"):
        assert updated["stages"][stage]["status"] == "reused"
