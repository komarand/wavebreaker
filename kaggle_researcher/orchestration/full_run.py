from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.config import load_config
from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    FinalStageResult,
    ReasoningStageResult,
    ResearchStageResult,
    load_eda_evidence_pack,
    load_experiment_plan,
    load_final_strategy,
    load_skeptical_review,
    write_experiment_plan,
    write_final_strategy,
    write_json_atomic,
)
from kaggle_researcher.contracts.eda_task_plan import (
    load_eda_task_plan,
    write_eda_task_plan_atomic,
)
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.ids import StageId
from kaggle_researcher.contracts.manifest import (
    ArtifactPointer,
    FinalOutputManifest,
    ManifestConfigSnapshot,
    RunManifest,
    RunStatus,
    StageErrorRecord,
    StageStatus,
    artifact_pointer,
    load_run_manifest,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_reused,
    mark_stage_running,
    mark_stage_stale,
    new_run_manifest,
    validate_artifact_pointer,
    write_run_manifest_atomic,
)
from kaggle_researcher.contracts.pipeline import validate_full_run_artifacts
from kaggle_researcher.contracts.registries import build_contract_registries
from kaggle_researcher.contracts.research_hypotheses import (
    load_research_hypotheses,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig
from kaggle_researcher.orchestration.state import (
    FullRunConfig,
    FullRunState,
    InputValidationResult,
    MissingStageDependencyError,
    RuntimeServices,
    StageFailure,
)
from kaggle_researcher.progress import ProgressConfig, progress_write
from kaggle_researcher.reasoning.experiment_planner import plan_experiments
from kaggle_researcher.reasoning.final_synthesizer import (
    render_final_strategy,
    render_final_strategy_summary,
    synthesize_final_strategy,
)
from kaggle_researcher.reasoning.leaderboard_auditor import audit_leaderboard_risk
from kaggle_researcher.reasoning.leakage_risk_analyst import analyze_leakage_risk
from kaggle_researcher.reasoning.metric_specialist import analyze_metric
from kaggle_researcher.reasoning.provenance import attach_default_provenance
from kaggle_researcher.reasoning.skeptical_reviewer import review
from kaggle_researcher.reasoning.validation_architect import design_validation
from kaggle_researcher.schemas import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    RetrievedDocument,
    ValidationResult,
)
from kaggle_researcher.workflow import (
    FinalSynthesisDegradedError,
    FinalSynthesisStageStatus,
    WorkflowStatus,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    display_name: str
    required: bool
    enabled: bool = True
    dependencies: tuple[str, ...] = ()


FULL_RUN_STAGES = (
    "input_validation",
    "research_scout",
    "eda_engine",
    "reasoning_context",
    "experiment_planner",
    "skeptical_reviewer",
    "final_strategy",
    "final_report",
    "artifact_validation",
)


@dataclass(frozen=True)
class FullRunResult:
    run_dir: Path
    manifest_path: Path
    final_strategy_path: Path
    final_report_path: Path
    status: str
    final_synthesis_status: Literal[
        "llm_success", "repaired_success", "degraded_fallback"
    ] | None = None
    final_synthesis_degraded: bool = False
    workflow_status: WorkflowStatus = "success"
    degraded_stages: list[str] = field(default_factory=list)
    final_synthesis_stage_status: FinalSynthesisStageStatus | None = None
    final_synthesis_diagnostics_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        is_degraded = self.final_synthesis_status == "degraded_fallback"
        if self.final_synthesis_degraded != is_degraded:
            raise ValueError(
                "final_synthesis_degraded must match final_synthesis_status"
            )
        if is_degraded:
            if self.workflow_status == "success":
                raise ValueError(
                    "degraded final synthesis cannot have workflow_status='success'"
                )
            if "final_synthesis" not in self.degraded_stages:
                raise ValueError(
                    "degraded final synthesis must list final_synthesis in degraded_stages"
                )
            expected_stage_status = (
                "failed" if self.workflow_status == "failed" else "degraded_fallback"
            )
            if self.final_synthesis_stage_status != expected_stage_status:
                raise ValueError(
                    "final synthesis stage status contradicts workflow_status"
                )


def stage_registry(config: FullRunConfig) -> tuple[StageDefinition, ...]:
    planning_enabled = config.profile != "minimal"
    review_enabled = config.profile != "minimal"
    final_dependencies = ["reasoning_context"]
    if planning_enabled:
        final_dependencies.append("experiment_planner")
    if review_enabled:
        final_dependencies.append("skeptical_reviewer")
    return (
        StageDefinition("input_validation", "Input validation", True),
        StageDefinition("research_scout", "Research Scout", True, dependencies=("input_validation",)),
        StageDefinition("eda_engine", "EDA Engine", True, dependencies=("research_scout",)),
        StageDefinition("reasoning_context", "Reasoning preparation", True, dependencies=("eda_engine",)),
        StageDefinition("experiment_planner", "Experiment planning", False, planning_enabled, ("reasoning_context",)),
        StageDefinition("skeptical_reviewer", "Skeptical review", False, review_enabled, ("experiment_planner",)),
        StageDefinition("final_strategy", "Final strategy synthesis", True, dependencies=tuple(final_dependencies)),
        StageDefinition("final_report", "Final report assembly", True, dependencies=("final_strategy",)),
        StageDefinition("artifact_validation", "Artifact validation", True, dependencies=("final_report",)),
    )


async def run_full_research(config: FullRunConfig) -> FullRunResult:
    _validate_config(config)
    run_dir = _resolve_run_dir(config)
    manifest_path = run_dir / "run_manifest.json"
    _configure_log(run_dir)
    manifest = _load_or_create_manifest(run_dir, config)
    if manifest.competition_id != config.competition_id:
        raise ValueError(
            f"Manifest competition_id {manifest.competition_id!r} does not match {config.competition_id!r}"
        )
    if _can_reuse(manifest, config, run_dir):
        reused_status, reused_degraded = _read_final_synthesis_status(
            run_dir / "final" / "final_strategy.json"
        )
        result = _build_full_run_result(
            run_dir,
            manifest_path,
            run_dir / "final" / "final_strategy.json",
            run_dir / "final" / "final_report.md",
            "reused",
            reused_status,
            reused_degraded,
            require_valid_final_synthesis=config.require_valid_final_synthesis,
        )
        if result.workflow_status == "failed":
            raise FinalSynthesisDegradedError(
                result.final_synthesis_diagnostics_path
                or run_dir / "final" / "final_synthesis_diagnostics.json",
                result=result,
            )
        return result

    services = _runtime_services(config)
    state = FullRunState(run_dir=run_dir, config=config, services=services, manifest=manifest)
    definitions = stage_registry(config)
    enabled = [stage for stage in definitions if stage.enabled]
    invalidated = invalidated_stage_ids(config)
    for stage in definitions:
        entry = state.manifest.stages[StageId(stage.stage_id)]
        if not stage.enabled and entry.status is StageStatus.PENDING:
            state.manifest = _replace_manifest_stage(
                state.manifest,
                stage.stage_id,
                entry.model_copy(update={"status": StageStatus.SKIPPED}),
            )
    state.manifest = state.manifest.model_copy(update={
        "status": RunStatus.RUNNING,
        "started_at": state.manifest.started_at or _utcnow(),
    })
    _persist_state(state, manifest_path)

    with tqdm(
        total=len(enabled),
        desc="Full research pipeline",
        unit="stage",
        disable=services.progress.disabled,
    ) as bar:
        for stage in enabled:
            bar.set_postfix_str(stage.display_name)
            reusable, stale_reason = _can_reuse_stage(
                state.manifest, stage.stage_id, run_dir, invalidated
            )
            if reusable:
                _hydrate_reused_stage(stage.stage_id, state)
                state.manifest = mark_stage_reused(state.manifest, stage_id=stage.stage_id)
                _persist_state(state, manifest_path)
                bar.update(1)
                continue
            entry = state.manifest.stages[StageId(stage.stage_id)]
            if stale_reason and entry.status in {StageStatus.COMPLETED, StageStatus.REUSED}:
                state.manifest = mark_stage_stale(
                    state.manifest, stage_id=stage.stage_id, reason=stale_reason
                )
                invalidated.add(stage.stage_id)
                invalidated.update(dependent_stage_ids(stage.stage_id, definitions))
                _persist_state(state, manifest_path)

            state.manifest = mark_stage_running(
                state.manifest, stage_id=stage.stage_id, started_at=_utcnow()
            )
            _persist_state(state, manifest_path)
            started = time.perf_counter()
            try:
                await _run_stage(stage.stage_id, state)
            except Exception as exc:
                logger.exception("Full-run stage %s failed", stage.stage_id)
                error = _error_record(exc, stage.stage_id)
                duration = round(time.perf_counter() - started, 3)
                state.manifest = mark_stage_failed(
                    state.manifest,
                    stage_id=stage.stage_id,
                    error=error,
                    finished_at=_utcnow(),
                    duration_sec=duration,
                    partial=not stage.required and not config.fail_fast,
                    outputs=_stage_output_pointers(stage.stage_id, run_dir),
                )
                _persist_state(state, manifest_path)
                if stage.required or config.fail_fast:
                    state.manifest = state.manifest.model_copy(update={
                        "status": RunStatus.FAILED,
                        "finished_at": _utcnow(),
                    })
                    _persist_state(state, manifest_path)
                    progress_write(
                        f"{stage.display_name} failed: {error.message}",
                        config=services.progress,
                    )
                    raise
                failure = StageFailure(StageId(stage.stage_id), error.message)
                state.optional_stage_failures = (*state.optional_stage_failures, failure)
            else:
                outputs = _stage_output_pointers(stage.stage_id, run_dir)
                state.manifest = mark_stage_completed(
                    state.manifest,
                    stage_id=stage.stage_id,
                    outputs=outputs,
                    finished_at=_utcnow(),
                    duration_sec=round(time.perf_counter() - started, 3),
                )
                _persist_state(state, manifest_path)
            bar.update(1)

    final_strategy_path = run_dir / "final" / "final_strategy.json"
    final_report_path = run_dir / "final" / "final_report.md"
    final_synthesis_diagnostics_path = (
        run_dir / "final" / "final_synthesis_diagnostics.json"
    )
    final_synthesis_status, final_synthesis_degraded = _read_final_synthesis_status(
        final_strategy_path
    )
    result = _build_full_run_result(
        run_dir,
        manifest_path,
        final_strategy_path,
        final_report_path,
        "completed",
        final_synthesis_status,
        final_synthesis_degraded,
        require_valid_final_synthesis=config.require_valid_final_synthesis,
    )
    state.manifest = state.manifest.model_copy(update={
        "status": (
            RunStatus.FAILED
            if result.workflow_status == "failed"
            else RunStatus.COMPLETED
        ),
        "finished_at": _utcnow(),
        "final_outputs": FinalOutputManifest(
            final_strategy=artifact_pointer(
                final_strategy_path,
                run_dir=run_dir,
                contract_family="final_strategy",
                schema_version="1.0",
            ),
            final_report=artifact_pointer(final_report_path, run_dir=run_dir),
            final_synthesis_diagnostics=artifact_pointer(
                final_synthesis_diagnostics_path,
                run_dir=run_dir,
            ),
        ),
    })
    _persist_state(state, manifest_path)
    _write_summary(run_dir, state.manifest)
    if result.workflow_status == "failed":
        raise FinalSynthesisDegradedError(
            result.final_synthesis_diagnostics_path
            or final_synthesis_diagnostics_path,
            result=result,
        )
    return result


def _build_full_run_result(
    run_dir: Path,
    manifest_path: Path,
    final_strategy_path: Path,
    final_report_path: Path,
    status: str,
    final_synthesis_status: str | None,
    final_synthesis_degraded: bool,
    *,
    require_valid_final_synthesis: bool,
) -> FullRunResult:
    diagnostics_path = run_dir / "final" / "final_synthesis_diagnostics.json"
    workflow_status: WorkflowStatus = "success"
    stage_status: FinalSynthesisStageStatus | None = None
    degraded_stages: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []
    if final_synthesis_status == "llm_success":
        stage_status = "success"
    elif final_synthesis_status == "repaired_success":
        stage_status = "repaired_success"
        warnings.append(
            "Final synthesis required deterministic contract repair and then passed validation."
        )
    elif final_synthesis_status == "degraded_fallback":
        degraded_stages.append("final_synthesis")
        warnings.append(
            "Final synthesis completed with a deterministic degraded fallback; "
            "this is not a successful LLM synthesis."
        )
        limitations.append(
            "The final strategy was assembled deterministically because the LLM "
            "output did not satisfy the final strategy contract."
        )
        if require_valid_final_synthesis:
            workflow_status = "failed"
            stage_status = "failed"
            status = "failed"
        else:
            workflow_status = "completed_with_degradation"
            stage_status = "degraded_fallback"
    return FullRunResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        final_strategy_path=final_strategy_path,
        final_report_path=final_report_path,
        status=status,
        final_synthesis_status=final_synthesis_status,
        final_synthesis_degraded=final_synthesis_degraded,
        workflow_status=workflow_status,
        degraded_stages=degraded_stages,
        final_synthesis_stage_status=stage_status,
        final_synthesis_diagnostics_path=(
            diagnostics_path if diagnostics_path.is_file() else None
        ),
        warnings=warnings,
        limitations=limitations,
    )


def _read_final_synthesis_status(path: Path) -> tuple[str | None, bool]:
    try:
        strategy = load_final_strategy(path)
    except (OSError, ValueError):
        return None, False
    return strategy.synthesis_status, strategy.fallback_used


async def _run_stage(stage_id: str, state: FullRunState) -> None:
    if stage_id == "input_validation":
        state.input_result = InputValidationResult(state.config.competition_id)
    elif stage_id == "research_scout":
        await _run_research_scout(state)
    elif stage_id == "eda_engine":
        await _run_eda_stage(state)
    elif stage_id == "reasoning_context":
        await _prepare_reasoning(state)
    elif stage_id == "experiment_planner":
        await _plan_experiments(state)
    elif stage_id == "skeptical_reviewer":
        await _review_plan(state)
    elif stage_id == "final_strategy":
        await _synthesize_strategy(state)
    elif stage_id == "final_report":
        _write_final_report(state)
    elif stage_id == "artifact_validation":
        _validate_artifacts(state.run_dir)
    else:
        raise ValueError(f"Unknown full-run stage: {stage_id}")


async def _run_research_scout(state: FullRunState) -> None:
    from kaggle_researcher.main import run_research

    config = state.config
    result = await run_research(
        competition_url=config.competition_url or "",
        competition_desc=config.competition_description,
        competition_id=config.competition_id,
        mode="scout",
        write_eda_plan=True,
        show_progress=not config.disable_progress,
    )
    source = Path(result.run_artifacts_path or "")
    if not source.is_dir():
        raise RuntimeError("Research Scout did not return a valid artifact directory.")
    target = state.run_dir / "research"
    _copy_artifacts(source, target)
    _canonicalize_research_artifact_bundle(target)
    state.research_result = _load_research_stage_result(state.run_dir)


async def _run_eda_stage(state: FullRunState) -> None:
    research = state.require_research("eda_engine")
    config = state.config
    result = await run_eda(EdaRunConfig(
        competition_id=config.competition_id,
        competition_url=config.competition_url,
        hypotheses_path=research.hypotheses_path,
        task_plan_path=research.task_plan_path,
        local_dataset_path=config.local_dataset_path,
        output_dir=state.run_dir / "eda" / "stage_runs",
        download_dataset=config.download_dataset,
        enable_p1_modules=_enabled(config, "p1"),
        enable_baseline=_enabled(config, "baseline"),
        enable_baseline_ablations=_enabled(config, "ablations"),
        enable_interaction_diagnostics=_enabled(config, "interactions"),
        enable_slice_diagnostics=_enabled(config, "slices"),
        enable_source_claim_validation=_enabled(config, "source_claims"),
        enable_visual_diagnostics=_enabled(config, "visuals"),
        fail_fast=config.fail_fast,
    ))
    target = state.run_dir / "eda"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.evidence_pack_path, target / "eda_evidence_pack.json")
    shutil.copy2(result.summary_path, target / "eda_summary.md")
    for filename in ("plots_manifest.json", "module_statuses.json"):
        candidate = result.output_dir / filename
        if candidate.is_file():
            shutil.copy2(candidate, target / filename)
    state.eda_result = _load_eda_stage_result(state.run_dir)


async def _prepare_reasoning(state: FullRunState) -> None:
    research = state.require_research("reasoning_context")
    state.require_eda("reasoning_context")
    client = state.services.reasoning_client
    model = state.services.reasoning_model
    config = state.config
    docs = list(research.retrieved_documents)
    metric = await analyze_metric(
        plan_data=research.plan_data, retrieved_documents=docs, client=client, model=model
    )
    validation = await design_validation(
        competition_desc=config.competition_description,
        plan_data=research.plan_data,
        retrieved_documents=docs,
        client=client,
        model=model,
    )
    leakage = await analyze_leakage_risk(
        competition_desc=config.competition_description,
        plan_data=research.plan_data,
        retrieved_documents=docs,
        client=client,
        model=model,
    )
    leaderboard = await audit_leaderboard_risk(
        competition_desc=config.competition_description,
        plan_data=research.plan_data,
        validation_result=validation,
        retrieved_documents=docs,
        client=client,
        model=model,
    )
    reasoning_dir = state.run_dir / "reasoning"
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("metric_result", metric),
        ("validation_result", validation),
        ("leakage_result", leakage),
        ("leaderboard_audit", leaderboard),
    ):
        write_json_atomic(reasoning_dir / f"{name}.json", value)
    state.reasoning_result = ReasoningStageResult(metric, validation, leakage, leaderboard)


async def _plan_experiments(state: FullRunState) -> None:
    research = state.require_research("experiment_planner")
    eda = state.require_eda("experiment_planner")
    reasoning = state.require_reasoning("experiment_planner")
    experiments = await plan_experiments(
        validation_result=reasoning.validation,
        leakage_result=reasoning.leakage,
        metric_result=reasoning.metric,
        retrieved_documents=list(research.retrieved_documents),
        client=state.services.reasoning_client,
        model=state.services.reasoning_model,
        eda_hypotheses=eda.evidence_pack.testable_hypotheses,
    )
    experiment_plan = ExperimentPlan(experiments=experiments)
    write_experiment_plan(state.run_dir / "reasoning" / "experiment_plan.json", experiment_plan)
    state.reasoning_result = replace(reasoning, experiments=experiment_plan)


async def _review_plan(state: FullRunState) -> None:
    research = state.require_research("skeptical_reviewer")
    reasoning = state.require_reasoning("skeptical_reviewer")
    if reasoning.experiments is None:
        raise MissingStageDependencyError(
            stage_id="skeptical_reviewer", missing_dependency="experiment_planner"
        )
    docs = list(research.retrieved_documents)
    sections = {
        "validation": attach_default_provenance(
            "validation", reasoning.validation.model_dump(mode="json"), docs
        ),
        "leakage": attach_default_provenance(
            "leakage", reasoning.leakage.model_dump(mode="json"), docs
        ),
        "metric": attach_default_provenance(
            "metric", reasoning.metric.model_dump(mode="json"), docs
        ),
        "experiments": attach_default_provenance(
            "experiments",
            [item.model_dump(mode="json") for item in reasoning.experiments.experiments],
            docs,
        ),
    }
    result = await review(
        draft_sections=sections,
        retrieved_documents=docs,
        client=state.services.reasoning_client,
        model=state.services.reasoning_model,
        artifact_dir=state.run_dir / "reasoning",
    )
    write_json_atomic(state.run_dir / "reasoning" / "skeptical_review.json", result)
    state.reasoning_result = replace(reasoning, review=result)


async def _synthesize_strategy(state: FullRunState) -> None:
    research = state.require_research("final_strategy")
    eda = state.require_eda("final_strategy")
    reasoning = state.require_reasoning("final_strategy")
    registries = build_contract_registries(
        research=research, eda=eda, reasoning=reasoning
    )
    context = build_final_synthesis_context(
        competition_desc=state.config.competition_description,
        research=research,
        eda=eda,
        reasoning=reasoning,
        registries=registries,
        eda_summary_text=eda.summary_path.read_text(encoding="utf-8"),
        optional_stage_failures=state.optional_stage_failures,
    )
    final = state.run_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    result = await synthesize_final_strategy(
        context=context,
        registries=registries,
        client=state.services.reasoning_client,
        model=state.services.reasoning_model,
        diagnostics_dir=final,
    )
    strategy_path = final / "final_strategy.json"
    report_path = final / "final_report.md"
    write_final_strategy(strategy_path, result)
    (final / "final_strategy.md").write_text(render_final_strategy(result), encoding="utf-8")
    (final / "final_strategy_summary.md").write_text(
        render_final_strategy_summary(result), encoding="utf-8"
    )
    state.final_result = FinalStageResult(result, strategy_path, report_path)


def _write_final_report(state: FullRunState) -> None:
    final = state.require_final("final_report")
    assert final.report_path is not None
    final.report_path.write_text(render_final_strategy(final.strategy), encoding="utf-8")


def _validate_artifacts(run_dir: Path) -> None:
    validate_full_run_artifacts(run_dir)


def _runtime_services(config: FullRunConfig) -> RuntimeServices:
    settings = load_config()
    return RuntimeServices(
        reasoning_client=DeepSeekClient(api_key=settings.deepseek_api_key),
        reasoning_model=settings.deepseek_v4_pro,
        progress=ProgressConfig(enabled=not config.disable_progress),
        logger=logger,
    )


def _load_research_stage_result(run_dir: Path) -> ResearchStageResult:
    research = run_dir / "research"
    hypotheses_path = research / "research_hypotheses.json"
    task_plan_path = research / "eda_task_plan.json"
    hypotheses, _ = load_research_hypotheses(hypotheses_path)
    task_plan, _ = load_eda_task_plan(task_plan_path, hypotheses=hypotheses)
    plan_data = PlanData.model_validate(_read_json(research / "plan.json"))
    documents = tuple(
        RetrievedDocument.model_validate(value)
        for value in _read_json(research / "retrieved_documents.json")
    )
    domain_patterns = tuple(_read_json_or(research / "domain_patterns.json", []))
    return ResearchStageResult(
        hypotheses,
        task_plan,
        hypotheses_path,
        task_plan_path,
        plan_data,
        documents,
        domain_patterns,
    )


def _load_eda_stage_result(run_dir: Path) -> EdaStageResult:
    target = run_dir / "eda"
    evidence_path = target / "eda_evidence_pack.json"
    return EdaStageResult(
        load_eda_evidence_pack(evidence_path),
        evidence_path,
        target / "eda_summary.md",
    )


def _load_reasoning_stage_result(run_dir: Path) -> ReasoningStageResult:
    reasoning = run_dir / "reasoning"
    return ReasoningStageResult(
        metric=MetricResult.model_validate(_read_json(reasoning / "metric_result.json")),
        validation=ValidationResult.model_validate(_read_json(reasoning / "validation_result.json")),
        leakage=LeakageRiskResult.model_validate(_read_json(reasoning / "leakage_result.json")),
        leaderboard=LeaderboardAuditResult.model_validate(_read_json(reasoning / "leaderboard_audit.json")),
    )


def _hydrate_reused_stage(stage_id: str, state: FullRunState) -> None:
    if stage_id == "research_scout":
        state.research_result = _load_research_stage_result(state.run_dir)
    elif stage_id == "eda_engine":
        state.eda_result = _load_eda_stage_result(state.run_dir)
    elif stage_id == "reasoning_context":
        state.reasoning_result = _load_reasoning_stage_result(state.run_dir)
    elif stage_id == "experiment_planner":
        reasoning = state.require_reasoning(stage_id)
        plan = load_experiment_plan(_manifest_output_path(
            state.manifest, state.run_dir, stage_id, "experiment_plan"
        ))
        state.reasoning_result = replace(reasoning, experiments=plan)
    elif stage_id == "skeptical_reviewer":
        reasoning = state.require_reasoning(stage_id)
        review_result = load_skeptical_review(_manifest_output_path(
            state.manifest, state.run_dir, stage_id, "review"
        ))
        state.reasoning_result = replace(reasoning, review=review_result)
    elif stage_id == "final_strategy":
        strategy_path = _manifest_output_path(
            state.manifest, state.run_dir, stage_id, "strategy"
        )
        state.final_result = FinalStageResult(
            load_final_strategy(strategy_path),
            strategy_path,
            state.run_dir / "final" / "final_report.md",
        )


def _can_reuse(manifest: RunManifest, config: FullRunConfig, run_dir: Path) -> bool:
    if config.force_rerun_stages or manifest.status is not RunStatus.COMPLETED:
        return False
    if manifest.config != _config_snapshot(config):
        return False
    try:
        _validate_artifacts(run_dir)
    except Exception:
        return False
    return all(
        entry.status in {StageStatus.COMPLETED, StageStatus.SKIPPED, StageStatus.REUSED}
        for entry in manifest.stages.values()
    )


def _can_reuse_stage(
    manifest: RunManifest,
    stage_id: str,
    run_dir: Path,
    invalidated: set[str],
) -> tuple[bool, str | None]:
    if stage_id == "input_validation" or stage_id in invalidated:
        return False, None
    entry = manifest.stages[StageId(stage_id)]
    if entry.status not in {StageStatus.COMPLETED, StageStatus.REUSED}:
        return False, None
    expected = _expected_output_names(stage_id)
    if expected and not expected <= set(entry.outputs):
        return False, "manifest is missing required artifact pointers"
    for name in expected:
        valid, reason = validate_artifact_pointer(entry.outputs[name], run_dir=run_dir)
        if not valid:
            return False, f"{name}: {reason}"
    try:
        if stage_id == "research_scout":
            _load_research_stage_result(run_dir)
        elif stage_id == "eda_engine":
            _load_eda_stage_result(run_dir)
        elif stage_id == "reasoning_context":
            _load_reasoning_stage_result(run_dir)
        elif stage_id == "experiment_planner":
            load_experiment_plan(_manifest_output_path(manifest, run_dir, stage_id, "experiment_plan"))
        elif stage_id == "skeptical_reviewer":
            load_skeptical_review(_manifest_output_path(manifest, run_dir, stage_id, "review"))
        elif stage_id == "final_strategy":
            load_final_strategy(_manifest_output_path(manifest, run_dir, stage_id, "strategy"))
        elif stage_id == "final_report":
            path = _manifest_output_path(manifest, run_dir, stage_id, "report")
            if not path.read_text(encoding="utf-8").strip():
                return False, "final report is empty"
    except Exception as exc:
        return False, f"canonical artifact validation failed: {type(exc).__name__}"
    return True, None


def _manifest_output_path(
    manifest: RunManifest, run_dir: Path, stage_id: str, output_name: str
) -> Path:
    try:
        pointer = manifest.stages[StageId(stage_id)].outputs[output_name]
    except KeyError as exc:
        raise MissingStageDependencyError(
            stage_id=stage_id, missing_dependency=stage_id
        ) from exc
    path = (run_dir / pointer.relative_path).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Unsafe artifact pointer for {stage_id}.{output_name}")
    return path


def _expected_output_names(stage_id: str) -> set[str]:
    return {
        "research_scout": {"research_hypotheses", "eda_task_plan", "plan", "retrieved_documents"},
        "eda_engine": {"evidence_pack", "summary"},
        "reasoning_context": {"validation", "metric", "leakage", "leaderboard"},
        "experiment_planner": {"experiment_plan"},
        "skeptical_reviewer": {"review"},
        "final_strategy": {"strategy", "diagnostics"},
        "final_report": {"report"},
    }.get(stage_id, set())


def _stage_output_pointers(stage_id: str, run_dir: Path) -> dict[str, ArtifactPointer]:
    paths = {
        "research_scout": {
            "research_hypotheses": ("research/research_hypotheses.json", "research_hypotheses"),
            "eda_task_plan": ("research/eda_task_plan.json", "eda_task_plan"),
            "plan": ("research/plan.json", None),
            "retrieved_documents": ("research/retrieved_documents.json", None),
        },
        "eda_engine": {
            "evidence_pack": ("eda/eda_evidence_pack.json", "eda_evidence_pack"),
            "summary": ("eda/eda_summary.md", None),
        },
        "reasoning_context": {
            "validation": ("reasoning/validation_result.json", "validation_result"),
            "metric": ("reasoning/metric_result.json", None),
            "leakage": ("reasoning/leakage_result.json", None),
            "leaderboard": ("reasoning/leaderboard_audit.json", None),
        },
        "experiment_planner": {
            "experiment_plan": ("reasoning/experiment_plan.json", "experiment_plan"),
        },
        "skeptical_reviewer": {
            "review": ("reasoning/skeptical_review.json", "skeptical_review"),
        },
        "final_strategy": {
            "strategy": ("final/final_strategy.json", "final_strategy"),
            "diagnostics": ("final/final_synthesis_diagnostics.json", None),
        },
        "final_report": {"report": ("final/final_report.md", None)},
    }.get(stage_id, {})
    return {
        name: artifact_pointer(
            run_dir / relative,
            run_dir=run_dir,
            contract_family=family,
            schema_version="1.0" if family else None,
        )
        for name, (relative, family) in paths.items()
        if (run_dir / relative).is_file()
    }


def dependent_stage_ids(
    stage_id: str, definitions: tuple[StageDefinition, ...]
) -> set[str]:
    result: set[str] = set()
    frontier = {stage_id}
    while frontier:
        parent = frontier.pop()
        direct = {
            definition.stage_id
            for definition in definitions
            if definition.stage_id not in result and parent in definition.dependencies
        }
        result.update(direct)
        frontier.update(direct)
    return result


def invalidated_stage_ids(config: FullRunConfig) -> set[str]:
    definitions = stage_registry(config)
    invalidated = set(config.force_rerun_stages)
    for stage_id in config.force_rerun_stages:
        invalidated.update(dependent_stage_ids(stage_id, definitions))
    return invalidated


def _load_or_create_manifest(run_dir: Path, config: FullRunConfig) -> RunManifest:
    path = run_dir / "run_manifest.json"
    if path.is_file():
        return load_run_manifest(path, run_dir=run_dir).value
    return new_run_manifest(
        run_id=run_dir.name,
        competition_id=config.competition_id,
        config=_config_snapshot(config),
    )


def _persist_state(state: FullRunState, path: Path) -> None:
    write_run_manifest_atomic(state.manifest, path)


def _replace_manifest_stage(manifest: RunManifest, stage_id: str, entry: Any) -> RunManifest:
    stages = dict(manifest.stages)
    stages[StageId(stage_id)] = entry
    return manifest.model_copy(update={"stages": stages})


def _error_record(exc: Exception, stage_id: str) -> StageErrorRecord:
    message = str(exc)[:2000]
    invalid_fields = [str(value) for value in getattr(exc, "field_paths", ())[:20]]
    invalid_ids = [str(value) for value in getattr(exc, "invalid_ids", ())[:50]]
    suggested = getattr(exc, "suggested_rerun_stage", stage_id)
    if str(suggested) not in FULL_RUN_STAGES:
        suggested = stage_id
    return StageErrorRecord(
        error_type=type(exc).__name__,
        message=message,
        recoverable=bool(getattr(exc, "recoverable", False)),
        suggested_rerun_stage=StageId(str(suggested)),
        contract=getattr(exc, "contract", None),
        invalid_fields=invalid_fields,
        invalid_ids=invalid_ids,
    )


def _enabled(config: FullRunConfig, module: str) -> bool:
    profile_defaults = {
        "minimal": set(),
        "standard": {"baseline", "source_claims"},
        "full": {"p1", "baseline", "ablations", "interactions", "slices", "source_claims", "visuals"},
    }
    explicit = {
        "p1": config.enable_p1_modules,
        "baseline": config.enable_baseline,
        "ablations": config.enable_baseline_ablations,
        "interactions": config.enable_interaction_diagnostics,
        "slices": config.enable_slice_diagnostics,
        "source_claims": config.enable_source_claim_validation,
        "visuals": config.enable_visual_diagnostics,
    }
    return explicit[module] or module in profile_defaults[config.profile]


def _validate_config(config: FullRunConfig) -> None:
    if not config.competition_id.strip():
        raise ValueError("competition_id is required")
    invalid = set(config.force_rerun_stages) - set(FULL_RUN_STAGES)
    if invalid:
        raise ValueError(f"Unknown full-run stage IDs: {', '.join(sorted(invalid))}")
    if config.local_dataset_path is not None and not config.local_dataset_path.is_dir():
        raise FileNotFoundError(f"Local dataset path does not exist: {config.local_dataset_path}")


def _resolve_run_dir(config: FullRunConfig) -> Path:
    if config.resume_run_dir is not None:
        return config.resume_run_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = config.output_root / f"{config.competition_id}_{stamp}"
    counter = 2
    while candidate.exists():
        candidate = config.output_root / f"{config.competition_id}_{stamp}_{counter:03d}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _config_snapshot(config: FullRunConfig) -> ManifestConfigSnapshot:
    return ManifestConfigSnapshot(
        competition_url=config.competition_url,
        competition_description=config.competition_description,
        local_dataset_path=str(config.local_dataset_path) if config.local_dataset_path else None,
        download_dataset=config.download_dataset,
        output_root=str(config.output_root),
        profile=config.profile,
        enable_p1_modules=config.enable_p1_modules,
        enable_baseline=config.enable_baseline,
        enable_baseline_ablations=config.enable_baseline_ablations,
        enable_interaction_diagnostics=config.enable_interaction_diagnostics,
        enable_slice_diagnostics=config.enable_slice_diagnostics,
        enable_source_claim_validation=config.enable_source_claim_validation,
        enable_visual_diagnostics=config.enable_visual_diagnostics,
        fail_fast=config.fail_fast,
        # Force-rerun and progress are invocation controls, not semantic artifacts.
        force_rerun_stages=[],
        disable_progress=config.disable_progress,
    )


def _copy_artifacts(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)


def _canonicalize_research_artifact_bundle(research_dir: Path) -> None:
    hypotheses_path = research_dir / "research_hypotheses.json"
    task_plan_path = research_dir / "eda_task_plan.json"
    hypotheses, hypothesis_migration = load_research_hypotheses(hypotheses_path)
    task_plan, task_plan_migration = load_eda_task_plan(task_plan_path, hypotheses=hypotheses)
    migrations = {
        "research_hypotheses": hypothesis_migration,
        "eda_task_plan": task_plan_migration,
    }
    for name, migration in migrations.items():
        if migration.migrated:
            original = research_dir / f"{name}.json"
            backup = research_dir / f"{name}.legacy.json"
            if not backup.exists():
                shutil.copy2(original, backup)
    if hypothesis_migration.migrated:
        write_research_hypotheses_atomic(hypotheses_path, hypotheses)
    if task_plan_migration.migrated:
        write_eda_task_plan_atomic(task_plan_path, task_plan)
    if any(migration.migrated for migration in migrations.values()):
        write_json_atomic(research_dir / "research_artifact_migrations.json", {
            name: {
                "source_schema_version": migration.source_schema_version,
                "target_schema_version": migration.target_schema_version,
                "applied_migrations": migration.applied_migrations,
                "warnings": migration.warnings,
            }
            for name, migration in migrations.items()
            if migration.migrated
        })


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or(path: Path, default: Any) -> Any:
    return _read_json(path) if path.is_file() else default


def _write_summary(run_dir: Path, manifest: RunManifest) -> None:
    config = manifest.config
    command = [
        "python -m kaggle_researcher.main full-run",
        f"--competition-id {manifest.competition_id}",
    ]
    if config.competition_url:
        command.append(f'--competition-url "{config.competition_url}"')
    if config.local_dataset_path:
        command.append(f'--local-dataset-path "{config.local_dataset_path}"')
    lines = [
        "# Full Research Run", "", f"## Competition\n\n`{manifest.competition_id}`", "",
        "## Configuration", "", f"- Profile: `{config.profile}`",
        f"- Dataset download: `{config.download_dataset}`", "", "## Stage Status", "",
    ]
    lines.extend(f"- {stage}: {entry.status.value}" for stage, entry in manifest.stages.items())
    lines.extend(["", "## Final Outputs", ""])
    for name in (
        "final_strategy",
        "final_report",
        "final_synthesis_diagnostics",
    ):
        pointer = getattr(manifest.final_outputs, name)
        if pointer:
            lines.append(f"- {name}: `{pointer.relative_path}`")
    lines.extend(["", "## Reproduction Command", "", "```powershell", " `\n  ".join(command), "```"])
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_log(run_dir: Path) -> None:
    log_path = run_dir / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if any(getattr(handler, "baseFilename", None) == str(log_path) for handler in logger.handlers):
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "FULL_RUN_STAGES", "FullRunConfig", "FullRunResult", "MissingStageDependencyError",
    "dependent_stage_ids", "invalidated_stage_ids", "run_full_research", "stage_registry",
]
