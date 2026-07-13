from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.config import load_config
from kaggle_researcher.contracts.research_hypotheses import (
    load_research_hypotheses,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.contracts.eda_task_plan import (
    load_eda_task_plan,
    write_eda_task_plan_atomic,
)
from kaggle_researcher.contracts.pipeline import validate_full_run_artifacts
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaRunConfig, ResearchHypotheses
from kaggle_researcher.progress import ProgressConfig, progress_write
from kaggle_researcher.reasoning.experiment_planner import plan_experiments
from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyResult,
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
from kaggle_researcher.schemas import PlanData, RetrievedDocument, ReviewResult


logger = logging.getLogger(__name__)
Profile = Literal["minimal", "standard", "full"]


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


@dataclass
class FullRunConfig:
    competition_id: str
    competition_url: str | None = None
    competition_description: str = ""
    local_dataset_path: Path | None = None
    download_dataset: bool = True
    output_root: Path = Path("runs")
    profile: Profile = "standard"
    enable_p1_modules: bool = False
    enable_baseline: bool = False
    enable_baseline_ablations: bool = False
    enable_interaction_diagnostics: bool = False
    enable_slice_diagnostics: bool = False
    enable_source_claim_validation: bool = False
    enable_visual_diagnostics: bool = False
    fail_fast: bool = False
    resume_run_dir: Path | None = None
    force_rerun_stages: set[str] = field(default_factory=set)
    disable_progress: bool = False


@dataclass(frozen=True)
class FullRunResult:
    run_dir: Path
    manifest_path: Path
    final_strategy_path: Path
    final_report_path: Path
    status: str


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
    progress = ProgressConfig(enabled=not config.disable_progress)
    manifest = _load_or_create_manifest(run_dir, config)
    if _can_reuse(manifest, config, run_dir):
        return FullRunResult(run_dir, manifest_path, run_dir / "final" / "final_strategy.json", run_dir / "final" / "final_report.md", "reused")

    enabled = [stage for stage in stage_registry(config) if stage.enabled]
    invalidated_stages = invalidated_stage_ids(config)
    for stage in stage_registry(config):
        if not stage.enabled and manifest["stages"][stage.stage_id]["status"] == "pending":
            manifest["stages"][stage.stage_id]["status"] = "skipped"
    manifest["status"] = "running"
    manifest["started_at"] = manifest.get("started_at") or _now()
    _write_manifest(manifest_path, manifest)
    final_strategy_path = run_dir / "final" / "final_strategy.json"
    final_report_path = run_dir / "final" / "final_report.md"
    context: dict[str, Any] = {}

    with tqdm(total=len(enabled), desc="Full research pipeline", unit="stage", disable=progress.disabled) as bar:
        for stage in enabled:
            bar.set_postfix_str(stage.display_name)
            if _can_reuse_stage(manifest, stage.stage_id, config, run_dir, invalidated_stages):
                entry = manifest["stages"][stage.stage_id]
                entry["status"] = "reused"
                entry["reused"] = True
                _write_manifest(manifest_path, manifest)
                bar.update(1)
                continue
            _transition(manifest, manifest_path, stage.stage_id, "running")
            started = time.perf_counter()
            try:
                await _run_stage(stage.stage_id, config, run_dir, context)
            except Exception as exc:
                logger.exception("Full-run stage %s failed", stage.stage_id)
                error = _error_payload(exc)
                _transition(manifest, manifest_path, stage.stage_id, "failed", error=error, duration=time.perf_counter() - started)
                if stage.required or config.fail_fast:
                    manifest["status"] = "failed"
                    manifest["finished_at"] = _now()
                    _write_manifest(manifest_path, manifest)
                    progress_write(f"{stage.display_name} failed: {error['message']}", config=progress)
                    raise
                context.setdefault("limitations", []).append(f"Optional stage {stage.stage_id} failed: {error['message']}")
                _transition(manifest, manifest_path, stage.stage_id, "partial", error=error, duration=time.perf_counter() - started)
            else:
                manifest["stages"][stage.stage_id]["outputs"] = _stage_outputs(stage.stage_id, run_dir)
                _transition(manifest, manifest_path, stage.stage_id, "completed", duration=time.perf_counter() - started)
            bar.update(1)

    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    manifest["final_outputs"] = {
        "final_strategy": str(final_strategy_path.relative_to(run_dir)),
        "final_report": str(final_report_path.relative_to(run_dir)),
    }
    _write_manifest(manifest_path, manifest)
    _write_summary(run_dir, manifest)
    return FullRunResult(run_dir, manifest_path, final_strategy_path, final_report_path, "completed")


async def _run_stage(stage_id: str, config: FullRunConfig, run_dir: Path, context: dict[str, Any]) -> None:
    if stage_id == "input_validation":
        return
    if stage_id == "research_scout":
        await _run_research_scout(config, run_dir)
        return
    if stage_id == "eda_engine":
        await _run_eda(config, run_dir)
        return
    if stage_id == "reasoning_context":
        await _prepare_reasoning(config, run_dir, context)
        return
    if stage_id == "experiment_planner":
        await _plan_experiments(config, run_dir, context)
        return
    if stage_id == "skeptical_reviewer":
        await _review_plan(config, run_dir, context)
        return
    if stage_id == "final_strategy":
        await _synthesize_strategy(config, run_dir, context)
        return
    if stage_id == "final_report":
        _write_final_report(run_dir, context)
        return
    if stage_id == "artifact_validation":
        _validate_artifacts(run_dir)
        return
    raise ValueError(f"Unknown full-run stage: {stage_id}")


async def _run_research_scout(config: FullRunConfig, run_dir: Path) -> None:
    from kaggle_researcher.main import run_research

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
    _copy_artifacts(source, run_dir / "research")
    _canonicalize_research_artifact_bundle(run_dir / "research")


async def _run_eda(config: FullRunConfig, run_dir: Path) -> None:
    research = run_dir / "research"
    result = await run_eda(EdaRunConfig(
        competition_id=config.competition_id,
        competition_url=config.competition_url,
        hypotheses_path=research / "research_hypotheses.json",
        task_plan_path=research / "eda_task_plan.json",
        local_dataset_path=config.local_dataset_path,
        output_dir=run_dir / "eda" / "stage_runs",
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
    target = run_dir / "eda"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.evidence_pack_path, target / "eda_evidence_pack.json")
    shutil.copy2(result.summary_path, target / "eda_summary.md")
    for filename in ("plots_manifest.json", "module_statuses.json"):
        candidate = result.output_dir / filename
        if candidate.is_file():
            shutil.copy2(candidate, target / filename)


async def _prepare_reasoning(config: FullRunConfig, run_dir: Path, context: dict[str, Any]) -> None:
    settings = load_config()
    client = DeepSeekClient(api_key=settings.deepseek_api_key)
    research = run_dir / "research"
    plan_data = PlanData.model_validate(_read_json(research / "plan.json"))
    docs = [RetrievedDocument.model_validate(value) for value in _read_json(research / "retrieved_documents.json")]
    model = settings.deepseek_v4_pro
    metric = await analyze_metric(plan_data=plan_data, retrieved_documents=docs, client=client, model=model)
    validation = await design_validation(competition_desc=config.competition_description, plan_data=plan_data, retrieved_documents=docs, client=client, model=model)
    leakage = await analyze_leakage_risk(competition_desc=config.competition_description, plan_data=plan_data, retrieved_documents=docs, client=client, model=model)
    leaderboard = await audit_leaderboard_risk(competition_desc=config.competition_description, plan_data=plan_data, validation_result=validation, retrieved_documents=docs, client=client, model=model)
    reasoning = run_dir / "reasoning"; reasoning.mkdir(parents=True, exist_ok=True)
    for name, value in (("metric_result", metric), ("validation_result", validation), ("leakage_result", leakage), ("leaderboard_audit", leaderboard)):
        _write_json(reasoning / f"{name}.json", value.model_dump(mode="json"))
    context.update(plan_data=plan_data, docs=docs, client=client, model=model, metric=metric, validation=validation, leakage=leakage, leaderboard=leaderboard)


async def _plan_experiments(config: FullRunConfig, run_dir: Path, context: dict[str, Any]) -> None:
    experiments = await plan_experiments(validation_result=context["validation"], leakage_result=context["leakage"], metric_result=context["metric"], retrieved_documents=context["docs"], client=context["client"], model=context["model"])
    context["experiments"] = experiments
    _write_json(run_dir / "reasoning" / "experiment_plan.json", [item.model_dump(mode="json") for item in experiments])


async def _review_plan(config: FullRunConfig, run_dir: Path, context: dict[str, Any]) -> None:
    experiments = context.get("experiments", [])
    sections = {
        "validation": attach_default_provenance("validation", context["validation"].model_dump(mode="json"), context["docs"]),
        "leakage": attach_default_provenance("leakage", context["leakage"].model_dump(mode="json"), context["docs"]),
        "metric": attach_default_provenance("metric", context["metric"].model_dump(mode="json"), context["docs"]),
        "experiments": attach_default_provenance("experiments", [item.model_dump(mode="json") for item in experiments], context["docs"]),
    }
    result = await review(draft_sections=sections, retrieved_documents=context["docs"], client=context["client"], model=context["model"], artifact_dir=run_dir / "reasoning")
    context["review"] = result
    _write_json(run_dir / "reasoning" / "skeptical_review.json", result.model_dump(mode="json"))


async def _synthesize_strategy(config: FullRunConfig, run_dir: Path, context: dict[str, Any]) -> None:
    pack = EdaEvidencePack.model_validate(_read_json(run_dir / "eda" / "eda_evidence_pack.json"))
    hypotheses, _ = load_research_hypotheses(run_dir / "research" / "research_hypotheses.json")
    reasoning_outputs = {name: value.model_dump(mode="json") for name, value in (("metric", context["metric"]), ("validation", context["validation"]), ("leakage", context["leakage"]), ("leaderboard", context["leaderboard"]))}
    if context.get("experiments"):
        reasoning_outputs["experiments"] = [item.model_dump(mode="json") for item in context["experiments"]]
    if context.get("review"):
        reasoning_outputs["review"] = context["review"].model_dump(mode="json")
    result = await synthesize_final_strategy(competition_desc=config.competition_description, plan_data=context["plan_data"], retrieved_documents=context["docs"], domain_patterns=_read_json_or(run_dir / "research" / "domain_patterns.json", []), research_hypotheses=hypotheses, eda_evidence_pack=pack, eda_summary_text=(run_dir / "eda" / "eda_summary.md").read_text(encoding="utf-8"), reasoning_outputs=reasoning_outputs, client=context["client"], model=context["model"])
    if context.get("limitations"):
        result.limitations.extend(context["limitations"])
    context["final_strategy"] = result
    final = run_dir / "final"; final.mkdir(parents=True, exist_ok=True)
    _write_json(final / "final_strategy.json", result.model_dump(mode="json"))
    (final / "final_strategy.md").write_text(render_final_strategy(result), encoding="utf-8")
    (final / "final_strategy_summary.md").write_text(render_final_strategy_summary(result), encoding="utf-8")


def _write_final_report(run_dir: Path, context: dict[str, Any]) -> None:
    result = context.get("final_strategy") or FinalStrategyResult.model_validate(_read_json(run_dir / "final" / "final_strategy.json"))
    (run_dir / "final" / "final_report.md").write_text(render_final_strategy(result), encoding="utf-8")


def _validate_artifacts(run_dir: Path) -> None:
    validate_full_run_artifacts(run_dir)


def _enabled(config: FullRunConfig, module: str) -> bool:
    profile_defaults = {
        "minimal": set(),
        "standard": {"baseline", "source_claims"},
        "full": {"p1", "baseline", "ablations", "interactions", "slices", "source_claims", "visuals"},
    }
    explicit = {"p1": config.enable_p1_modules, "baseline": config.enable_baseline, "ablations": config.enable_baseline_ablations, "interactions": config.enable_interaction_diagnostics, "slices": config.enable_slice_diagnostics, "source_claims": config.enable_source_claim_validation, "visuals": config.enable_visual_diagnostics}
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
        candidate = config.output_root / f"{config.competition_id}_{stamp}_{counter:03d}"; counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _load_or_create_manifest(run_dir: Path, config: FullRunConfig) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if path.is_file():
        return _read_json(path)
    return {"schema_version": "1.0", "run_id": run_dir.name, "competition_id": config.competition_id, "started_at": None, "finished_at": None, "status": "pending", "config": _config_payload(config), "stages": {stage: {"status": "pending", "attempt": 0, "outputs": {}, "error": None} for stage in FULL_RUN_STAGES}, "final_outputs": {}}


def _can_reuse(manifest: dict[str, Any], config: FullRunConfig, run_dir: Path) -> bool:
    if config.force_rerun_stages or manifest.get("status") != "completed" or manifest.get("config") != _config_payload(config):
        return False
    try:
        _validate_artifacts(run_dir)
    except Exception:
        return False
    return all(value.get("status") in {"completed", "skipped", "reused"} for value in manifest.get("stages", {}).values())


def _can_reuse_stage(
    manifest: dict[str, Any],
    stage: str,
    config: FullRunConfig,
    run_dir: Path,
    invalidated_stages: set[str] | None = None,
) -> bool:
    invalidated = invalidated_stages if invalidated_stages is not None else invalidated_stage_ids(config)
    if stage not in {"research_scout", "eda_engine"} or stage in invalidated:
        return False
    entry = manifest["stages"].get(stage, {})
    if entry.get("status") not in {"completed", "reused"}:
        return False
    outputs = _stage_outputs(stage, run_dir)
    if not outputs or not all((run_dir / path).is_file() for path in outputs.values()):
        return False
    try:
        if stage == "research_scout":
            hypotheses, _ = load_research_hypotheses(run_dir / "research" / "research_hypotheses.json")
            load_eda_task_plan(run_dir / "research" / "eda_task_plan.json", hypotheses=hypotheses)
        else:
            EdaEvidencePack.model_validate(_read_json(run_dir / "eda" / "eda_evidence_pack.json"))
    except Exception:
        return False
    return True


def dependent_stage_ids(
    stage_id: str,
    definitions: tuple[StageDefinition, ...],
) -> set[str]:
    """Return the transitive dependents of a stage from the canonical registry."""

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


def _transition(manifest: dict[str, Any], path: Path, stage: str, status: str, *, error: dict[str, Any] | None = None, duration: float | None = None) -> None:
    entry = manifest["stages"][stage]
    if status == "running":
        entry["attempt"] = int(entry.get("attempt", 0)) + 1; entry["started_at"] = _now()
    entry["status"] = status
    if duration is not None: entry["duration_sec"] = round(duration, 3); entry["finished_at"] = _now()
    if error is not None: entry["error"] = error
    _write_manifest(path, manifest)


def _error_payload(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    if type(exc).__name__ == "ReasoningResponseValidationError":
        return {
            "error_type": "ReasoningResponseValidationError",
            "stage": getattr(exc, "stage", "reasoning_context"),
            "result_model": getattr(exc, "result_model", None),
            "invalid_fields": [
                {"path": ".".join(str(part) for part in error.get("loc", ())), "reason": error.get("type")}
                for error in getattr(exc, "validation_errors", [])[:8]
            ],
            "message": message,
            "recoverable": True,
            "suggested_action": "Resume after correcting the reasoning response contract or force-rerun reasoning_context.",
        }
    if hasattr(exc, "issues") and hasattr(exc, "field_paths"):
        return {
            "error_type": type(exc).__name__,
            "stage": getattr(exc, "stage", "artifact_validation"),
            "contract": getattr(exc, "contract", "full_research_pipeline"),
            "invalid_fields": list(getattr(exc, "field_paths", ()))[:8],
            "invalid_ids": list(getattr(exc, "invalid_ids", ()))[:12],
            "message": message,
            "recoverable": getattr(exc, "recoverable", True),
            "suggested_action": f"Force-rerun {getattr(exc, 'suggested_rerun_stage', 'final_strategy')}.",
        }
    evidence = "unknown evidence_ids" in message.lower()
    return {"error_type": "EvidenceReferenceValidationError" if evidence else type(exc).__name__, "message": message, "recoverable": evidence, "suggested_action": "Resume after fixing evidence aliases or force-rerun experiment_planner." if evidence else "Inspect logs/pipeline.log and resume the run after resolving the failure."}


def _copy_artifacts(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file(): shutil.copy2(item, target / item.name)


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
        if not migration.migrated:
            continue
        original = research_dir / f"{name}.json"
        backup = research_dir / f"{name}.legacy.json"
        if not backup.exists():
            shutil.copy2(original, backup)
    if hypothesis_migration.migrated:
        write_research_hypotheses_atomic(hypotheses_path, hypotheses)
    if task_plan_migration.migrated:
        write_eda_task_plan_atomic(task_plan_path, task_plan)
    if any(migration.migrated for migration in migrations.values()):
        _write_json(research_dir / "research_artifact_migrations.json", {
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    config = manifest.get("config", {})
    command = ["python -m kaggle_researcher.main full-run", f"--competition-id {manifest['competition_id']}"]
    if config.get("competition_url"):
        command.append(f'--competition-url "{config["competition_url"]}"')
    if config.get("local_dataset_path"):
        command.append(f'--local-dataset-path "{config["local_dataset_path"]}"')
    lines = ["# Full Research Run", "", f"## Competition\n\n`{manifest['competition_id']}`", "", "## Configuration", "", f"- Profile: `{config.get('profile', 'standard')}`", f"- Dataset download: `{config.get('download_dataset')}`", "", "## Stage Status", ""]
    lines.extend(f"- {stage}: {value['status']}" for stage, value in manifest["stages"].items())
    lines.extend(["", "## Final Outputs", ""])
    lines.extend(f"- {name}: `{path}`" for name, path in manifest.get("final_outputs", {}).items())
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


def _config_payload(config: FullRunConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("local_dataset_path", "output_root", "resume_run_dir"):
        if value[key] is not None: value[key] = str(value[key])
    value["force_rerun_stages"] = sorted(value["force_rerun_stages"])
    value.pop("resume_run_dir")
    return value


def _stage_outputs(stage: str, run_dir: Path) -> dict[str, str]:
    outputs = {
        "research_scout": {"research_hypotheses": "research/research_hypotheses.json", "eda_task_plan": "research/eda_task_plan.json"},
        "eda_engine": {"evidence_pack": "eda/eda_evidence_pack.json", "summary": "eda/eda_summary.md"},
        "reasoning_context": {"validation": "reasoning/validation_result.json", "metric": "reasoning/metric_result.json"},
        "experiment_planner": {"experiment_plan": "reasoning/experiment_plan.json"},
        "skeptical_reviewer": {"review": "reasoning/skeptical_review.json"},
        "final_strategy": {"strategy": "final/final_strategy.json"},
        "final_report": {"report": "final/final_report.md"},
    }
    result = {name: path for name, path in outputs.get(stage, {}).items() if (run_dir / path).exists()}
    migration = "research/research_artifact_migrations.json"
    if stage == "research_scout" and (run_dir / migration).is_file():
        result["artifact_migrations"] = migration
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
