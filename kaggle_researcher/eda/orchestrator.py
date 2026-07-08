from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from kaggle_researcher.eda.config import load_eda_config
from kaggle_researcher.eda.io.artifact_writer import ArtifactWriter
from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.io.dataset_resolver import resolve_dataset
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.hypothesis_evaluator import evaluate_hypotheses
from kaggle_researcher.eda.modules.leakage_checker import check_leakage
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.presets import get_preset
from kaggle_researcher.eda.schemas import (
    EdaEvidencePack,
    EdaRunConfig,
    EdaRunResult,
    EdaTaskPlan,
    ResearchHypotheses,
    competition_ids_match,
)


P1_PLACEHOLDERS = {
    "relationship_evidence": {},
    "drift_evidence": {},
    "baseline_evidence": {},
    "feature_probe_evidence": [],
    "notebook_static_analysis": {},
}


async def run_eda(config: EdaRunConfig) -> EdaRunResult:
    started = time.perf_counter()
    settings = load_eda_config()
    output_root = Path(config.output_dir or settings.eda_runs_dir)
    cache_dir = Path(settings.kaggle_datasets_dir)
    module_statuses: dict[str, str] = {}
    warnings: list[str] = []
    limitations: list[str] = []

    hypotheses = _load_model(config.hypotheses_path, ResearchHypotheses)
    task_plan = _load_model(config.task_plan_path, EdaTaskPlan)
    if not competition_ids_match(hypotheses, task_plan):
        raise ValueError(
            "research_hypotheses.json and eda_task_plan.json target different competitions."
        )
    if hypotheses.competition_id != config.competition_id:
        warnings.append(
            "Config competition_id differs from input JSON; using config "
            "competition_id for run naming."
        )

    writer = ArtifactWriter(output_root)
    run_dir = writer.create_run_dir(config.competition_id)
    writer.copy_input(config.hypotheses_path, "input_research_hypotheses.json")
    writer.copy_input(config.task_plan_path, "input_eda_task_plan.json")

    dataset_path = resolve_dataset(
        competition_id=config.competition_id,
        competition_url=config.competition_url or task_plan.dataset.get("competition_url"),
        local_dataset_path=_local_dataset_path(config, task_plan),
        download=config.download_dataset,
        force_download=config.force_download,
        cache_dir=cache_dir,
    )
    reader = DatasetReader(dataset_path)
    preset = get_preset(config.competition_id)

    file_inventory = build_file_inventory(dataset_path, preset=preset)
    module_statuses["file_inventory"] = "completed"
    writer.write_json("file_inventory.json", file_inventory)

    inferred_schema = infer_schema(file_inventory, reader, preset=preset)
    module_statuses["schema_inferer"] = "completed"
    writer.write_json("inferred_schema.json", inferred_schema)

    table_profiles = profile_tables(
        file_inventory,
        inferred_schema,
        reader,
        sample_rows=config.profile_sample_rows,
        max_full_scan_rows=config.max_profile_rows_full_scan,
    )
    module_statuses["table_profiler"] = "completed"
    writer.write_json("table_profiles.json", table_profiles)

    metric_evidence = analyze_metric(task_plan, inferred_schema, table_profiles)
    module_statuses["metric_analyzer"] = "completed"
    writer.write_json("metric_evidence.json", metric_evidence)

    validation_evidence = analyze_validation(
        inferred_schema,
        table_profiles,
        metric_evidence,
        reader,
    )
    module_statuses["validation_analyzer"] = "completed"
    writer.write_json("validation_evidence.json", validation_evidence)

    leakage_evidence = check_leakage(inferred_schema, validation_evidence, reader)
    module_statuses["leakage_checker"] = "completed"
    writer.write_json("leakage_evidence.json", leakage_evidence)

    for artifact_name, placeholder in P1_PLACEHOLDERS.items():
        module_statuses[_module_name_for_placeholder(artifact_name)] = "skipped"
        writer.write_json(f"{artifact_name}.json", placeholder)
    writer.write_json("module_statuses.json", module_statuses)

    evidence_pack_partial = {
        "file_inventory": file_inventory,
        "inferred_schema": inferred_schema,
        "table_profiles": table_profiles,
        "metric_evidence": metric_evidence,
        "validation_evidence": validation_evidence,
        "leakage_evidence": leakage_evidence,
        **P1_PLACEHOLDERS,
    }
    hypothesis_results = evaluate_hypotheses(
        hypotheses.hypotheses,
        evidence_pack_partial,
        module_statuses=module_statuses,
    )
    writer.write_json("hypothesis_results.json", hypothesis_results)

    recommended_next_actions = build_recommended_next_actions(
        evidence_pack_partial,
        hypothesis_results,
    )
    writer.write_json("recommended_next_actions.json", recommended_next_actions)

    created_at = datetime.now().astimezone().isoformat()
    evidence_pack = EdaEvidencePack(
        competition_id=config.competition_id,
        created_at=created_at,
        run_id=run_dir.name,
        dataset={
            "dataset_path": str(dataset_path),
            "source": "local" if config.local_dataset_path else "cache",
        },
        file_inventory=file_inventory.model_dump(mode="json"),
        inferred_schema=inferred_schema.model_dump(mode="json"),
        table_profiles=[profile.model_dump(mode="json") for profile in table_profiles],
        metric_evidence=metric_evidence.model_dump(mode="json"),
        validation_evidence=validation_evidence.model_dump(mode="json"),
        leakage_evidence=[item.model_dump(mode="json") for item in leakage_evidence],
        relationship_evidence=P1_PLACEHOLDERS["relationship_evidence"],
        drift_evidence=P1_PLACEHOLDERS["drift_evidence"],
        baseline_evidence=P1_PLACEHOLDERS["baseline_evidence"],
        feature_probe_evidence=P1_PLACEHOLDERS["feature_probe_evidence"],
        notebook_static_analysis=P1_PLACEHOLDERS["notebook_static_analysis"],
        hypothesis_results=hypothesis_results,
        recommended_next_actions=recommended_next_actions,
        warnings=warnings,
        limitations=limitations,
        artifacts={
            "run_dir": str(run_dir),
            "module_statuses": module_statuses,
        },
    )
    evidence_pack_path = writer.write_json("eda_evidence_pack.json", evidence_pack)
    summary_path = writer.write_markdown(
        "eda_summary.md",
        _summary_markdown(evidence_pack, module_statuses),
    )

    return EdaRunResult(
        competition_id=config.competition_id,
        run_id=run_dir.name,
        output_dir=run_dir,
        evidence_pack_path=evidence_pack_path,
        summary_path=summary_path,
        module_statuses=module_statuses,
        hypothesis_results_count=len(hypothesis_results),
        warnings=warnings,
        limitations=limitations,
        duration_sec=round(time.perf_counter() - started, 6),
    )


def _load_model(path: Path, model_type: Any) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_type(**payload)


def _local_dataset_path(config: EdaRunConfig, task_plan: EdaTaskPlan) -> Path | None:
    if config.local_dataset_path is not None:
        return config.local_dataset_path
    local_path = task_plan.dataset.get("local_dataset_path")
    return Path(local_path) if local_path else None


def _module_name_for_placeholder(artifact_name: str) -> str:
    return {
        "relationship_evidence": "relationship_inferer",
        "drift_evidence": "drift_analyzer",
        "baseline_evidence": "baseline_runner",
        "feature_probe_evidence": "feature_probe",
        "notebook_static_analysis": "notebook_static_analysis",
    }[artifact_name]


def _summary_markdown(
    evidence_pack: EdaEvidencePack,
    module_statuses: dict[str, str],
) -> str:
    validation = evidence_pack.validation_evidence.get("primary_validation", {})
    metric = evidence_pack.metric_evidence.get("metric_name", "unknown")
    actions = "\n".join(
        f"- {action.priority}: {action.action}"
        for action in evidence_pack.recommended_next_actions
    ) or "- None"
    statuses = "\n".join(f"- {name}: {status}" for name, status in sorted(module_statuses.items()))
    return (
        f"# EDA Summary\n\n"
        f"Competition: `{evidence_pack.competition_id}`\n\n"
        f"Metric: `{metric}`\n\n"
        f"Primary validation: `{validation.get('method', 'unknown')}`\n\n"
        f"## Recommended Actions\n\n{actions}\n\n"
        f"## Module Statuses\n\n{statuses}\n"
    )


__all__ = ["run_eda"]
