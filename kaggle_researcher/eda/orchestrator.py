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
from kaggle_researcher.eda.modules.baseline_runner import run_baseline
from kaggle_researcher.eda.modules.drift_analyzer import analyze_drift
from kaggle_researcher.eda.modules.feature_probe import probe_feature_families
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.hypothesis_evaluator import evaluate_hypotheses
from kaggle_researcher.eda.modules.leakage_checker import check_leakage
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.notebook_static_analyzer import analyze_notebooks_static
from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.modules.relationship_inferer import infer_relationships
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
P1_MODULES = (
    "relationship_inferer",
    "drift_analyzer",
    "baseline_runner",
    "feature_probe",
    "notebook_static_analysis",
)


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

    p1_evidence = _run_p1_modules(
        config=config,
        task_plan=task_plan,
        writer=writer,
        file_inventory=file_inventory,
        inferred_schema=inferred_schema,
        table_profiles=table_profiles,
        metric_evidence=metric_evidence,
        validation_evidence=validation_evidence,
        leakage_evidence=leakage_evidence,
        reader=reader,
        module_statuses=module_statuses,
        warnings=warnings,
    )
    writer.write_json("module_statuses.json", module_statuses)

    evidence_pack_partial = {
        "file_inventory": file_inventory,
        "inferred_schema": inferred_schema,
        "table_profiles": table_profiles,
        "metric_evidence": metric_evidence,
        "validation_evidence": validation_evidence,
        "leakage_evidence": leakage_evidence,
        **p1_evidence,
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
        relationship_evidence=p1_evidence["relationship_evidence"],
        drift_evidence=p1_evidence["drift_evidence"],
        baseline_evidence=p1_evidence["baseline_evidence"],
        feature_probe_evidence=p1_evidence["feature_probe_evidence"],
        notebook_static_analysis=p1_evidence["notebook_static_analysis"],
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


def _run_p1_modules(
    *,
    config: EdaRunConfig,
    task_plan: EdaTaskPlan,
    writer: ArtifactWriter,
    file_inventory: Any,
    inferred_schema: Any,
    table_profiles: list[Any],
    metric_evidence: Any,
    validation_evidence: Any,
    leakage_evidence: list[Any],
    reader: DatasetReader,
    module_statuses: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    p1_evidence: dict[str, Any] = {
        key: _copy_placeholder(value) for key, value in P1_PLACEHOLDERS.items()
    }

    for module_name in P1_MODULES:
        artifact_name = _artifact_name_for_module(module_name)
        if not _should_run_p1_module(module_name, config, task_plan):
            module_statuses[module_name] = "skipped"
            writer.write_json(f"{artifact_name}.json", p1_evidence[artifact_name])
            continue
        if module_name == "baseline_runner" and not config.enable_baseline:
            skipped = _skipped_p1_payload(
                module_name,
                "Baseline runner requires enable_baseline=true.",
            )
            module_statuses[module_name] = "skipped"
            p1_evidence[artifact_name] = skipped
            writer.write_json(f"{artifact_name}.json", skipped)
            continue

        try:
            if module_name == "relationship_inferer":
                result = infer_relationships(inferred_schema, file_inventory, reader)
            elif module_name == "drift_analyzer":
                result = analyze_drift(
                    inferred_schema,
                    validation_evidence,
                    reader,
                    max_rows=config.max_adversarial_rows,
                    random_seed=config.random_seed,
                )
            elif module_name == "baseline_runner":
                result = run_baseline(
                    inferred_schema,
                    validation_evidence,
                    metric_evidence,
                    leakage_evidence,
                    reader,
                    output_dir=writer.artifact_path("baseline"),
                    max_rows=config.max_baseline_rows,
                    random_seed=config.random_seed,
                )
            elif module_name == "feature_probe":
                result = probe_feature_families(
                    inferred_schema,
                    table_profiles,
                    p1_evidence["relationship_evidence"],
                    leakage_evidence,
                    p1_evidence["baseline_evidence"],
                    metric_evidence=metric_evidence.model_dump(mode="json"),
                )
            else:
                result = analyze_notebooks_static(
                    [],
                    output_dir=writer.artifact_path("notebooks"),
                )
            module_statuses[module_name] = _status_from_p1_result(result)
            p1_evidence[artifact_name] = result
            writer.write_json(f"{artifact_name}.json", result)
        except Exception as exc:
            failed = _failed_p1_payload(module_name, exc)
            module_statuses[module_name] = "failed"
            p1_evidence[artifact_name] = (
                [failed] if artifact_name == "feature_probe_evidence" else failed
            )
            writer.write_json(f"{artifact_name}.json", p1_evidence[artifact_name])
            warnings.append(f"{module_name} failed: {exc}")

    return p1_evidence


def _should_run_p1_module(
    module_name: str,
    config: EdaRunConfig,
    task_plan: EdaTaskPlan,
) -> bool:
    if module_name in _normalised_modules(config.skip_modules):
        return False
    explicit_modules = _normalised_modules(config.modules or [])
    if module_name in explicit_modules:
        return True
    if _artifact_name_for_module(module_name) in explicit_modules:
        return True
    if module_name == "notebook_static_analysis" and config.enable_notebook_static_analysis:
        return True
    return bool(config.enable_p1_modules)


def _status_from_p1_result(result: Any) -> str:
    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        if status in {"skipped", "not_testable"}:
            return "skipped"
        if status == "failed":
            return "failed"
    return "completed"


def _skipped_p1_payload(module_name: str, reason: str) -> dict[str, Any]:
    return {"status": "skipped", "module": module_name, "reason": reason}


def _failed_p1_payload(module_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "module": module_name,
        "error_message": str(exc),
    }


def _normalised_modules(values: list[str] | tuple[str, ...]) -> set[str]:
    return {str(value).strip().lower() for value in values}


def _copy_placeholder(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _artifact_name_for_module(module_name: str) -> str:
    return {
        "relationship_inferer": "relationship_evidence",
        "drift_analyzer": "drift_evidence",
        "baseline_runner": "baseline_evidence",
        "feature_probe": "feature_probe_evidence",
        "notebook_static_analysis": "notebook_static_analysis",
    }[module_name]


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
    p1_summary = _p1_summary_markdown(evidence_pack)
    return (
        f"# EDA Summary\n\n"
        f"Competition: `{evidence_pack.competition_id}`\n\n"
        f"Metric: `{metric}`\n\n"
        f"Primary validation: `{validation.get('method', 'unknown')}`\n\n"
        f"{p1_summary}"
        f"## Recommended Actions\n\n{actions}\n\n"
        f"## Module Statuses\n\n{statuses}\n"
    )


def _p1_summary_markdown(evidence_pack: EdaEvidencePack) -> str:
    sections: list[str] = []
    relationship = evidence_pack.relationship_evidence
    if relationship:
        relationships = relationship.get("relationships", [])
        sections.append(f"Relationships: `{len(relationships)}` checked")
    drift = evidence_pack.drift_evidence
    if drift:
        sections.append(f"Drift severity: `{drift.get('severity', drift.get('status', 'unknown'))}`")
    baseline = evidence_pack.baseline_evidence
    if baseline:
        sections.append(f"Baseline: `{baseline.get('status', 'unknown')}`")
    feature_probe = evidence_pack.feature_probe_evidence
    if feature_probe:
        high_potential = sum(
            1 for item in feature_probe if item.get("status") == "high_potential"
        )
        sections.append(f"Feature probes: `{len(feature_probe)}` families, `{high_potential}` high potential")
    notebook = evidence_pack.notebook_static_analysis
    if notebook:
        sections.append(f"Notebook static analysis: `{notebook.get('status', 'unknown')}`")
    if not sections:
        return ""
    body = "\n".join(f"- {section}" for section in sections)
    return f"## P1 Evidence\n\n{body}\n\n"


__all__ = ["run_eda"]
