from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from kaggle_researcher.eda.config import load_eda_config
from kaggle_researcher.eda.io.artifact_writer import ArtifactWriter
from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.io.dataset_resolver import resolve_dataset
from kaggle_researcher.eda.modules.baseline_ablations import run_baseline_ablations
from kaggle_researcher.eda.modules.baseline_runner import run_baseline
from kaggle_researcher.eda.modules.drift_analyzer import analyze_drift
from kaggle_researcher.eda.modules.feature_diagnostics import diagnose_features
from kaggle_researcher.eda.modules.feature_probe import probe_feature_families
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.hypothesis_evaluator import evaluate_hypotheses
from kaggle_researcher.eda.modules.interaction_diagnostics import diagnose_interactions
from kaggle_researcher.eda.modules.leakage_checker import check_leakage
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.notebook_static_analyzer import analyze_notebooks_static
from kaggle_researcher.eda.modules.recommendations import build_recommended_next_actions
from kaggle_researcher.eda.modules.relationship_inferer import infer_relationships
from kaggle_researcher.eda.modules.risk_register import build_eda_risk_register, risk_summary
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.source_claim_validation import collect_source_claims, validate_source_claims
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.target_diagnostics import diagnose_target
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
from kaggle_researcher.eda.summary import build_eda_summary


P1_PLACEHOLDERS = {
    "relationship_evidence": {},
    "drift_evidence": {},
    "baseline_evidence": {},
    "baseline_ablation_evidence": {},
    "interaction_diagnostics": {},
    "source_claim_validation": {},
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
    module_status_details: dict[str, dict[str, Any]] = {}
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
    dataset_path: Path | None = None
    file_inventory: Any = None
    inferred_schema: Any = None
    table_profiles: list[Any] = []
    metric_evidence: Any = None
    validation_evidence: Any = None
    leakage_evidence: list[Any] = []
    feature_diagnostics: dict[str, Any] = {}
    target_diagnostics: dict[str, Any] = {}
    interaction_diagnostics: dict[str, Any] = {}
    source_claim_validation: dict[str, Any] = {}
    eda_strategy_hints: dict[str, list[dict[str, Any]]] = {}
    p1_evidence: dict[str, Any] = {
        key: _copy_placeholder(value) for key, value in P1_PLACEHOLDERS.items()
    }

    def write_partial_failure(exc: Exception) -> None:
        sanitized = _sanitize_error(exc)
        warnings.append(f"EDA run failed before completion: {sanitized}")
        limitations.append("Partial EDA evidence pack; unsupported conclusions were omitted.")
        writer.write_module_statuses(module_status_details)
        _write_partial_evidence_pack(
            config=config,
            writer=writer,
            run_dir=run_dir,
            dataset_path=dataset_path,
            file_inventory=file_inventory,
            inferred_schema=inferred_schema,
            table_profiles=table_profiles,
            metric_evidence=metric_evidence,
            validation_evidence=validation_evidence,
            leakage_evidence=leakage_evidence,
            feature_diagnostics=feature_diagnostics,
            target_diagnostics=target_diagnostics,
            interaction_diagnostics=interaction_diagnostics,
            source_claim_validation=source_claim_validation,
            eda_strategy_hints=eda_strategy_hints,
            p1_evidence=p1_evidence,
            module_statuses=module_statuses,
            module_status_details=module_status_details,
            warnings=warnings,
            limitations=limitations,
        )

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

    try:
        file_inventory = _run_blocking_module(
            "file_inventory",
            module_statuses,
            module_status_details,
            lambda: build_file_inventory(dataset_path, preset=preset),
        )
        _ensure_supported_data_files(file_inventory)
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("file_inventory.json", file_inventory)

    try:
        inferred_schema = _run_blocking_module(
            "schema_inferer",
            module_statuses,
            module_status_details,
            lambda: infer_schema(
                file_inventory,
                reader,
                preset=preset,
                task_type_hint=task_plan.task_type,
                metric_hint=str((task_plan.metric or {}).get("name") or ""),
            ),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("inferred_schema.json", inferred_schema)

    try:
        table_profiles = _run_blocking_module(
            "table_profiler",
            module_statuses,
            module_status_details,
            lambda: profile_tables(
                file_inventory,
                inferred_schema,
                reader,
                sample_rows=config.profile_sample_rows,
                max_full_scan_rows=config.max_profile_rows_full_scan,
                max_table_bytes=config.max_table_bytes,
                max_column_cardinality_scan_rows=config.max_column_cardinality_scan_rows,
            ),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("table_profiles.json", table_profiles)

    try:
        metric_evidence = _run_blocking_module(
            "metric_analyzer",
            module_statuses,
            module_status_details,
            lambda: analyze_metric(task_plan, inferred_schema, table_profiles),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("metric_evidence.json", metric_evidence)

    try:
        validation_evidence = _run_blocking_module(
            "validation_analyzer",
            module_statuses,
            module_status_details,
            lambda: analyze_validation(
                inferred_schema,
                table_profiles,
                metric_evidence,
                reader,
            ),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("validation_evidence.json", validation_evidence)

    try:
        leakage_evidence = _run_blocking_module(
            "leakage_checker",
            module_statuses,
            module_status_details,
            lambda: check_leakage(inferred_schema, validation_evidence, reader),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
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
        module_status_details=module_status_details,
        warnings=warnings,
    )
    writer.write_module_statuses(module_status_details)

    try:
        feature_diagnostics = _run_blocking_module(
            "feature_diagnostics",
            module_statuses,
            module_status_details,
            lambda: diagnose_features(
                inferred_schema,
                table_profiles,
                metric_evidence,
                p1_evidence.get("drift_evidence"),
                reader,
                max_rows=min(config.profile_sample_rows, 200_000),
            ),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("feature_diagnostics.json", feature_diagnostics)

    baseline_ablation_evidence = _run_baseline_ablation_module(
        config=config,
        writer=writer,
        inferred_schema=inferred_schema,
        validation_evidence=validation_evidence,
        metric_evidence=metric_evidence,
        leakage_evidence=leakage_evidence,
        feature_diagnostics=feature_diagnostics,
        reader=reader,
        baseline_evidence=p1_evidence.get("baseline_evidence"),
        module_statuses=module_statuses,
        module_status_details=module_status_details,
        warnings=warnings,
    )
    p1_evidence["baseline_ablation_evidence"] = baseline_ablation_evidence
    writer.write_module_statuses(module_status_details)

    try:
        target_diagnostics = _run_blocking_module(
            "target_diagnostics",
            module_statuses,
            module_status_details,
            lambda: diagnose_target(
                inferred_schema,
                metric_evidence,
                validation_evidence,
                feature_diagnostics,
                table_profiles,
                reader,
                max_rows=min(config.profile_sample_rows, 200_000),
            ),
        )
    except Exception as exc:
        write_partial_failure(exc)
        raise
    writer.write_json("target_diagnostics.json", target_diagnostics)

    interaction_diagnostics = _run_interaction_diagnostics_module(
        config=config,
        task_plan=task_plan,
        writer=writer,
        inferred_schema=inferred_schema,
        table_profiles=table_profiles,
        metric_evidence=metric_evidence,
        reader=reader,
        feature_diagnostics=feature_diagnostics,
        target_diagnostics=target_diagnostics,
        drift_evidence=p1_evidence.get("drift_evidence"),
        baseline_ablation_evidence=p1_evidence.get("baseline_ablation_evidence"),
        leakage_evidence=leakage_evidence,
        module_statuses=module_statuses,
        module_status_details=module_status_details,
        warnings=warnings,
    )
    p1_evidence["interaction_diagnostics"] = interaction_diagnostics
    writer.write_module_statuses(module_status_details)

    source_claim_validation = _run_source_claim_validation_module(
        config=config, task_plan=task_plan, writer=writer, hypotheses=hypotheses,
        evidence_pack={
            "inferred_schema": inferred_schema.model_dump(mode="json"), "metric_evidence": metric_evidence.model_dump(mode="json"),
            "validation_evidence": validation_evidence.model_dump(mode="json"), "leakage_evidence": [item.model_dump(mode="json") for item in leakage_evidence],
            "feature_diagnostics": feature_diagnostics, "target_diagnostics": target_diagnostics,
            "interaction_diagnostics": interaction_diagnostics, **p1_evidence,
        }, module_statuses=module_statuses, module_status_details=module_status_details, warnings=warnings,
    )
    p1_evidence["source_claim_validation"] = source_claim_validation
    writer.write_module_statuses(module_status_details)

    evidence_pack_partial = {
        "file_inventory": file_inventory,
        "inferred_schema": inferred_schema,
        "table_profiles": table_profiles,
        "metric_evidence": metric_evidence,
        "validation_evidence": validation_evidence,
        "leakage_evidence": leakage_evidence,
        "feature_diagnostics": feature_diagnostics,
        "target_diagnostics": target_diagnostics,
        "interaction_diagnostics": interaction_diagnostics,
        "source_claim_validation": source_claim_validation,
        **p1_evidence,
    }
    eda_risk_register = build_eda_risk_register(
        inferred_schema=inferred_schema.model_dump(mode="json"),
        metric_evidence=metric_evidence.model_dump(mode="json"),
        validation_evidence=validation_evidence.model_dump(mode="json"),
        target_diagnostics=target_diagnostics,
        leakage_evidence=[item.model_dump(mode="json") for item in leakage_evidence],
        drift_evidence=p1_evidence.get("drift_evidence"),
        relationship_evidence=p1_evidence.get("relationship_evidence"),
        feature_probe_evidence=p1_evidence.get("feature_probe_evidence"),
        feature_diagnostics=feature_diagnostics,
        baseline_evidence=p1_evidence.get("baseline_evidence"),
        baseline_ablation_evidence=p1_evidence.get("baseline_ablation_evidence"),
        interaction_diagnostics=interaction_diagnostics,
        source_claim_validation=source_claim_validation,
        notebook_static_analysis=p1_evidence.get("notebook_static_analysis"),
    )
    evidence_pack_partial["eda_risk_register"] = eda_risk_register
    evidence_pack_partial["risk_summary"] = risk_summary(eda_risk_register)
    writer.write_json("eda_risk_register.json", eda_risk_register)
    eda_strategy_hints = build_eda_strategy_hints(evidence_pack_partial)
    evidence_pack_partial["eda_strategy_hints"] = eda_strategy_hints
    writer.write_json("eda_strategy_hints.json", eda_strategy_hints)
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
        file_inventory=_file_inventory_payload(file_inventory, inferred_schema),
        inferred_schema=inferred_schema.model_dump(mode="json"),
        table_profiles=[profile.model_dump(mode="json") for profile in table_profiles],
        metric_evidence=metric_evidence.model_dump(mode="json"),
        validation_evidence=validation_evidence.model_dump(mode="json"),
        leakage_evidence=[item.model_dump(mode="json") for item in leakage_evidence],
        relationship_evidence=p1_evidence["relationship_evidence"],
        drift_evidence=p1_evidence["drift_evidence"],
        baseline_evidence=p1_evidence["baseline_evidence"],
        baseline_ablation_evidence=p1_evidence["baseline_ablation_evidence"],
        feature_probe_evidence=p1_evidence["feature_probe_evidence"],
        feature_diagnostics=feature_diagnostics,
        target_diagnostics=target_diagnostics,
        interaction_diagnostics=interaction_diagnostics,
        source_claim_validation=source_claim_validation,
        eda_risk_register=eda_risk_register,
        risk_summary=risk_summary(eda_risk_register),
        eda_strategy_hints=eda_strategy_hints,
        notebook_static_analysis=p1_evidence["notebook_static_analysis"],
        hypothesis_results=hypothesis_results,
        recommended_next_actions=recommended_next_actions,
        warnings=warnings,
        limitations=limitations,
        artifacts={
            "run_dir": str(run_dir),
            "module_statuses": module_statuses,
            "module_status_details": module_status_details,
        },
    )
    evidence_pack_path = writer.write_json("eda_evidence_pack.json", evidence_pack)
    summary_path = writer.write_markdown(
        "eda_summary.md",
        build_eda_summary(evidence_pack),
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


def _ensure_supported_data_files(file_inventory: Any) -> None:
    supported_extensions = {".csv", ".parquet", ".json", ".jsonl"}
    files = getattr(file_inventory, "files", [])
    supported_files = [
        file
        for file in files
        if getattr(file, "can_read", False)
        and getattr(file, "extension", "").lower() in supported_extensions
    ]
    if supported_files:
        return
    archive_files = [
        getattr(file, "path", getattr(file, "name", "unknown"))
        for file in files
        if getattr(file, "extension", "").lower() == ".zip"
    ]
    if archive_files:
        raise ValueError(
            "No supported tabular data files were found after dataset discovery. "
            f"Archive files are present but not extracted or contain no supported data: {archive_files}."
        )
    raise ValueError(
        "No supported tabular data files were found in the dataset directory. "
        "Expected at least one .csv, .parquet, .json, or .jsonl file."
    )


def _file_inventory_payload(file_inventory: Any, inferred_schema: Any) -> dict[str, Any]:
    payload = file_inventory.model_dump(mode="json") if hasattr(file_inventory, "model_dump") else _jsonable_dict(file_inventory)
    payload["reconciled_table_roles"] = _reconciled_table_roles(inferred_schema)
    return payload


def _reconciled_table_roles(inferred_schema: Any) -> dict[str, str]:
    if inferred_schema is None:
        return {}
    roles: dict[str, str] = {}
    train_base = getattr(inferred_schema, "train_base_table", None)
    test_base = getattr(inferred_schema, "test_base_table", None)
    sample = getattr(inferred_schema, "sample_submission_table", None)
    for table in getattr(inferred_schema, "tables", []):
        if table.path == train_base:
            roles[table.path] = "train_base"
        elif table.path == test_base:
            roles[table.path] = "test_base"
        elif table.path == sample:
            roles[table.path] = "sample_submission"
        elif table.role in {"train", "test"}:
            suffix = table.table_type if table.table_type != "unknown" else "table"
            roles[table.path] = f"{table.role}_{suffix}"
        else:
            roles[table.path] = table.role
    return roles


def _local_dataset_path(config: EdaRunConfig, task_plan: EdaTaskPlan) -> Path | None:
    if config.local_dataset_path is not None:
        return config.local_dataset_path
    local_path = task_plan.dataset.get("local_dataset_path")
    return Path(local_path) if local_path else None


def _run_blocking_module(
    module_name: str,
    module_statuses: dict[str, str],
    module_status_details: dict[str, dict[str, Any]],
    factory: Any,
) -> Any:
    started_at = _now_iso()
    started_perf = time.perf_counter()
    try:
        result = factory()
    except Exception as exc:
        _record_module_status(
            module_name,
            "failed",
            module_statuses=module_statuses,
            module_status_details=module_status_details,
            legacy_status="failed",
            started_at=started_at,
            started_perf=started_perf,
            error_message=_sanitize_error(exc),
        )
        raise
    _record_module_status(
        module_name,
        "success",
        module_statuses=module_statuses,
        module_status_details=module_status_details,
        legacy_status="completed",
        started_at=started_at,
        started_perf=started_perf,
    )
    return result


def _record_module_status(
    module_name: str,
    status: str,
    *,
    module_statuses: dict[str, str],
    module_status_details: dict[str, dict[str, Any]],
    legacy_status: str,
    started_at: str | None = None,
    started_perf: float | None = None,
    error_message: str | None = None,
) -> None:
    finished_at = _now_iso()
    module_statuses[module_name] = legacy_status
    module_status_details[module_name] = {
        "module": module_name,
        "status": status,
        "started_at": started_at or finished_at,
        "finished_at": finished_at,
        "duration_sec": (
            0.0
            if started_perf is None
            else round(max(0.0, time.perf_counter() - started_perf), 6)
        ),
        "error_message": error_message,
    }


def _detail_status_from_legacy(legacy_status: str) -> str:
    if legacy_status == "completed":
        return "success"
    if legacy_status in {"failed", "skipped"}:
        return legacy_status
    return "success"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _sanitize_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    message = re.sub(r"[\r\n\t]+", " ", message)
    message = re.sub(
        r"(?i)\b(api[_-]?key|kaggle[_-]?key|token|secret|password|passwd|pwd|key)\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}",
        "Bearer [REDACTED]",
        message,
    )
    message = re.sub(r"\s{2,}", " ", message).strip()
    if len(message) > 500:
        message = f"{message[:497]}..."
    return message


def _write_partial_evidence_pack(
    *,
    config: EdaRunConfig,
    writer: ArtifactWriter,
    run_dir: Path,
    dataset_path: Path | None,
    file_inventory: Any,
    inferred_schema: Any,
    table_profiles: list[Any],
    metric_evidence: Any,
    validation_evidence: Any,
    leakage_evidence: list[Any],
    feature_diagnostics: dict[str, Any],
    target_diagnostics: dict[str, Any],
    interaction_diagnostics: dict[str, Any],
    source_claim_validation: dict[str, Any],
    eda_strategy_hints: dict[str, list[dict[str, Any]]],
    p1_evidence: dict[str, Any],
    module_statuses: dict[str, str],
    module_status_details: dict[str, dict[str, Any]],
    warnings: list[str],
    limitations: list[str],
) -> None:
    evidence_pack = EdaEvidencePack(
        competition_id=config.competition_id,
        created_at=_now_iso(),
        run_id=run_dir.name,
        dataset={
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "source": "local" if config.local_dataset_path else "cache",
            "partial": True,
        },
        file_inventory=_jsonable_dict(file_inventory),
        inferred_schema=_jsonable_dict(inferred_schema),
        table_profiles=_jsonable_list(table_profiles),
        metric_evidence=_jsonable_dict(metric_evidence),
        validation_evidence=_jsonable_dict(validation_evidence),
        leakage_evidence=_jsonable_list(leakage_evidence),
        relationship_evidence=_jsonable_dict(p1_evidence.get("relationship_evidence")),
        drift_evidence=_jsonable_dict(p1_evidence.get("drift_evidence")),
        baseline_evidence=_jsonable_dict(p1_evidence.get("baseline_evidence")),
        baseline_ablation_evidence=_jsonable_dict(p1_evidence.get("baseline_ablation_evidence")),
        feature_probe_evidence=_jsonable_list(p1_evidence.get("feature_probe_evidence")),
        feature_diagnostics=_jsonable_dict(feature_diagnostics),
        target_diagnostics=_jsonable_dict(target_diagnostics),
        interaction_diagnostics=_jsonable_dict(interaction_diagnostics),
        source_claim_validation=_jsonable_dict(source_claim_validation),
        eda_strategy_hints=_jsonable_dict(eda_strategy_hints),
        notebook_static_analysis=_jsonable_dict(p1_evidence.get("notebook_static_analysis")),
        hypothesis_results=[],
        recommended_next_actions=[],
        warnings=list(warnings),
        limitations=list(dict.fromkeys(limitations)),
        artifacts={
            "run_dir": str(run_dir),
            "partial": True,
            "module_statuses": dict(module_statuses),
            "module_status_details": module_status_details,
        },
    )
    writer.write_json("eda_evidence_pack_partial.json", evidence_pack)
    writer.write_json("eda_evidence_pack.json", evidence_pack)


def _jsonable_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    jsonable = _jsonable_value(value)
    return jsonable if isinstance(jsonable, dict) else {}


def _jsonable_list(value: Any) -> list[Any]:
    if value is None:
        return []
    jsonable = _jsonable_value(value)
    if isinstance(jsonable, list):
        return jsonable
    return []


def _jsonable_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


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
    module_status_details: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    p1_evidence: dict[str, Any] = {
        key: _copy_placeholder(value) for key, value in P1_PLACEHOLDERS.items()
    }

    for module_name in P1_MODULES:
        artifact_name = _artifact_name_for_module(module_name)
        if not _should_run_p1_module(module_name, config, task_plan):
            _record_module_status(
                module_name,
                "skipped",
                module_statuses=module_statuses,
                module_status_details=module_status_details,
                legacy_status="skipped",
            )
            writer.write_json(f"{artifact_name}.json", p1_evidence[artifact_name])
            continue
        if module_name == "baseline_runner" and not (config.enable_baseline or config.enable_baseline_ablations):
            skipped = _skipped_p1_payload(
                module_name,
                "Baseline runner requires enable_baseline=true.",
            )
            _record_module_status(
                module_name,
                "skipped",
                module_statuses=module_statuses,
                module_status_details=module_status_details,
                legacy_status="skipped",
            )
            p1_evidence[artifact_name] = skipped
            writer.write_json(f"{artifact_name}.json", skipped)
            continue

        started_perf = time.perf_counter()
        started_at = _now_iso()
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
                    validation_evidence=validation_evidence.model_dump(mode="json"),
                )
            else:
                result = analyze_notebooks_static(
                    [],
                    output_dir=writer.artifact_path("notebooks"),
                )
            legacy_status = _status_from_p1_result(result)
            detail_status = _detail_status_from_legacy(legacy_status)
            _record_module_status(
                module_name,
                detail_status,
                module_statuses=module_statuses,
                module_status_details=module_status_details,
                legacy_status=legacy_status,
                started_at=started_at,
                started_perf=started_perf,
            )
            p1_evidence[artifact_name] = result
            writer.write_json(f"{artifact_name}.json", result)
        except Exception as exc:
            failed = _failed_p1_payload(module_name, exc)
            _record_module_status(
                module_name,
                "failed",
                module_statuses=module_statuses,
                module_status_details=module_status_details,
                legacy_status="failed",
                started_at=started_at,
                started_perf=started_perf,
                error_message=_sanitize_error(exc),
            )
            p1_evidence[artifact_name] = (
                [failed] if artifact_name == "feature_probe_evidence" else failed
            )
            writer.write_json(f"{artifact_name}.json", p1_evidence[artifact_name])
            warnings.append(f"{module_name} failed: {_sanitize_error(exc)}")

    return p1_evidence


def _run_baseline_ablation_module(
    *,
    config: EdaRunConfig,
    writer: ArtifactWriter,
    inferred_schema: Any,
    validation_evidence: Any,
    metric_evidence: Any,
    leakage_evidence: list[Any],
    feature_diagnostics: dict[str, Any],
    reader: DatasetReader,
    baseline_evidence: dict[str, Any] | None,
    module_statuses: dict[str, str],
    module_status_details: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    module_name = "baseline_ablation_runner"
    artifact_name = "baseline_ablation_evidence"
    if not config.enable_baseline_ablations:
        skipped = _skipped_p1_payload(
            module_name,
            "Baseline ablations require enable_baseline_ablations=true.",
        )
        _record_module_status(
            module_name,
            "skipped",
            module_statuses=module_statuses,
            module_status_details=module_status_details,
            legacy_status="skipped",
        )
        writer.write_json(f"{artifact_name}.json", skipped)
        return skipped

    started_perf = time.perf_counter()
    started_at = _now_iso()
    try:
        result = run_baseline_ablations(
            inferred_schema,
            validation_evidence,
            metric_evidence,
            leakage_evidence,
            reader,
            output_dir=writer.artifact_path("baseline_ablations"),
            baseline_evidence=baseline_evidence or {},
            feature_diagnostics=feature_diagnostics,
            max_rows=config.max_ablation_rows,
            max_ablations=config.max_ablations,
            random_seed=config.random_seed,
            n_folds=config.ablation_n_folds,
            max_runtime_sec=config.max_ablation_runtime_sec,
        )
        legacy_status = _status_from_p1_result(result)
        _record_module_status(
            module_name,
            _detail_status_from_legacy(legacy_status),
            module_statuses=module_statuses,
            module_status_details=module_status_details,
            legacy_status=legacy_status,
            started_at=started_at,
            started_perf=started_perf,
        )
        writer.write_json(f"{artifact_name}.json", result)
        return result
    except Exception as exc:
        failed = _failed_p1_payload(module_name, exc)
        _record_module_status(
            module_name,
            "failed",
            module_statuses=module_statuses,
            module_status_details=module_status_details,
            legacy_status="failed",
            started_at=started_at,
            started_perf=started_perf,
            error_message=_sanitize_error(exc),
        )
        writer.write_json(f"{artifact_name}.json", failed)
        warnings.append(f"{module_name} failed: {_sanitize_error(exc)}")
        return failed


def _run_interaction_diagnostics_module(
    *,
    config: EdaRunConfig,
    task_plan: EdaTaskPlan,
    writer: ArtifactWriter,
    inferred_schema: Any,
    table_profiles: list[Any],
    metric_evidence: Any,
    reader: DatasetReader,
    feature_diagnostics: dict[str, Any],
    target_diagnostics: dict[str, Any],
    drift_evidence: dict[str, Any] | None,
    baseline_ablation_evidence: dict[str, Any] | None,
    leakage_evidence: list[Any],
    module_statuses: dict[str, str],
    module_status_details: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    module_name = "interaction_diagnostics"
    if not _should_run_interaction_diagnostics(config, task_plan):
        skipped = _skipped_p1_payload(module_name, "Interaction diagnostics require enable_interaction_diagnostics=true or P1 modules.")
        _record_module_status(module_name, "skipped", module_statuses=module_statuses, module_status_details=module_status_details, legacy_status="skipped")
        writer.write_json("interaction_diagnostics.json", skipped)
        return skipped
    started_perf, started_at = time.perf_counter(), _now_iso()
    try:
        result = diagnose_interactions(
            inferred_schema, table_profiles, metric_evidence, reader,
            feature_diagnostics=feature_diagnostics,
            target_diagnostics=target_diagnostics,
            drift_evidence=drift_evidence,
            baseline_ablation_evidence=baseline_ablation_evidence,
            leakage_evidence=[item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in leakage_evidence],
            max_rows=config.max_interaction_rows,
            max_numeric_columns=config.max_interaction_numeric_columns,
            max_categorical_columns=config.max_interaction_categorical_columns,
            max_pair_candidates=config.max_interaction_pair_candidates,
            max_reported_interactions_per_type=config.max_reported_interactions_per_type,
            min_group_rows=config.interaction_min_group_rows,
            random_state=config.random_seed,
        )
        legacy_status = _status_from_p1_result(result)
        _record_module_status(module_name, _detail_status_from_legacy(legacy_status), module_statuses=module_statuses, module_status_details=module_status_details, legacy_status=legacy_status, started_at=started_at, started_perf=started_perf)
        writer.write_json("interaction_diagnostics.json", result)
        return result
    except Exception as exc:
        failed = _failed_p1_payload(module_name, exc)
        _record_module_status(module_name, "failed", module_statuses=module_statuses, module_status_details=module_status_details, legacy_status="failed", started_at=started_at, started_perf=started_perf, error_message=_sanitize_error(exc))
        writer.write_json("interaction_diagnostics.json", failed)
        warnings.append(f"{module_name} failed: {_sanitize_error(exc)}")
        return failed


def _should_run_interaction_diagnostics(config: EdaRunConfig, task_plan: EdaTaskPlan) -> bool:
    module_name = "interaction_diagnostics"
    if module_name in _normalised_modules(config.skip_modules):
        return False
    explicit = _normalised_modules(config.modules or [])
    if module_name in explicit:
        return True
    if any(task.module == module_name for task in task_plan.eda_tasks):
        return True
    return bool(config.enable_interaction_diagnostics or config.enable_p1_modules)


def _run_source_claim_validation_module(
    *, config: EdaRunConfig, task_plan: EdaTaskPlan, writer: ArtifactWriter, hypotheses: ResearchHypotheses,
    evidence_pack: dict[str, Any], module_statuses: dict[str, str], module_status_details: dict[str, dict[str, Any]], warnings: list[str],
) -> dict[str, Any]:
    module_name = "source_claim_validation"
    requested = module_name in _normalised_modules(config.modules or []) or any(task.module == module_name for task in task_plan.eda_tasks) or config.enable_source_claim_validation or config.enable_p1_modules
    if module_name in _normalised_modules(config.skip_modules) or not requested:
        skipped = _skipped_p1_payload(module_name, "Source claim validation is optional.")
        _record_module_status(module_name, "skipped", module_statuses=module_statuses, module_status_details=module_status_details, legacy_status="skipped")
        writer.write_json("source_claim_validation.json", skipped)
        return skipped
    started_perf, started_at = time.perf_counter(), _now_iso()
    try:
        notebook = evidence_pack.get("notebook_static_analysis")
        claims = collect_source_claims(hypotheses, notebook if isinstance(notebook, dict) else {})
        result = validate_source_claims(claims, evidence_pack)
        legacy_status = _status_from_p1_result(result)
        _record_module_status(module_name, _detail_status_from_legacy(legacy_status), module_statuses=module_statuses, module_status_details=module_status_details, legacy_status=legacy_status, started_at=started_at, started_perf=started_perf)
        writer.write_json("source_claim_validation.json", result)
        return result
    except Exception as exc:
        failed = _failed_p1_payload(module_name, exc)
        _record_module_status(module_name, "failed", module_statuses=module_statuses, module_status_details=module_status_details, legacy_status="failed", started_at=started_at, started_perf=started_perf, error_message=_sanitize_error(exc))
        writer.write_json("source_claim_validation.json", failed); warnings.append(f"{module_name} failed: {_sanitize_error(exc)}")
        return failed


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
    if module_name == "baseline_runner" and (config.enable_baseline or config.enable_baseline_ablations):
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
        "error_message": _sanitize_error(exc),
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


__all__ = ["run_eda"]
