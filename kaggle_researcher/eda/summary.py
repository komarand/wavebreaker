from __future__ import annotations

from collections import Counter
from typing import Any

from kaggle_researcher.eda.schemas import EdaEvidencePack


def build_eda_summary(pack: EdaEvidencePack) -> str:
    """Build a concise markdown summary strictly from an EDA evidence pack."""

    sections = [
        "# EDA Summary",
        _dataset_section(pack),
        _schema_section(pack),
        _metric_section(pack),
        _validation_section(pack),
        _leakage_section(pack),
        _relationships_section(pack),
        _drift_section(pack),
        _baseline_section(pack),
        _feature_probes_section(pack),
        _feature_diagnostics_section(pack),
        _strategy_hints_section(pack),
        _hypothesis_results_section(pack),
        _recommended_next_actions_section(pack),
        _warnings_section(pack),
        _limitations_section(pack),
    ]
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def _dataset_section(pack: EdaEvidencePack) -> str:
    dataset = _as_dict(pack.dataset)
    lines = ["## Dataset"]
    lines.append(f"- Competition: `{pack.competition_id}`")
    lines.append(f"- Run: `{pack.run_id}`")
    if dataset.get("dataset_path"):
        lines.append(f"- Dataset path: `{dataset['dataset_path']}`")
    if dataset.get("source"):
        lines.append(f"- Source: `{dataset['source']}`")
    if dataset.get("partial"):
        lines.append("- Status: partial run")
    return _join(lines)


def _schema_section(pack: EdaEvidencePack) -> str:
    schema = _as_dict(pack.inferred_schema)
    inventory = _as_dict(pack.file_inventory)
    profiles = [_as_dict(item) for item in pack.table_profiles]
    lines = ["## Schema"]
    if not schema and not inventory and not profiles:
        lines.append("- Not testable: schema evidence is not present.")
        return _join(lines)

    files = inventory.get("files", [])
    tables = schema.get("tables", [])
    lines.append(f"- Files inventoried: `{len(files)}`")
    lines.append(f"- Tables inferred: `{len(tables)}`")
    reconciled = _as_dict(inventory.get("reconciled_table_roles"))
    if reconciled:
        lines.append(f"- Final table roles: `{_csv([f'{path}: {role}' for path, role in reconciled.items()])}`")
    size_rows = [
        f"{item.get('path') or item.get('table_name')}: {item.get('n_rows', 'unknown')} rows x {item.get('n_cols', 'unknown')} cols"
        for item in profiles[:5]
    ]
    if size_rows:
        lines.append(f"- Data size: `{_csv(size_rows)}`")
    if schema.get("target_column"):
        lines.append(f"- Target column: `{schema['target_column']}`")
    if schema.get("primary_id_column"):
        lines.append(f"- Primary id column: `{schema['primary_id_column']}`")
    if schema.get("sample_submission_table"):
        lines.append(f"- Sample submission table: `{schema['sample_submission_table']}`")
    global_roles = _as_dict(schema.get("global_roles"))
    role_warnings = list(global_roles.get("role_inference_warnings") or [])
    role_candidates = _role_candidate_summary(global_roles)
    if role_candidates:
        lines.append(f"- Role candidates: {role_candidates}")
    if role_warnings:
        lines.append(f"- Role inference warnings: {_csv(role_warnings)}")
    if schema.get("candidate_time_columns") or schema.get("candidate_date_columns"):
        candidates = list(schema.get("candidate_time_columns") or [])
        candidates.extend(schema.get("candidate_date_columns") or [])
        lines.append(f"- Time/date candidates: `{_csv(candidates)}`")
    sampled = [item.get("table_name") for item in profiles if item.get("sampled")]
    if sampled:
        lines.append(f"- Sampled table profiles: `{_csv(sampled)}`")
    return _join(lines)


def _metric_section(pack: EdaEvidencePack) -> str:
    metric = _as_dict(pack.metric_evidence)
    lines = ["## Metric"]
    if not metric:
        lines.append("- Not testable: metric evidence is not present.")
        return _join(lines)

    lines.append(f"- Metric: `{metric.get('metric_name', 'unknown')}`")
    if metric.get("task_type"):
        lines.append(f"- Task type: `{metric['task_type']}`")
    if metric.get("metric_family"):
        lines.append(f"- Metric family: `{metric['metric_family']}`")
    if metric.get("prediction_output_type"):
        lines.append(f"- Prediction output: `{metric['prediction_output_type']}`")
    if metric.get("local_metric_available") is not None:
        lines.append(f"- Local metric available: `{bool(metric.get('local_metric_available'))}`")
    if metric.get("needs_custom_implementation"):
        lines.append("- Custom implementation is needed.")
    return _join(lines)


def _validation_section(pack: EdaEvidencePack) -> str:
    validation = _as_dict(pack.validation_evidence)
    lines = ["## Validation"]
    if not validation:
        lines.append("- Not testable: validation evidence is not present.")
        return _join(lines)

    primary = _as_dict(validation.get("primary_validation"))
    method = primary.get("method", "unknown")
    lines.append(f"- Primary validation: `{method}`{_refs(validation.get('evidence_refs'))}")
    if primary.get("reason"):
        lines.append(f"- Primary reason: {primary['reason']}")
    if validation.get("reasoning_summary"):
        lines.append(f"- Reasoning: {validation['reasoning_summary']}")

    diagnostics = [_as_dict(item) for item in validation.get("diagnostic_validations", [])]
    if diagnostics:
        diagnostic_methods = [item.get("method", "unknown") for item in diagnostics]
        lines.append(f"- Diagnostic validations: `{_csv(diagnostic_methods)}`")
        if _has_temporal_signal(diagnostic_methods) and not _is_temporal_method(str(method)):
            lines.append("- Temporal validation is diagnostic, not the selected primary validation.")
    rejected = [_as_dict(item) for item in validation.get("rejected_validations", [])]
    if rejected:
        rejected_methods = [item.get("method", "unknown") for item in rejected]
        lines.append(f"- Rejected validations: `{_csv(rejected_methods)}`")
    return _join(lines)


def _leakage_section(pack: EdaEvidencePack) -> str:
    checks = [_as_dict(item) for item in pack.leakage_evidence]
    lines = ["## Leakage"]
    if not checks:
        lines.append("- Not testable: leakage evidence is not present.")
        return _join(lines)

    counts = Counter(str(item.get("status", "unknown")) for item in checks)
    lines.append(f"- Checks: `{len(checks)}` ({_format_counts(counts)})")
    for item in checks:
        status = item.get("status", "unknown")
        check_id = item.get("check_id", "unknown_check")
        finding = item.get("finding") or "No finding text recorded."
        lines.append(f"- `{check_id}`: `{status}` - {finding}")
    return _join(lines)


def _relationships_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.relationship_evidence)
    lines = ["## Relationships"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'relationship_inferer')}")
        return _join(lines)

    status = evidence.get("status")
    if status in {"skipped", "not_testable", "failed"}:
        lines.append(f"- `{status}`: {evidence.get('reason') or evidence.get('error_message') or 'No relationship evidence recorded.'}")
        return _join(lines)
    relationships = evidence.get("relationships", [])
    rejected = evidence.get("rejected_relationships", [])
    lines.append(f"- Relationships checked: `{len(relationships)}`")
    if rejected:
        reasons = [
            f"{item.get('left_table')}->{item.get('right_table')}: {item.get('reason')}"
            for item in rejected[:5]
        ]
        lines.append(f"- Rejected relationships: `{_csv(reasons)}`")
    if evidence.get("candidate_keys"):
        lines.append(f"- Candidate keys: `{_csv(evidence['candidate_keys'])}`")
    return _join(lines)


def _drift_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.drift_evidence)
    lines = ["## Drift"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'drift_analyzer')}")
        return _join(lines)

    status = evidence.get("status", "completed")
    if status in {"skipped", "not_testable", "failed"}:
        lines.append(f"- `{status}`: {evidence.get('reason') or evidence.get('error_message') or 'No drift evidence recorded.'}")
        return _join(lines)
    lines.append(f"- Status: `{status}`")
    severity = evidence.get("feature_drift_severity") or evidence.get("severity")
    if severity:
        lines.append(f"- Safe feature drift severity: `{severity}`")
    adversarial = _as_dict(evidence.get("adversarial_validation"))
    if adversarial.get("auc") is not None:
        lines.append(f"- Adversarial AUC: `{adversarial['auc']}`")
    if evidence.get("excluded_columns"):
        excluded = [
            f"{item.get('column')} ({item.get('reason')})"
            for item in evidence.get("excluded_columns", [])[:8]
        ]
        lines.append(f"- Excluded drift artifacts: `{_csv(excluded)}`")
    if evidence.get("drift_interpretation"):
        lines.append(f"- Interpretation: {evidence['drift_interpretation']}")
    return _join(lines)


def _feature_diagnostics_section(pack: EdaEvidencePack) -> str:
    diagnostics = _as_dict(pack.feature_diagnostics)
    lines = ["## Feature diagnostics"]
    if not diagnostics:
        lines.append("- Not testable: feature diagnostics are not present.")
        return _join(lines)

    numeric = _as_dict(diagnostics.get("numeric_feature_diagnostics"))
    categorical = _as_dict(diagnostics.get("categorical_feature_diagnostics"))
    missingness = _as_dict(diagnostics.get("missingness_diagnostics"))
    text = _as_dict(diagnostics.get("text_feature_diagnostics"))
    date_time = _as_dict(diagnostics.get("date_time_diagnostics"))
    if numeric.get("top_predictive_candidates"):
        lines.append(f"- Top numeric candidates: `{_csv(_column_names(numeric['top_predictive_candidates']))}`")
    if categorical.get("low_cardinality_candidates"):
        lines.append(f"- Low-cardinality categoricals: `{_csv(_column_names(categorical['low_cardinality_candidates']))}`")
    if categorical.get("high_cardinality_candidates"):
        lines.append(f"- High-cardinality risks: `{_csv(_column_names(categorical['high_cardinality_candidates']))}`")
    if missingness.get("recommended_indicators"):
        lines.append(f"- Missingness indicators: `{_csv(_column_names(missingness['recommended_indicators']))}`")
    shifted = list(numeric.get("shifted_features") or [])
    if shifted:
        lines.append(f"- Shifted numeric features: `{_csv(_column_names(shifted))}`")
    if text.get("columns"):
        lines.append(f"- Text/code-like columns: `{_csv(_column_names(text['columns']))}`")
    if date_time.get("columns"):
        lines.append(f"- Date/time columns: `{_csv(_column_names(date_time['columns']))}`")
        lines.append(f"- Temporal signal: `{date_time.get('temporal_validation_signal', 'unknown')}`")
    if len(lines) == 1:
        lines.append("- No high-signal generic feature diagnostics were found.")
    return _join(lines)


def _strategy_hints_section(pack: EdaEvidencePack) -> str:
    hints = _as_dict(pack.eda_strategy_hints)
    lines = ["## Strategy hints"]
    if not hints:
        lines.append("- None recorded.")
        return _join(lines)
    for category, items in hints.items():
        category_items = [_as_dict(item) for item in items]
        if not category_items:
            continue
        actions = [
            f"{item.get('priority', 'P?')}: {item.get('action')}"
            for item in category_items[:3]
        ]
        lines.append(f"- {category}: `{_csv(actions)}`")
    if len(lines) == 1:
        lines.append("- None recorded.")
    return _join(lines)


def _baseline_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.baseline_evidence)
    lines = ["## Baseline"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'baseline_runner')}")
        return _join(lines)

    status = evidence.get("status", "completed")
    lines.append(f"- Status: `{status}`")
    if status in {"skipped", "not_testable", "failed"}:
        reason = evidence.get("reason") or evidence.get("error_message")
        if reason:
            lines.append(f"- Reason: {reason}")
        return _join(lines)
    if evidence.get("metric_name"):
        lines.append(f"- Metric: `{evidence['metric_name']}`")
    if evidence.get("mean_score") is not None:
        lines.append(f"- Mean validation score: `{evidence['mean_score']}`")
    return _join(lines)


def _feature_probes_section(pack: EdaEvidencePack) -> str:
    probes = [_as_dict(item) for item in pack.feature_probe_evidence]
    lines = ["## Feature probes"]
    if not probes:
        lines.append(f"- {_module_absence(pack, 'feature_probe')}")
        return _join(lines)

    counts = Counter(str(item.get("status", "unknown")) for item in probes)
    lines.append(f"- Families checked: `{len(probes)}` ({_format_counts(counts)})")
    for item in probes:
        family = item.get("feature_family") or item.get("family") or item.get("module") or "unknown_family"
        status = item.get("status", "unknown")
        finding = item.get("finding") or item.get("summary") or item.get("reason") or ""
        suffix = f" - {finding}" if finding else ""
        lines.append(f"- `{family}`: `{status}`{suffix}")
    return _join(lines)


def _hypothesis_results_section(pack: EdaEvidencePack) -> str:
    results = list(pack.hypothesis_results)
    lines = ["## Hypothesis results"]
    if not results:
        lines.append("- Not testable: no hypothesis results are present.")
        return _join(lines)

    counts = Counter(result.status for result in results)
    lines.append(f"- Results: `{len(results)}` ({_format_counts(counts)})")
    for result in results:
        lines.append(
            f"- `{result.hypothesis_id}`: `{result.status}` - "
            f"{result.finding}{_refs(result.evidence_refs)}"
        )
    return _join(lines)


def _recommended_next_actions_section(pack: EdaEvidencePack) -> str:
    actions = list(pack.recommended_next_actions)
    lines = ["## Recommended next actions"]
    if not actions:
        lines.append("- None recorded.")
        return _join(lines)

    for action in actions:
        lines.append(
            f"- `{action.priority}`: {action.action} "
            f"Why: {action.why}{_refs(action.evidence_refs)}"
        )
    return _join(lines)


def _warnings_section(pack: EdaEvidencePack) -> str:
    return _text_list_section("## Warnings", pack.warnings, empty="None recorded.")


def _limitations_section(pack: EdaEvidencePack) -> str:
    return _text_list_section("## Limitations", pack.limitations, empty="None recorded.")


def _module_absence(pack: EdaEvidencePack, module_name: str) -> str:
    artifacts = _as_dict(pack.artifacts)
    detail = _as_dict(_as_dict(artifacts.get("module_status_details")).get(module_name))
    legacy = _as_dict(artifacts.get("module_statuses")).get(module_name)
    status = detail.get("status") or legacy
    if status in {"success", "completed"}:
        return "Not testable: module completed but no evidence was recorded."
    if status in {"failed", "skipped", "not_testable"}:
        message = detail.get("error_message")
        suffix = f": {message}" if message else "."
        return f"`{status}`{suffix}"
    return "Not testable: evidence is not present in this pack."


def _text_list_section(title: str, items: list[str], *, empty: str) -> str:
    lines = [title]
    if not items:
        lines.append(f"- {empty}")
    else:
        lines.extend(f"- {item}" for item in items)
    return _join(lines)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _refs(refs: Any) -> str:
    if not refs:
        return ""
    return f" [refs: {_csv(refs)}]"


def _csv(values: Any) -> str:
    if values is None:
        return "none"
    if isinstance(values, str):
        return values
    return ", ".join(str(value) for value in values) or "none"


def _column_names(rows: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("column")) for item in rows[:8] if item.get("column")]


def _format_counts(counts: Counter) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "none"


def _role_candidate_summary(global_roles: dict[str, Any]) -> str:
    parts: list[str] = []
    for label, key in (
        ("target", "target_column_candidates"),
        ("id", "primary_id_column_candidates"),
        ("submission", "sample_submission_table_candidates"),
    ):
        candidates = [_as_dict(item) for item in global_roles.get(key, [])]
        if len(candidates) > 1:
            names = [str(item.get("name")) for item in candidates[:3] if item.get("name")]
            if names:
                parts.append(f"{label}: `{_csv(names)}`")
    return "; ".join(parts)


def _has_temporal_signal(methods: list[str]) -> bool:
    return any(_is_temporal_method(method) for method in methods)


def _is_temporal_method(method: str) -> bool:
    lowered = method.lower()
    return "temporal" in lowered or "time" in lowered or "oot" in lowered


def _join(lines: list[str]) -> str:
    return "\n".join(lines)


__all__ = ["build_eda_summary"]
