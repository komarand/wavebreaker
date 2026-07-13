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
        _target_diagnostics_section(pack),
        _leakage_section(pack),
        _relationships_section(pack),
        _drift_section(pack),
        _baseline_section(pack),
        _baseline_ablations_section(pack),
        _interaction_diagnostics_section(pack),
        _visual_diagnostics_section(pack),
        _slice_diagnostics_section(pack),
        _feature_probes_section(pack),
        _feature_diagnostics_section(pack),
        _risk_register_section(pack),
        _strategy_hints_section(pack),
        _hypothesis_results_section(pack),
        _safety_constraints_section(pack),
        _validation_requirements_section(pack),
        _testable_hypotheses_section(pack),
        _warnings_section(pack),
        _limitations_section(pack),
        "These findings are diagnostic evidence for the downstream reasoning and strategy layer; they are not the final competition strategy.",
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


def _target_diagnostics_section(pack: EdaEvidencePack) -> str:
    diagnostics = _as_dict(pack.target_diagnostics)
    lines = ["## Target diagnostics"]
    if not diagnostics:
        lines.append("- Not testable: target diagnostics are not present.")
        return _join(lines)

    status = diagnostics.get("status", "unknown")
    lines.append(f"- Status: `{status}`")
    if status != "completed":
        reason = diagnostics.get("reason")
        if reason:
            lines.append(f"- Reason: {reason}")
        return _join(lines)

    distribution = _as_dict(diagnostics.get("distribution"))
    target_type = distribution.get("target_type", "unknown")
    lines.append(f"- Target type: `{target_type}`")
    if target_type in {"binary", "multiclass", "ranking"}:
        classes = [
            f"{item.get('class')}: {round(float(item.get('pct') or 0.0) * 100, 2)}%"
            for item in list(distribution.get("classes") or [])[:8]
        ]
        if classes:
            lines.append(f"- Class balance: `{_csv(classes)}`")
        imbalance = _as_dict(diagnostics.get("imbalance"))
        if imbalance:
            lines.append(f"- Imbalance: `{imbalance.get('severity', 'unknown')}`")
    elif target_type == "regression":
        quantiles = _as_dict(distribution.get("quantiles"))
        lines.append(
            "- Regression target: "
            f"`mean={distribution.get('mean')}, q50={quantiles.get('q50')}, "
            f"max={distribution.get('max')}`"
        )
        lines.append(f"- Heavy tail: `{bool(distribution.get('heavy_tail'))}`")

    metric_implications = [
        _as_dict(item).get("implication")
        for item in diagnostics.get("metric_implications", [])
        if _as_dict(item).get("implication")
    ]
    validation_implications = [
        _as_dict(item).get("implication")
        for item in diagnostics.get("validation_implications", [])
        if _as_dict(item).get("implication")
    ]
    if metric_implications:
        lines.append(f"- Metric implications: `{_csv(metric_implications[:5])}`")
    if validation_implications:
        lines.append(f"- Validation implications: `{_csv(validation_implications[:5])}`")

    by_feature = _as_dict(diagnostics.get("target_by_feature"))
    numeric = [_as_dict(item).get("column") for item in by_feature.get("numeric_binned", [])[:5]]
    categorical = [_as_dict(item).get("column") for item in by_feature.get("categorical", [])[:5]]
    missingness = [_as_dict(item).get("column") for item in diagnostics.get("target_by_missingness", [])[:5]]
    if any(numeric):
        lines.append(f"- Top target-associated numeric features: `{_csv([item for item in numeric if item])}`")
    if any(categorical):
        lines.append(f"- Top target-associated categorical features: `{_csv([item for item in categorical if item])}`")
    if any(missingness):
        lines.append(f"- Missingness associated with target: `{_csv([item for item in missingness if item])}`")
    suspicious = [_as_dict(item) for item in diagnostics.get("suspicious_patterns", [])]
    if suspicious:
        severities = Counter(str(item.get("severity", "unknown")) for item in suspicious)
        lines.append(f"- Suspicious target patterns: `{_format_counts(severities)}`")
    else:
        lines.append("- Suspicious target patterns: `none`")
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
    reliable_numeric = [
        item
        for item in [_as_dict(row) for row in numeric.get("top_predictive_candidates", [])]
        if item.get("target_association_reliability") == "reliable"
    ]
    cautious_numeric = [
        item
        for item in [_as_dict(row) for row in numeric.get("columns", [])]
        if str(item.get("target_association_reliability", "")).startswith("caution")
        or str(item.get("outlier_reliability", "")).startswith("caution")
    ]
    if reliable_numeric:
        lines.append(f"- Top reliable numeric candidates: `{_csv(_column_names(reliable_numeric))}`")
    elif numeric.get("top_predictive_candidates"):
        lines.append(f"- Top numeric candidates: `{_csv(_column_names(numeric['top_predictive_candidates']))}`")
    if cautious_numeric:
        lines.append(f"- Numeric features needing cautious interpretation: `{_csv(_column_names(cautious_numeric))}`")
    if categorical.get("low_cardinality_candidates"):
        lines.append(f"- Low-cardinality categoricals: `{_csv(_column_names(categorical['low_cardinality_candidates']))}`")
    if categorical.get("high_cardinality_candidates"):
        lines.append(f"- High-cardinality risks: `{_csv(_column_names(categorical['high_cardinality_candidates']))}`")
    if categorical.get("target_association_cautions"):
        lines.append(f"- Target-association caution: `{_csv(_column_names(categorical['target_association_cautions']))}`")
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
    implications = [_as_dict(item) for item in pack.eda_implications]
    lines = ["## EDA implications (Strategy hints compatibility)"]
    if not implications and pack.eda_strategy_hints:
        for category, items in pack.eda_strategy_hints.items():
            for item in items:
                payload = _as_dict(item)
                lines.append(f"- {category}: `{payload.get('priority', 'P?')}: {payload.get('action')}`{_refs(payload.get('evidence_refs', []))}")
        return _join(lines)
    if not implications:
        lines.append("- None recorded.")
        return _join(lines)
    for item in implications:
        lines.append(f"- `{item.get('priority_signal', 'informational')}`: {item.get('implication')} Finding: {item.get('finding')}{_refs(item.get('evidence_refs', []))}")
    return _join(lines)


def _risk_register_section(pack: EdaEvidencePack) -> str:
    risk_source = pack.eda_risks or pack.eda_risk_register
    risks = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else _as_dict(item)
        for item in risk_source
    ]
    lines = ["## Risk register (EDA-local risks)"]
    if not risks:
        lines.append("- Risks: `0`")
        return _join(lines)

    severity_counts = Counter(str(item.get("severity", "unknown")) for item in risks)
    status_counts = Counter(str(item.get("status", "unknown")) for item in risks)
    lines.append(f"- Risks: `{len(risks)}` ({_format_counts(severity_counts)})")
    high = [item for item in risks if item.get("severity") in {"critical", "high"}]
    medium = [item for item in risks if item.get("severity") == "medium"]
    if high:
        lines.append(f"- High risks: `{_risk_titles(high[:5])}`")
    if medium:
        lines.append(f"- Medium risks: `{_risk_titles(medium[:5])}`")
    lines.append(f"- Mitigated by policy: `{status_counts.get('mitigated_by_policy', 0)}`")
    lines.append(f"- Not testable: `{status_counts.get('not_testable', 0)}`")
    return _join(lines)


def _risk_titles(risks: list[dict[str, Any]]) -> str:
    return _csv(
        f"{item.get('risk_type')}: {item.get('title')} - {item.get('status')}"
        for item in risks
    )


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
    if evidence.get("model_type"):
        lines.append(f"- Model: `{evidence['model_type']}`")
    if evidence.get("metric_name"):
        metric_value = evidence.get("metric_value", evidence.get("mean_score"))
        if metric_value is not None:
            lines.append(f"- Metric: `{evidence['metric_name']} = {metric_value}`")
        else:
            lines.append(f"- Metric: `{evidence['metric_name']}`")
    validation_policy = _as_dict(evidence.get("validation_policy"))
    if validation_policy.get("method"):
        fold_suffix = (
            f", {validation_policy['n_folds']} folds"
            if validation_policy.get("n_folds") is not None
            else ""
        )
        lines.append(f"- Validation: `{validation_policy['method']}`{fold_suffix}")
    preprocessing = _as_dict(evidence.get("preprocessing_policy"))
    if preprocessing:
        fit_scope = preprocessing.get("fit_scope")
        safety = _as_dict(preprocessing.get("safety_checks"))
        label = "fold-safe" if safety.get("fits_preprocessing_inside_folds") else fit_scope
        if label:
            lines.append(f"- Preprocessing: `{label}`")
        numeric_summary = _numeric_policy_summary(preprocessing)
        if numeric_summary:
            lines.append(f"- Numeric preprocessing: `{numeric_summary}`")
        categorical_summary = _categorical_policy_summary(preprocessing)
        if categorical_summary:
            lines.append(f"- Categorical preprocessing: `{categorical_summary}`")
        high_cardinality_summary = _high_cardinality_policy_summary(preprocessing)
        if high_cardinality_summary:
            lines.append(f"- High-cardinality policy: `{high_cardinality_summary}`")
        excluded_summary = _excluded_columns_summary(
            preprocessing.get("excluded_roles") or evidence.get("excluded_column_details")
        )
        if excluded_summary:
            lines.append(f"- Excluded columns: `{excluded_summary}`")
        limitations = list(preprocessing.get("limitations") or evidence.get("limitations") or [])
        if limitations:
            lines.append(f"- Limitations: `{_csv(limitations[:3])}`")
    return _join(lines)


def _slice_diagnostics_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.slice_diagnostics)
    lines = ["## Slice diagnostics"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'slice_diagnostics')}")
        return _join(lines)
    lines.append(f"- Status: `{evidence.get('status', 'unknown')}`")
    if evidence.get("status") != "completed":
        lines.append(f"- Reason: `{evidence.get('reason', 'not available')}`")
        return _join(lines)
    summary = _as_dict(evidence.get("summary"))
    lines.append(f"- Fold-safe OOF slices: `{summary.get('slice_count', 0)}` across `{summary.get('total_rows', 0)}` rows")
    return _join(lines)


def _visual_diagnostics_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.visual_diagnostics)
    lines = ["## Visual diagnostics"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'visual_diagnostics')}")
        return _join(lines)
    lines.append(f"- Status: `{evidence.get('status', 'unknown')}`")
    if evidence.get("status") != "completed":
        if evidence.get("reason") or evidence.get("error_message"):
            lines.append(f"- Reason: `{evidence.get('reason') or evidence.get('error_message')}`")
        return _join(lines)
    lines.append(f"- Generated plots: `{len(evidence.get('generated_plots', []))}`")
    if _as_dict(evidence.get("summary_dashboard")).get("status") == "generated":
        lines.append(f"- Dashboard: `{_as_dict(evidence['summary_dashboard']).get('artifact_path')}`")
    main = [_as_dict(item) for item in evidence.get("generated_plots", []) if _as_dict(item).get("plot_type") != "summary_dashboard"][:4]
    if main:
        lines.append(f"- Main artifacts: `{_csv([item.get('artifact_path') for item in main])}`")
    if evidence.get("manifest_path"):
        lines.append(f"- Plot manifest: `{evidence['manifest_path']}`")
    return _join(lines)


def _source_claim_validation_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.source_claim_validation)
    lines = ["## Source claim validation"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'source_claim_validation')}")
        return _join(lines)
    lines.append(f"- Status: `{evidence.get('status', 'unknown')}`")
    if evidence.get("status") != "completed":
        if evidence.get("reason"):
            lines.append(f"- Reason: `{evidence['reason']}`")
        return _join(lines)
    claims = [_as_dict(item) for item in evidence.get("validated_claims", [])]
    counts = Counter(str(item.get("validation_status")) for item in claims)
    lines.append(f"- Claims analyzed: `{len(claims)}`; confirmed `{counts['confirmed']}`, partially supported `{counts['partially_supported']}`, contradicted `{counts['contradicted']}`, unsafe `{counts['unsafe']}`, analogous only `{counts['analogous_only']}`, not testable `{counts['not_testable']}`")
    accepted = [item for item in claims if item.get("validation_status") in {"confirmed", "partially_supported"}][:5]
    rejected = [item for item in claims if item.get("validation_status") in {"contradicted", "unsafe"}][:5]
    if accepted:
        lines.append(f"- Key validated claims: `{_csv([item.get('claim_text') for item in accepted])}`")
    if rejected:
        lines.append(f"- Key rejected or unsafe claims: `{_csv([item.get('claim_text') for item in rejected])}`")
    if evidence.get("claim_conflicts"):
        lines.append(f"- Claim conflicts: `{len(evidence['claim_conflicts'])}`")
    lines.append("- Main recommendation: Use analogous claims as experiments, not as confirmed facts.")
    return _join(lines)


def _interaction_diagnostics_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.interaction_diagnostics)
    lines = ["## Interaction diagnostics"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'interaction_diagnostics')}")
        return _join(lines)
    status = evidence.get("status", "unknown")
    lines.append(f"- Status: `{status}`")
    if status in {"skipped", "not_testable", "failed"}:
        if evidence.get("reason") or evidence.get("error_message"):
            lines.append(f"- Reason: {evidence.get('reason') or evidence.get('error_message')}")
        return _join(lines)
    selection = _as_dict(evidence.get("candidate_selection"))
    lines.append(f"- Candidate columns: `{len(selection.get('numeric_columns', []))} numeric, {len(selection.get('categorical_columns', []))} categorical`")
    lines.append(f"- Reported pairs: `{len(evidence.get('numeric_numeric', []))} numeric-numeric, {len(evidence.get('numeric_categorical', []))} numeric-categorical, {len(evidence.get('categorical_categorical', []))} categorical-categorical, {len(evidence.get('missingness_interactions', []))} missingness`")
    lines.append(f"- Redundancy groups: `{len(evidence.get('redundancy_groups', []))}`")
    hypotheses = [_as_dict(item) for item in evidence.get("interaction_hypotheses", [])]
    if hypotheses:
        rendered = [f"{' x '.join(item.get('columns', []))}: {item.get('materiality')}, {item.get('reliability')}" for item in hypotheses[:3]]
        lines.append(f"- Top interaction hypotheses: `{_csv(rendered)}`")
    cautions = [item.get("reliability") for group in (evidence.get("numeric_categorical", []), evidence.get("categorical_categorical", [])) for item in group if _as_dict(item).get("reliability", "reliable") != "reliable"]
    if cautions:
        lines.append(f"- Main caution: `{cautions[0]}`")
    return _join(lines)


def _baseline_ablations_section(pack: EdaEvidencePack) -> str:
    evidence = _as_dict(pack.baseline_ablation_evidence)
    lines = ["## Baseline ablations"]
    if not evidence:
        lines.append(f"- {_module_absence(pack, 'baseline_ablation_runner')}")
        return _join(lines)
    status = evidence.get("status", "unknown")
    lines.append(f"- Status: `{status}`")
    if status in {"skipped", "not_testable", "failed"}:
        reason = evidence.get("reason") or evidence.get("error_message")
        if reason:
            lines.append(f"- Reason: {reason}")
        return _join(lines)
    if evidence.get("metric_name"):
        lines.append(f"- Metric: `{evidence['metric_name']}`")
    reference = _as_dict(evidence.get("baseline_reference"))
    if reference:
        lines.append(
            f"- Reference ablation: `{reference.get('ablation_id')} = {reference.get('metric_value')}`"
        )
    best = _as_dict(evidence.get("best_ablation"))
    if best:
        lines.append(f"- Best ablation: `{best.get('ablation_id')} = {best.get('metric_value')}`")
        simpler_id = best.get("simpler_competitive_ablation_id")
        if simpler_id:
            lines.append(f"- Simpler competitive ablation: `{simpler_id} = {best.get('simpler_competitive_metric_value')}`")
            delta = best.get("delta_vs_simpler_competitive")
            materiality = best.get("materiality_vs_simpler_competitive")
            if isinstance(delta, (int, float)):
                lines.append(f"- Best-vs-simpler delta: `{delta:+.6f}` ({materiality or 'unknown'})")
        stability = best.get("stability_vs_best_prior")
        if stability:
            lines.append(f"- Best-vs-prior fold stability: `{stability}`")
    findings = [_as_dict(item) for item in evidence.get("feature_block_findings", [])]
    if findings:
        rendered = []
        for finding in findings[:6]:
            label = finding.get("feature_block") or finding.get("configuration") or "configuration"
            delta = finding.get("delta_vs_best_prior", finding.get("delta_metric"))
            materiality = finding.get("materiality", finding.get("materiality_vs_best_prior"))
            stability = finding.get("stability", finding.get("stability_vs_best_prior"))
            suffix = f" ({delta:+.6f})" if isinstance(delta, (int, float)) else ""
            qualifiers = " + ".join(str(value) for value in (materiality, stability) if value)
            wins = finding.get("fold_wins")
            losses = finding.get("fold_losses")
            win_suffix = f", wins {wins}, losses {losses}" if isinstance(wins, int) and isinstance(losses, int) else ""
            rendered.append(f"{label}: {finding.get('status')}{suffix}{f' [{qualifiers}]' if qualifiers else ''}{win_suffix}")
        lines.append(f"- Feature block findings: `{_csv(rendered)}`")
    configurations = [item for item in findings if item.get("finding_type") == "configuration"]
    if configurations:
        recommendation = configurations[0].get("recommendation")
        if recommendation:
            lines.append(f"- Recommendation: {recommendation}")
    limitations = list(evidence.get("limitations") or [])
    if limitations:
        lines.append(f"- Limitations: `{_csv(limitations[:2])}`")
    return _join(lines)


def _numeric_policy_summary(preprocessing: dict[str, Any]) -> str:
    numeric = _as_dict(preprocessing.get("numeric"))
    if not numeric:
        return ""
    imputation = _as_dict(numeric.get("imputation")).get("strategy")
    scaling = _as_dict(numeric.get("scaling"))
    parts = []
    if imputation:
        parts.append(f"{imputation} imputation")
    if scaling.get("enabled"):
        parts.append(str(scaling.get("strategy") or "scaling"))
    else:
        parts.append("no scaling")
    return ", ".join(parts)


def _categorical_policy_summary(preprocessing: dict[str, Any]) -> str:
    categorical = _as_dict(preprocessing.get("categorical"))
    if not categorical:
        return ""
    missing = _as_dict(categorical.get("missing_value_handling")).get("strategy")
    encoding = _as_dict(categorical.get("encoding"))
    parts = []
    if missing:
        parts.append(f"{missing} missing handling")
    if encoding.get("strategy"):
        unknown = encoding.get("handle_unknown")
        suffix = f", unknown={unknown}" if unknown else ""
        parts.append(f"{encoding['strategy']} encoding{suffix}")
    return ", ".join(parts)


def _high_cardinality_policy_summary(preprocessing: dict[str, Any]) -> str:
    high_cardinality = _as_dict(preprocessing.get("high_cardinality"))
    if not high_cardinality:
        return ""
    strategy = high_cardinality.get("strategy")
    if not strategy or strategy == "not_applicable":
        return ""
    encoding = high_cardinality.get("encoding_strategy")
    suffix = f" using fold-fitted {encoding}" if encoding else ""
    return f"{strategy}{suffix}"


def _excluded_columns_summary(details: Any) -> str:
    grouped: dict[str, list[str]] = {}
    for item in details or []:
        detail = _as_dict(item)
        column = detail.get("column")
        reason = detail.get("reason")
        if not column or not reason:
            continue
        grouped.setdefault(str(reason), []).append(str(column))
    return "; ".join(
        f"{reason}: {_csv(columns)}" for reason, columns in grouped.items()
    )


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


def _safety_constraints_section(pack: EdaEvidencePack) -> str:
    rows = [_as_dict(item) for item in pack.safety_constraints][:8]
    lines = ["## Safety constraints"]
    if not rows:
        lines.append("- None recorded.")
        return _join(lines)
    for item in rows:
        lines.append(f"- [{item.get('severity', 'mandatory')}] {item.get('rule')}{_refs(item.get('evidence_refs', []))}")
    return _join(lines)


def _validation_requirements_section(pack: EdaEvidencePack) -> str:
    rows = [_as_dict(item) for item in pack.validation_requirements][:6]
    lines = ["## Validation requirements"]
    if not rows:
        lines.append("- None recorded.")
        return _join(lines)
    for item in rows:
        condition = f" Condition: {item.get('condition')}" if item.get("condition") else ""
        lines.append(f"- [{item.get('status', 'conditional')}] {item.get('rule')}{condition}{_refs(item.get('evidence_refs', []))}")
    return _join(lines)


def _testable_hypotheses_section(pack: EdaEvidencePack) -> str:
    rows = [_as_dict(item) for item in pack.testable_hypotheses][:10]
    lines = ["## Testable follow-up hypotheses"]
    if not rows:
        lines.append("- None recorded.")
        return _join(lines)
    for item in rows:
        lines.append(f"- [{item.get('priority_signal', 'optional')}] {item.get('statement')}")
        lines.append(f"  Evidence: {_csv(item.get('evidence_refs', []))}")
        lines.append(f"  Controls: {_csv(item.get('required_controls', []))}")
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
