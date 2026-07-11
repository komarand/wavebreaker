from __future__ import annotations

from typing import Any


def build_eda_strategy_hints(evidence_pack_partial: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    validation = _as_dict(evidence_pack_partial.get("validation_evidence"))
    metric = _as_dict(evidence_pack_partial.get("metric_evidence"))
    drift = _as_dict(evidence_pack_partial.get("drift_evidence"))
    baseline = _as_dict(evidence_pack_partial.get("baseline_evidence"))
    feature_diagnostics = _as_dict(evidence_pack_partial.get("feature_diagnostics"))
    leakage = [_as_dict(item) for item in evidence_pack_partial.get("leakage_evidence", [])]

    hints = {
        "validation": _validation_hints(validation, metric),
        "leakage": _leakage_hints(leakage),
        "feature_engineering": _feature_hints(feature_diagnostics),
        "drift_and_leaderboard": _drift_hints(drift),
        "baseline": _baseline_hints(baseline, metric),
        "do_not_do": _do_not_do_hints(validation, feature_diagnostics),
        "first_experiments": _first_experiments(validation, metric, feature_diagnostics, drift),
    }
    return {key: _dedupe(items) for key, items in hints.items()}


def _validation_hints(validation: dict[str, Any], metric: dict[str, Any]) -> list[dict[str, Any]]:
    primary = _as_dict(validation.get("primary_validation"))
    method = str(primary.get("method") or "").lower()
    hints: list[dict[str, Any]] = []
    if method:
        action = {
            "stratified_kfold": "Use StratifiedKFold for model validation.",
            "kfold": "Use KFold for model validation.",
            "group_kfold": "Respect grouped validation splits.",
            "stratified_group_kfold": "Respect grouped validation splits.",
            "ranking_group_cv": "Use query/group-aware validation for ranking metrics.",
            "temporal_holdout": "Use temporal validation for model selection.",
            "temporal_cv": "Use temporal validation for model selection.",
            "expanding_window": "Use temporal validation for model selection.",
        }.get(method, f"Use {method} for model comparison.")
        hints.append(_hint("P0", action, "EDA selected this primary validation policy.", ["validation_evidence.primary_validation"], "low", ["validation", "model_selection"]))
    if metric.get("requires_threshold"):
        hints.append(_hint("P0", "Tune thresholds only inside validation folds.", "Metric evidence requires thresholded outputs.", ["metric_evidence.requires_threshold"], "medium", ["validation", "metric"]))
    diagnostics = validation.get("diagnostic_validations", [])
    if diagnostics and method not in {"temporal_holdout", "expanding_window", "temporal_cv"}:
        hints.append(_hint("P1", "Keep temporal splits diagnostic unless additional evidence requires them.", "Validation evidence did not select temporal validation as primary.", ["validation_evidence.diagnostic_validations"], "low", ["validation"]))
    return hints


def _leakage_hints(leakage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints = [
        _hint("P0", "Exclude target, prediction, primary ID, and submission-only columns from model features.", "Schema and leakage diagnostics reserve these columns for roles, not ordinary features.", ["inferred_schema.global_roles"], "high", ["leakage", "features"]),
        _hint("P1", "Use out-of-fold target encoding only.", "Naive target encoding can leak target information across folds.", ["feature_probe_evidence"], "high", ["leakage", "feature_engineering"]),
    ]
    risky = [item for item in leakage if item.get("status") in {"failed", "warning"}]
    if risky:
        hints.append(_hint("P0", "Resolve leakage warnings before trusting model gains.", "Leakage evidence contains failed or warning checks.", ["leakage_evidence"], "high", ["leakage"]))
    return hints


def _feature_hints(feature_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    numeric = _as_dict(feature_diagnostics.get("numeric_feature_diagnostics"))
    categorical = _as_dict(feature_diagnostics.get("categorical_feature_diagnostics"))
    missingness = _as_dict(feature_diagnostics.get("missingness_diagnostics"))
    text = _as_dict(feature_diagnostics.get("text_feature_diagnostics"))
    date_time = _as_dict(feature_diagnostics.get("date_time_diagnostics"))
    if numeric.get("top_predictive_candidates"):
        hints.append(_hint("P1", "Start with safe numeric base features.", "Numeric diagnostics found usable non-role numeric columns.", ["feature_diagnostics.numeric_feature_diagnostics"], "low", ["feature_engineering"]))
    if categorical.get("low_cardinality_candidates"):
        hints.append(_hint("P1", "Add fold-fitted categorical encoders.", "Categorical diagnostics found low-cardinality candidates.", ["feature_diagnostics.categorical_feature_diagnostics"], "medium", ["feature_engineering"]))
    if categorical.get("high_cardinality_candidates"):
        hints.append(_hint("P1", "Treat high-cardinality categoricals carefully with rare handling or fold-fitted encoders.", "Categorical diagnostics found high-cardinality columns.", ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"], "medium", ["feature_engineering"]))
    if missingness.get("recommended_indicators"):
        hints.append(_hint("P1", "Evaluate missingness indicators.", "Missingness diagnostics found meaningful missing values.", ["feature_diagnostics.missingness_diagnostics.recommended_indicators"], "low", ["feature_engineering"]))
    if text.get("columns"):
        hints.append(_hint("P2", "Extract simple text/code features before heavier NLP.", "Text diagnostics found free-text or code-like columns.", ["feature_diagnostics.text_feature_diagnostics"], "medium", ["feature_engineering"]))
    if date_time.get("columns"):
        hints.append(_hint("P2", "Add date/time extraction as diagnostics-backed features.", "Date/time diagnostics found parseable or named date/time columns.", ["feature_diagnostics.date_time_diagnostics"], "low", ["feature_engineering"]))
    return hints


def _drift_hints(drift: dict[str, Any]) -> list[dict[str, Any]]:
    severity = drift.get("feature_drift_severity") or drift.get("severity")
    if severity in {"medium", "high"}:
        return [_hint("P1", "Treat safe-feature drift as leaderboard risk.", "Drift evidence shows medium/high safe-feature drift.", ["drift_evidence.feature_drift_severity"], "medium", ["drift", "leaderboard"])]
    if drift.get("id_artifact_drift", {}).get("columns"):
        return [_hint("P2", "Do not overreact to ID/index drift artifacts.", "Drift evidence excludes ID/group artifacts from feature severity.", ["drift_evidence.id_artifact_drift"], "low", ["drift"])]
    return []


def _baseline_hints(baseline: dict[str, Any], metric: dict[str, Any]) -> list[dict[str, Any]]:
    if baseline.get("status") == "completed":
        return [_hint("P1", "Use the EDA baseline as a reproducible sanity floor.", "Baseline runner completed under the selected validation policy.", ["baseline_evidence"], "low", ["baseline"])]
    return [_hint("P2", "Start with a simple model family matching the task type.", "Baseline runner is optional or skipped; metric evidence identifies task semantics.", ["metric_evidence.task_type"], "low", ["baseline"])]


def _do_not_do_hints(validation: dict[str, Any], feature_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    hints = [
        _hint("P0", "Do not use primary IDs as ordinary predictive features.", "Column role policy excludes primary IDs from feature diagnostics.", ["inferred_schema.primary_id_column"], "high", ["do_not_do", "features"]),
        _hint("P0", "Do not tune on test or sample-submission outputs.", "Sample submission columns are prediction outputs only.", ["inferred_schema.sample_submission_table"], "high", ["do_not_do", "metric"]),
    ]
    date_time = _as_dict(feature_diagnostics.get("date_time_diagnostics"))
    primary = _as_dict(validation.get("primary_validation"))
    if date_time.get("temporal_validation_signal") == "diagnostic_only" and str(primary.get("method")) not in {"temporal_holdout", "expanding_window"}:
        hints.append(_hint("P1", "Do not force temporal CV from a date column alone.", "Date/time diagnostics are diagnostic only for this run.", ["feature_diagnostics.date_time_diagnostics"], "medium", ["do_not_do", "validation"]))
    return hints


def _first_experiments(
    validation: dict[str, Any],
    metric: dict[str, Any],
    feature_diagnostics: dict[str, Any],
    drift: dict[str, Any],
) -> list[dict[str, Any]]:
    experiments = [
        _hint("P0", "Build a baseline with safe numeric and categorical features.", "Validation and feature diagnostics define safe feature roles.", ["validation_evidence.primary_validation", "feature_diagnostics"], "low", ["first_experiments"]),
        _hint("P1", "Add missingness indicators and compare against baseline.", "Missingness diagnostics identify candidate indicators.", ["feature_diagnostics.missingness_diagnostics"], "low", ["first_experiments"]),
        _hint("P1", "Test high-cardinality treatment inside folds.", "Categorical diagnostics identify high-cardinality risks.", ["feature_diagnostics.categorical_feature_diagnostics"], "medium", ["first_experiments"]),
    ]
    if metric.get("requires_threshold"):
        experiments.append(_hint("P1", "Run validation-only threshold tuning.", "Metric evidence requires thresholded predictions.", ["metric_evidence.requires_threshold"], "medium", ["first_experiments"]))
    if (drift.get("feature_drift_severity") or drift.get("severity")) in {"medium", "high"}:
        experiments.append(_hint("P1", "Compare validation by shifted feature slices.", "Drift diagnostics found safe-feature shift.", ["drift_evidence"], "medium", ["first_experiments"]))
    return experiments[:10]


def _hint(priority: str, action: str, why: str, evidence_refs: list[str], risk: str, applies_to: list[str]) -> dict[str, Any]:
    return {
        "priority": priority,
        "action": action,
        "why": why,
        "evidence_refs": evidence_refs,
        "risk": risk,
        "applies_to": applies_to,
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not item.get("evidence_refs"):
            continue
        key = (str(item.get("action")), tuple(item.get("evidence_refs", [])))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


__all__ = ["build_eda_strategy_hints"]
