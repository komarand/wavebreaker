from __future__ import annotations

from typing import Any


def build_eda_strategy_hints(evidence_pack_partial: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    validation = _as_dict(evidence_pack_partial.get("validation_evidence"))
    metric = _as_dict(evidence_pack_partial.get("metric_evidence"))
    drift = _as_dict(evidence_pack_partial.get("drift_evidence"))
    baseline = _as_dict(evidence_pack_partial.get("baseline_evidence"))
    baseline_ablations = _as_dict(evidence_pack_partial.get("baseline_ablation_evidence"))
    interactions = _as_dict(evidence_pack_partial.get("interaction_diagnostics"))
    source_claims = _as_dict(evidence_pack_partial.get("source_claim_validation"))
    feature_diagnostics = _as_dict(evidence_pack_partial.get("feature_diagnostics"))
    target_diagnostics = _as_dict(evidence_pack_partial.get("target_diagnostics"))
    risk_register = [_as_dict(item) for item in evidence_pack_partial.get("eda_risk_register", [])]
    leakage = [_as_dict(item) for item in evidence_pack_partial.get("leakage_evidence", [])]

    hints = {
        "validation": _validation_hints(validation, metric) + _target_validation_hints(target_diagnostics),
        "leakage": _leakage_hints(leakage),
        "feature_engineering": _feature_hints(feature_diagnostics) + _target_feature_hints(target_diagnostics),
        "drift_and_leaderboard": _drift_hints(drift),
        "baseline": _baseline_hints(baseline, metric),
        "baseline_ablations": _baseline_ablation_hints(baseline_ablations),
        "interaction_diagnostics": _interaction_hints(interactions),
        "source_claim_validation": _source_claim_hints(source_claims),
        "risk_register": _risk_register_hints(risk_register),
        "do_not_do": _do_not_do_hints(validation, feature_diagnostics),
        "first_experiments": _first_experiments(validation, metric, feature_diagnostics, drift) + _target_experiment_hints(target_diagnostics),
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
    if categorical.get("high_cardinality_candidates") or categorical.get("unseen_category_risks"):
        hints.append(_hint("P1", "Treat high-cardinality categoricals as hypotheses requiring robust encoding and validation.", "High-cardinality or sparse categories can produce unreliable target associations and train/test category mismatch.", ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"], "medium", ["feature_engineering"]))
    if any(
        _as_dict(item).get("feature_value_type")
        in {"binary", "ordinal_low_cardinality", "count_zero_inflated", "count"}
        for item in numeric.get("columns", [])
    ):
        hints.append(_hint("P2", "Handle low-cardinality/count numeric features according to their value type, not as continuous outlier-heavy variables.", "Feature diagnostics classify some numeric columns as ordinal/count-like.", ["feature_diagnostics.numeric_feature_diagnostics"], "low", ["feature_engineering"]))
    if missingness.get("recommended_indicators"):
        hints.append(_hint("P1", "Evaluate missingness indicators.", "Missingness diagnostics found meaningful missing values.", ["feature_diagnostics.missingness_diagnostics.recommended_indicators"], "low", ["feature_engineering"]))
    if text.get("columns"):
        hints.append(_hint("P2", "Extract simple text/code features before heavier NLP.", "Text diagnostics found free-text or code-like columns.", ["feature_diagnostics.text_feature_diagnostics"], "medium", ["feature_engineering"]))
    if date_time.get("columns"):
        hints.append(_hint("P2", "Add date/time extraction as diagnostics-backed features.", "Date/time diagnostics found parseable or named date/time columns.", ["feature_diagnostics.date_time_diagnostics"], "low", ["feature_engineering"]))
    return hints


def _target_validation_hints(target_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    if target_diagnostics.get("status") != "completed":
        return []
    hints: list[dict[str, Any]] = []
    distribution = _as_dict(target_diagnostics.get("distribution"))
    imbalance = _as_dict(target_diagnostics.get("imbalance"))
    if distribution.get("target_type") in {"binary", "multiclass"}:
        hints.append(_hint("P0", "Preserve target distribution in validation folds.", "Classification target distribution must remain stable across folds.", ["target_diagnostics.distribution"], "medium", ["validation", "target"]))
    if imbalance.get("severity") in {"moderate", "severe", "extreme"}:
        hints.append(_hint("P1", "Track minority-class performance in validation diagnostics.", "Target imbalance can make aggregate metrics misleading.", ["target_diagnostics.imbalance"], "medium", ["metric", "validation", "target"]))
    if any(_as_dict(item).get("implication") == "fold_class_count_checks_required" for item in target_diagnostics.get("validation_implications", [])):
        hints.append(_hint("P0", "Check per-fold class counts before trusting validation scores.", "Severe imbalance requires every fold to contain enough minority examples.", ["target_diagnostics.imbalance"], "high", ["validation", "target"]))
    return hints


def _target_feature_hints(target_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    if target_diagnostics.get("status") != "completed":
        return []
    missingness = [
        _as_dict(item)
        for item in target_diagnostics.get("target_by_missingness", [])
        if float(_as_dict(item).get("absolute_difference") or 0.0) >= 0.2
    ]
    if not missingness:
        return []
    return [
        _hint(
            "P1",
            "Evaluate missingness indicators for columns whose missingness changes target behavior.",
            "Missingness is associated with target changes.",
            ["target_diagnostics.target_by_missingness"],
            "low",
            ["feature_engineering", "target"],
        )
    ]


def _target_experiment_hints(target_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    if target_diagnostics.get("status") != "completed":
        return []
    distribution = _as_dict(target_diagnostics.get("distribution"))
    if distribution.get("target_type") == "regression" and distribution.get("heavy_tail"):
        return [
            _hint(
                "P1",
                "Test target transforms or robust losses as validation hypotheses.",
                "Regression target has heavy-tail or outlier evidence; transforms are hypotheses to validate.",
                ["target_diagnostics.distribution"],
                "medium",
                ["metric", "validation", "target"],
            )
        ]
    return []


def _drift_hints(drift: dict[str, Any]) -> list[dict[str, Any]]:
    severity = drift.get("feature_drift_severity") or drift.get("severity")
    if severity in {"medium", "high"}:
        return [_hint("P1", "Treat safe-feature drift as leaderboard risk.", "Drift evidence shows medium/high safe-feature drift.", ["drift_evidence.feature_drift_severity"], "medium", ["drift", "leaderboard"])]
    if drift.get("id_artifact_drift", {}).get("columns"):
        return [_hint("P2", "Do not overreact to ID/index drift artifacts.", "Drift evidence excludes ID/group artifacts from feature severity.", ["drift_evidence.id_artifact_drift"], "low", ["drift"])]
    return []


def _baseline_hints(baseline: dict[str, Any], metric: dict[str, Any]) -> list[dict[str, Any]]:
    if baseline.get("status") == "completed":
        refs = ["baseline_evidence.metric_value", "baseline_evidence.preprocessing_policy"]
        return [_hint("P1", "Compare future experiments against the completed EDA baseline.", "Baseline runner completed with documented fold-safe preprocessing.", refs, "low", ["baseline", "model_selection"])]
    return [_hint("P0", "Run a simple fold-safe baseline before advanced modeling.", "No completed baseline evidence is available.", ["validation_evidence.primary_validation", "feature_diagnostics"], "medium", ["baseline"])]


def _baseline_ablation_hints(ablations: dict[str, Any]) -> list[dict[str, Any]]:
    if ablations.get("status") != "completed":
        return []
    hints: list[dict[str, Any]] = []
    for finding in [_as_dict(item) for item in ablations.get("feature_block_findings", [])]:
        finding_type = finding.get("finding_type", "feature_block")
        block = finding.get("feature_block")
        status = finding.get("status")
        ref = f"baseline_ablation_evidence.feature_block_findings.{block or finding.get('configuration')}"
        if finding_type == "configuration":
            if finding.get("materiality_vs_best_prior") == "negligible":
                hints.append(_hint("P1", "Prefer the simpler feature configuration until the added complexity shows a material gain.", "The more complex configuration was only negligibly better than a simpler prior configuration.", ["baseline_ablation_evidence.complexity_tradeoffs", ref], "low", ["feature_engineering", "baseline"]))
            elif finding.get("status") in {"unstable", "competitive"}:
                hints.append(_hint("P2", "Retest the composite feature configuration before adopting it by default.", "Its marginal paired-fold improvement was small or inconsistent.", [ref], "medium", ["feature_engineering", "baseline"]))
            continue
        if status == "helped" and finding.get("materiality") == "material" and finding.get("stability") == "stable":
            hints.append(_hint("P1", "Prioritize the feature block that produced a stable material validation improvement.", "Paired fold comparison showed a material and stable gain.", [ref], "low", ["feature_engineering", "baseline"]))
        elif status == "unstable" or finding.get("materiality") == "small":
            hints.append(_hint("P2", "Retest the feature block in a controlled experiment before adopting it by default.", "The aggregate improvement was small or inconsistent across folds.", [ref], "medium", ["feature_engineering", "baseline"]))
        elif block == "high_cardinality_categorical" and status in {"hurt", "neutral"}:
            hints.append(_hint("P1", "Treat high-cardinality categorical features as controlled experiments, not default features.", "Baseline ablation did not show stable benefit from this block.", [ref], "medium", ["feature_engineering", "risk"]))
    return hints


def _source_claim_hints(claims: dict[str, Any]) -> list[dict[str, Any]]:
    if claims.get("status") != "completed":
        return []
    hints = []
    if claims.get("final_strategy_claims", {}).get("adopt"):
        hints.append(_hint("P1", "Use source claims only where current EDA directly confirms them.", "Source claim validation found claims with direct current-dataset evidence.", ["source_claim_validation.validated_claims"], "low", ["strategy", "validation"]))
    if claims.get("final_strategy_claims", {}).get("test_as_hypothesis"):
        hints.append(_hint("P2", "Treat unconfirmed or analogous source advice as controlled experiments.", "Source evidence without direct EDA confirmation remains hypothesis-generating evidence.", ["source_claim_validation.recommended_experiments"], "medium", ["experiments", "source_claims"]))
    if claims.get("final_strategy_claims", {}).get("reject"):
        hints.append(_hint("P0", "Reject source advice that violates leakage, role, validation, or metric safety.", "Source claim validation identified unsafe or contradicted advice.", ["source_claim_validation.validated_claims"], "high", ["leakage", "do_not_do"]))
    return hints


def _interaction_hints(interactions: dict[str, Any]) -> list[dict[str, Any]]:
    if interactions.get("status") != "completed":
        return []
    hints: list[dict[str, Any]] = []
    reliable = [item for item in interactions.get("interaction_hypotheses", []) if _as_dict(item).get("materiality") == "material" and _as_dict(item).get("reliability") == "reliable"]
    if reliable:
        hints.append(_hint("P1", "Test the highest-confidence feature interactions in isolated fold-safe experiments.", "Interaction diagnostics found reliable pairwise effects beyond individual feature evidence.", ["interaction_diagnostics.interaction_hypotheses"], "medium", ["feature_engineering", "experiments"]))
    if interactions.get("redundancy_groups"):
        hints.append(_hint("P2", "Ablate redundant feature groups before increasing model complexity.", "Strongly related features may add complexity without independent signal.", ["interaction_diagnostics.redundancy_groups"], "low", ["feature_engineering", "modeling"]))
    sparse = [_as_dict(item) for item in interactions.get("categorical_categorical", []) if _as_dict(item).get("reliability") in {"caution_sparse_combinations", "caution_test_mismatch"}]
    if sparse:
        hints.append(_hint("P1", "Avoid uncontrolled categorical crosses with sparse or unseen combinations.", "Sparse combinations can overfit and fail to generalize to test data.", ["interaction_diagnostics.categorical_categorical"], "high", ["feature_engineering", "do_not_do"]))
    return hints


def _risk_register_hints(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for risk in risks:
        severity = str(risk.get("severity") or "")
        risk_type = str(risk.get("risk_type") or "")
        status = str(risk.get("status") or "")
        if severity in {"info", "low"} or status in {"informational", "resolved"}:
            continue
        ref = f"eda_risk_register.{risk.get('risk_id')}"
        if risk_type == "leakage" and severity in {"critical", "high"}:
            hints.append(_hint("P0", "Exclude leakage-prone role columns from model features.", "Risk register identifies a high-severity leakage risk.", [ref], "high", ["leakage", "features"]))
        elif risk_type in {"drift", "leaderboard"} and severity in {"critical", "high", "medium"}:
            hints.append(_hint("P1", "Treat train/test feature drift as leaderboard risk and monitor CV/LB gap.", "Risk register identifies safe-feature drift or leaderboard transfer risk.", [ref], "medium", ["drift", "leaderboard"]))
        elif risk_type == "validation" and severity in {"critical", "high"}:
            hints.append(_hint("P0", "Resolve high-severity validation risks before comparing models.", "Risk register identifies a high-severity validation risk.", [ref], "high", ["validation", "model_selection"]))
        elif risk_type == "metric" and severity in {"critical", "high", "medium"}:
            hints.append(_hint("P0" if severity == "high" else "P1", "Lock metric-compatible predictions and postprocessing inside validation.", "Risk register identifies a metric-sensitive modeling risk.", [ref], "medium", ["metric", "validation"]))
        elif risk_type == "target" and severity in {"critical", "high"}:
            hints.append(_hint("P0", "Audit target-driven validation checks before modeling.", "Risk register identifies a high-severity target risk.", [ref], "high", ["target", "validation"]))
        elif risk_type == "high_cardinality" and severity == "medium":
            hints.append(_hint("P1", "Use robust high-cardinality encoding and validate its lift.", "Risk register identifies high-cardinality feature risk.", [ref], "medium", ["feature_engineering"]))
        elif risk_type == "missingness" and severity == "medium":
            hints.append(_hint("P1", "Evaluate missingness indicators with fold-safe imputation.", "Risk register identifies missingness risk.", [ref], "medium", ["feature_engineering", "missingness"]))
        elif risk_type == "baseline" and status == "skipped":
            hints.append(_hint("P0", "Run a simple fold-safe baseline before advanced modeling.", "Risk register identifies missing baseline evidence.", [ref], "medium", ["baseline"]))
    return hints[:8]


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
