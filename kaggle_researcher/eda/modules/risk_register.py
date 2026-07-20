from __future__ import annotations

import re
from collections import Counter
from typing import Any

from kaggle_researcher.eda.schemas import EdaRisk


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
SEVERITIES = set(SEVERITY_ORDER)
STATUS_ORDER = {
    "confirmed": 0,
    "suspected": 1,
    "mitigated_by_policy": 2,
    "not_testable": 3,
    "skipped": 4,
    "resolved": 5,
    "informational": 6,
}
STATUSES = set(STATUS_ORDER)
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
RISK_TYPE_ORDER = {
    "leakage": 0,
    "validation": 1,
    "metric": 2,
    "target": 3,
    "drift": 4,
    "high_cardinality": 5,
    "missingness": 6,
    "baseline": 7,
    "schema": 8,
    "relationship": 9,
    "submission": 10,
    "feature_engineering": 11,
    "data_quality": 12,
    "leaderboard": 13,
    "notebook_source": 14,
    "unsupported": 15,
}
KNOWN_RISK_INTENTS = {
    "primary_id_feature_leakage",
    "sample_submission_feature_leakage",
    "naive_target_encoding_leakage",
    "oof_target_encoding_policy",
    "safe_feature_drift",
    "id_artifact_drift",
    "threshold_metric_risk",
    "target_distribution_validation",
    "target_fold_class_count",
    "target_imbalance_metric",
    "high_cardinality_encoding",
    "high_cardinality_target_association_reliability",
    "missingness_informative",
    "missingness_shift",
    "baseline_completed",
    "baseline_skipped",
}
NOT_RISK_NOT_TESTABLE_CHECKS = {
    "future_time_risk",
    "group_overlap",
    "ranking_query_overlap",
}


def build_eda_risk_register(
    *,
    inferred_schema: dict[str, Any] | None,
    metric_evidence: dict[str, Any] | None,
    validation_evidence: dict[str, Any] | None,
    target_diagnostics: dict[str, Any] | None,
    leakage_evidence: list[dict[str, Any]] | None,
    drift_evidence: dict[str, Any] | None,
    relationship_evidence: dict[str, Any] | None,
    feature_probe_evidence: list[dict[str, Any]] | None,
    feature_diagnostics: dict[str, Any] | None,
    baseline_evidence: dict[str, Any] | None,
    baseline_ablation_evidence: dict[str, Any] | None = None,
    interaction_diagnostics: dict[str, Any] | None = None,
    source_claim_validation: dict[str, Any] | None = None,
    notebook_static_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic, concise, evidence-linked EDA risk register."""

    risks: list[dict[str, Any]] = []
    schema = _as_dict(inferred_schema)
    metric = _as_dict(metric_evidence)
    validation = _as_dict(validation_evidence)
    target = _as_dict(target_diagnostics)
    leakage = [_as_dict(item) for item in leakage_evidence or []]
    drift = _as_dict(drift_evidence)
    relationships = _as_dict(relationship_evidence)
    probes = [_as_dict(item) for item in feature_probe_evidence or []]
    features = _as_dict(feature_diagnostics)
    baseline = _as_dict(baseline_evidence)
    ablations = _as_dict(baseline_ablation_evidence)
    interactions = _as_dict(interaction_diagnostics)
    source_claims = _as_dict(source_claim_validation)
    notebooks = _as_dict(notebook_static_analysis)

    _schema_risks(risks, schema)
    _metric_risks(risks, metric)
    _validation_risks(risks, validation, target, schema)
    _leakage_risks(risks, leakage, probes)
    _drift_risks(risks, drift)
    _feature_risks(risks, features, target)
    _target_risks(risks, target)
    _baseline_risks(risks, baseline)
    _ablation_risks(risks, ablations)
    _interaction_risks(risks, interactions)
    _source_claim_risks(risks, source_claims)
    _relationship_risks(risks, relationships, schema)
    _notebook_risks(risks, notebooks)

    return _assign_ids(deduplicate_eda_risks(risks))


def risk_summary(risks: list[dict[str, Any]] | list[EdaRisk]) -> dict[str, Any]:
    payloads = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in risks
    ]
    return {
        "total": len(payloads),
        "by_severity": dict(Counter(str(item.get("severity")) for item in payloads)),
        "by_type": dict(Counter(str(item.get("risk_type")) for item in payloads)),
        "by_status": dict(Counter(str(item.get("status")) for item in payloads)),
    }


def deduplicate_eda_risks(risks: list[dict[str, Any]] | list[EdaRisk]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for item in risks:
        risk = _risk_payload(item)
        risk["risk_intent"] = _risk_intent(risk)
        key = (str(risk.get("risk_type") or ""), str(risk.get("risk_intent") or ""))
        if key not in groups:
            groups[key] = risk
            continue
        _merge_risk(groups[key], risk)
    return sorted(groups.values(), key=_sort_key)


def validate_eda_risk_register(risks: list[dict[str, Any]] | list[EdaRisk]) -> list[str]:
    warnings: list[str] = []
    seen_title: dict[tuple[str, str], str] = {}
    seen_intent: dict[str, str] = {}
    for index, item in enumerate(risks):
        risk = _risk_payload(item)
        risk_id = str(risk.get("risk_id") or f"risk[{index}]")
        risk_type = str(risk.get("risk_type") or "")
        title = str(risk.get("title") or "").strip()
        intent = _risk_intent(risk)
        if not title:
            warnings.append(f"{risk_id} has empty title.")
        if not risk.get("evidence_refs"):
            warnings.append(f"{risk_id} has empty evidence_refs.")
        severity = str(risk.get("severity") or "")
        if severity not in SEVERITIES:
            warnings.append(f"{risk_id} has invalid severity: {severity}.")
        status = str(risk.get("status") or "")
        if status not in STATUSES:
            warnings.append(f"{risk_id} has invalid status: {status}.")
        if severity in {"critical", "high"} and not str(risk.get("mitigation") or "").strip():
            warnings.append(f"{risk_id} is {severity} but has no mitigation.")
        title_key = (risk_type, _normalize_text(title))
        if title_key in seen_title:
            warnings.append(
                f"{risk_id} duplicates normalized title with {seen_title[title_key]}."
            )
        else:
            seen_title[title_key] = risk_id
        if intent in seen_intent:
            warnings.append(f"{risk_id} duplicates risk_intent with {seen_intent[intent]}.")
        else:
            seen_intent[intent] = risk_id
    return warnings


def _schema_risks(risks: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    target_column = schema.get("target_column")
    confidence = str(schema.get("confidence") or "")
    if not target_column or confidence == "low":
        _add(
            risks,
            "schema",
            "high",
            "suspected",
            "medium",
            "Target column inference is uncertain",
            "Target inference is missing or low confidence.",
            "Modeling and validation may optimize the wrong objective if the target role is wrong.",
            "Require manual target confirmation before final modeling.",
            ["schema", "validation", "modeling"],
            ["inferred_schema.target_column", "inferred_schema.confidence"],
            ["Confirm the target column before final modeling."],
        )
    if schema.get("primary_id_column"):
        _add(
            risks,
            "leakage",
            "high",
            "mitigated_by_policy",
            "high",
            "Primary ID must not be used as a model feature",
            "A primary ID column was inferred and excluded by role policy.",
            "Using primary IDs as features can cause overfitting or train/test artifact learning.",
            "Exclude primary IDs from model features and use them only for row tracking/submission alignment.",
            ["features", "leakage", "modeling"],
            ["inferred_schema.primary_id_column"],
            ["Do not use primary IDs as ordinary predictive features."],
        )
    if schema.get("sample_submission_table"):
        _add(
            risks,
            "submission",
            "medium",
            "mitigated_by_policy",
            "high",
            "Sample submission outputs are not training features",
            "A sample submission table was inferred as an output schema source.",
            "Using submission outputs as features would leak output format artifacts into modeling.",
            "Use sample submission only for output schema alignment.",
            ["submission", "features", "leakage"],
            ["inferred_schema.sample_submission_table"],
            ["Keep sample submission columns out of model features."],
        )


def _metric_risks(risks: list[dict[str, Any]], metric: dict[str, Any]) -> None:
    if metric.get("requires_threshold"):
        _add(
            risks,
            "metric",
            "medium",
            "confirmed",
            "high",
            "Threshold choice affects metric score",
            "Metric evidence requires thresholded predictions.",
            "Tuning thresholds outside validation folds can overfit validation or leaderboard feedback.",
            "Tune thresholds only inside validation folds.",
            ["metric", "validation", "model_selection"],
            ["metric_evidence.requires_threshold"],
            ["Tune thresholds only inside validation folds."],
        )
    if metric.get("requires_probabilities"):
        _add(
            risks,
            "metric",
            "high",
            "confirmed",
            "high",
            "Metric requires probability or score outputs",
            "Metric evidence requires probability or ranking-score style predictions.",
            "Hard labels may produce invalid or poor metric values.",
            "Use models or postprocessing that produce calibrated probability or score outputs.",
            ["metric", "modeling"],
            ["metric_evidence.requires_probabilities"],
            ["Output probabilities or ranking scores, not hard labels."],
        )
    if metric and metric.get("local_metric_available") is False:
        _add(
            risks,
            "metric",
            "high",
            "confirmed",
            "medium",
            "Metric needs local implementation validation",
            "Metric evidence says the local metric implementation is unavailable.",
            "Model comparison may optimize a proxy that does not match the competition metric.",
            "Implement and unit-test the local metric before model comparison.",
            ["metric", "model_selection"],
            ["metric_evidence.local_metric_available"],
            ["Implement and unit-test the competition metric locally."],
        )


def _validation_risks(
    risks: list[dict[str, Any]],
    validation: dict[str, Any],
    target: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    distribution = _as_dict(target.get("distribution"))
    target_type = distribution.get("target_type") or _metric_target_type(validation)
    if target_type in {"binary", "multiclass"}:
        _add(
            risks,
            "validation",
            "medium",
            "confirmed",
            "high",
            "Validation folds must preserve target distribution",
            "Classification target diagnostics require distribution-aware validation checks.",
            "Unbalanced folds can make validation metrics noisy or misleading.",
            "Use stratified validation unless group or time constraints override it.",
            ["validation", "target", "model_selection"],
            ["target_diagnostics.distribution", "validation_evidence.primary_validation"],
            ["Preserve target distribution in validation folds."],
        )
    imbalance = _as_dict(target.get("imbalance"))
    if imbalance.get("severity") in {"severe", "extreme"}:
        _add(
            risks,
            "validation",
            "high",
            "confirmed",
            "high",
            "Rare classes may be missing from folds",
            "Target diagnostics found severe or extreme class imbalance.",
            "Some folds may contain too few minority examples for trustworthy scores.",
            "Check per-fold class counts and track minority-class performance.",
            ["validation", "target", "metric"],
            ["target_diagnostics.imbalance"],
            ["Check per-fold class counts before trusting validation scores."],
        )
    primary = _as_dict(validation.get("primary_validation"))
    method = str(primary.get("method") or "").lower()
    if primary.get("group_column") or "group" in method:
        _add(
            risks,
            "validation",
            "high",
            "confirmed",
            "high",
            "Group leakage can invalidate validation",
            "Validation evidence selected or detected a group-aware split constraint.",
            "Rows from the same entity in train and validation can inflate scores.",
            "Use group-aware validation and keep group columns out of ordinary model features.",
            ["validation", "leakage", "model_selection"],
            ["validation_evidence.primary_validation"],
            ["Respect grouped validation splits."],
        )
    if method in {"temporal_holdout", "temporal_cv", "expanding_window"}:
        _add(
            risks,
            "validation",
            "high",
            "confirmed",
            "high",
            "Temporal leakage can invalidate validation",
            "Validation evidence selected a temporal validation policy.",
            "Training on future data relative to validation can overstate model quality.",
            "Preserve chronological order in validation and feature engineering.",
            ["validation", "leakage", "model_selection"],
            ["validation_evidence.primary_validation"],
            ["Use temporal validation for model selection."],
        )
    if not schema.get("target_column"):
        _add(
            risks,
            "validation",
            "medium",
            "not_testable",
            "medium",
            "Validation cannot be fully tested without target evidence",
            "No target column was available for target-aware validation diagnostics.",
            "Validation recommendations may be incomplete.",
            "Confirm the target column and rerun target-aware diagnostics.",
            ["validation", "schema"],
            ["inferred_schema.target_column"],
            [],
        )


def _leakage_risks(
    risks: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
    probes: list[dict[str, Any]],
) -> None:
    for item in leakage:
        status = item.get("status")
        check_id = str(item.get("check_id") or "leakage_check")
        if status == "failed":
            severity = "critical" if check_id == "target_in_test" else "high"
            title = {
                "target_in_test": "Target leakage is present in test-like data",
                "id_overlap": "Train/test ID overlap may indicate contamination",
                "duplicate_base_rows": "Duplicate base rows may contaminate validation",
            }.get(check_id, "Leakage check failed")
            _add(
                risks,
                "leakage",
                severity,
                "confirmed",
                "high",
                title,
                str(item.get("finding") or "A leakage check failed."),
                "Leakage can make validation and leaderboard conclusions unreliable.",
                "Resolve leakage findings before trusting model gains.",
                ["leakage", "validation", "modeling"],
                ["leakage_evidence"],
                ["Resolve leakage warnings before trusting model gains."],
            )
        elif status == "warning":
            _add(
                risks,
                "leakage",
                "medium",
                "suspected",
                "medium",
                "Leakage check produced a warning",
                str(item.get("finding") or "A leakage check produced a warning."),
                "Warning-level leakage can still distort validation if ignored.",
                "Audit the warning and exclude risky columns or rows as needed.",
                ["leakage", "validation"],
                ["leakage_evidence"],
                ["Audit warning-level leakage checks."],
            )
        elif status == "not_testable" and check_id not in NOT_RISK_NOT_TESTABLE_CHECKS:
            _add(
                risks,
                "leakage",
                "low",
                "not_testable",
                "low",
                "Leakage check was not testable",
                str(item.get("finding") or "A leakage check could not be tested."),
                "Untested leakage paths can leave residual uncertainty.",
                "Run the check when the required evidence becomes available.",
                ["leakage"],
                ["leakage_evidence"],
                [],
            )
    for probe in probes:
        family = str(probe.get("feature_family") or "")
        status = probe.get("status")
        if "naive_target_encoding" in family and status == "unsafe":
            _add(
                risks,
                "leakage",
                "high",
                "confirmed",
                "high",
                "Naive target encoding can leak target information",
                "Feature probe marked naive target encoding or WoE as unsafe.",
                "Encoding categories using validation targets can inflate validation scores.",
                "Use out-of-fold target encoding only.",
                ["leakage", "feature_engineering"],
                ["feature_probe_evidence"],
                ["Use out-of-fold target encoding only."],
            )
        if "oof_target_encoding" in family and status in {"medium_potential", "high_potential"}:
            _add(
                risks,
                "feature_engineering",
                "medium",
                "suspected",
                "medium",
                "Out-of-fold target encoding still requires careful fold policy",
                "Feature probe found OOF target encoding potentially useful.",
                "OOF encoders can still leak if fitted with the wrong grouping or time boundary.",
                "Fit encoders strictly inside validation folds and respect group/time splits.",
                ["feature_engineering", "leakage"],
                ["feature_probe_evidence"],
                ["Use out-of-fold target encoding only."],
            )


def _drift_risks(risks: list[dict[str, Any]], drift: dict[str, Any]) -> None:
    if not drift or drift.get("status") not in {None, "completed"}:
        return
    safe_severity = str(drift.get("feature_drift_severity") or drift.get("severity") or "").lower()
    if safe_severity in {"high", "medium"}:
        severity = "high" if safe_severity == "high" else "medium"
        _add(
            risks,
            "drift",
            severity,
            "confirmed",
            "high",
            "Train/test safe-feature drift may affect leaderboard reliability",
            "Drift evidence found medium or high safe-feature shift.",
            "Validation scores may not transfer cleanly to the leaderboard distribution.",
            "Track CV/LB gap and evaluate shifted feature slices.",
            ["drift", "leaderboard", "validation"],
            ["drift_evidence.feature_drift_severity"],
            ["Treat safe-feature drift as leaderboard risk."],
        )
    id_artifact = _as_dict(drift.get("id_artifact_drift"))
    artifact_severity = str(id_artifact.get("severity") or "").lower()
    if safe_severity not in {"high", "medium"} and (
        artifact_severity in {"high", "medium"} or id_artifact.get("columns")
    ):
        _add(
            risks,
            "drift",
            "low",
            "mitigated_by_policy",
            "medium",
            "ID/index drift artifact excluded from feature drift",
            "Drift evidence separated role-excluded ID/index artifacts from safe features.",
            "ID artifacts can create false drift alarms if treated as model features.",
            "Keep role-excluded ID/index columns out of model features.",
            ["drift", "features", "leakage"],
            ["drift_evidence.id_artifact_drift"],
            ["Do not overreact to ID/index drift artifacts."],
        )
    adversarial = _as_dict(drift.get("adversarial_validation"))
    drivers = adversarial.get("top_drivers") or adversarial.get("top_features") or []
    if safe_severity in {"high", "medium"} and _driver_names_suggest_sparse_text(drivers):
        _add(
            risks,
            "leaderboard",
            "medium" if safe_severity == "medium" else "high",
            "confirmed",
            "medium",
            "Adversarial drift may be driven by sparse text or high-cardinality features",
            "Adversarial validation drivers suggest sparse categorical or text-like feature drift.",
            "Simple encoders may overfit train/test distribution differences.",
            "Use robust encoding and check performance on shifted feature slices.",
            ["drift", "leaderboard", "feature_engineering"],
            ["drift_evidence.adversarial_validation"],
            ["Compare validation by shifted feature slices."],
            ["Driver names are a lightweight diagnostic and should be confirmed with feature diagnostics."],
        )


def _feature_risks(
    risks: list[dict[str, Any]],
    features: dict[str, Any],
    target: dict[str, Any],
) -> None:
    categorical = _as_dict(features.get("categorical_feature_diagnostics"))
    high_cardinality = categorical.get("high_cardinality_candidates") or []
    if high_cardinality:
        _add(
            risks,
            "high_cardinality",
            "medium",
            "confirmed",
            "high",
            "High-cardinality categorical features need robust encoding",
            "Feature diagnostics found high-cardinality categorical candidates.",
            "Sparse category levels can overfit validation and fail on unseen test categories.",
            "Use rare handling, frequency/hash encoding, or fold-fitted encoders; validate impact.",
            ["feature_engineering", "validation"],
            ["feature_diagnostics.categorical_feature_diagnostics.high_cardinality_candidates"],
            ["Treat high-cardinality categoricals as hypotheses requiring robust encoding and validation."],
        )
    if categorical.get("target_association_cautions"):
        _add(
            risks,
            "feature_engineering",
            "medium",
            "suspected",
            "medium",
            "High-cardinality target association may be unreliable",
            "Feature diagnostics flagged cautious or unreliable target association.",
            "Target rates in sparse categories can look predictive by chance or leakage.",
            "Treat derived features as hypotheses and require cross-validation evidence.",
            ["feature_engineering", "validation"],
            ["feature_diagnostics.categorical_feature_diagnostics.target_association_cautions"],
            ["Require CV validation for high-cardinality derived features."],
        )
    baseline_policy = _as_dict(features.get("baseline_preprocessing_policy"))
    baseline_high_cardinality = _as_dict(baseline_policy.get("high_cardinality"))
    if baseline_high_cardinality.get("columns"):
        _add(
            risks,
            "high_cardinality",
            "medium",
            "confirmed",
            "medium",
            "High-cardinality categorical features need robust encoding",
            "Baseline preprocessing policy included high-cardinality categorical columns with caution.",
            "Sparse category levels can overfit validation and fail on unseen test categories.",
            "Use rare handling, frequency/hash encoding, or fold-fitted encoders; validate impact.",
            ["feature_engineering", "validation"],
            ["feature_diagnostics.baseline_preprocessing_policy.high_cardinality"],
            ["Treat high-cardinality categoricals as hypotheses requiring robust encoding and validation."],
        )
    missingness = _as_dict(features.get("missingness_diagnostics"))
    if missingness.get("target_associated_missingness") or _target_missingness_signal(target):
        _add(
            risks,
            "missingness",
            "medium",
            "confirmed",
            "medium",
            "Missingness appears informative",
            "Missingness diagnostics found target-associated missing patterns.",
            "Ignoring missingness can discard predictive signal or introduce biased imputation.",
            "Evaluate missingness indicators and fold-safe imputation.",
            ["missingness", "feature_engineering", "target"],
            ["feature_diagnostics.missingness_diagnostics.target_associated_missingness"],
            ["Evaluate missingness indicators."],
        )
    if missingness.get("train_test_missingness_shift"):
        _add(
            risks,
            "missingness",
            "medium",
            "confirmed",
            "medium",
            "Missingness pattern differs between train and test",
            "Feature diagnostics found train/test missingness shift.",
            "Imputation behavior may not transfer cleanly to test data.",
            "Use robust imputation and monitor feature slice performance.",
            ["missingness", "drift", "feature_engineering"],
            ["feature_diagnostics.missingness_diagnostics.train_test_missingness_shift"],
            ["Monitor shifted missingness slices."],
        )


def _target_risks(risks: list[dict[str, Any]], target: dict[str, Any]) -> None:
    if target.get("status") and target.get("status") != "completed":
        return
    distribution = _as_dict(target.get("distribution"))
    imbalance = _as_dict(target.get("imbalance"))
    if imbalance.get("severity") in {"moderate", "severe", "extreme"}:
        severity = "high" if imbalance.get("severity") in {"severe", "extreme"} else "medium"
        _add(
            risks,
            "target",
            severity,
            "confirmed",
            "high",
            "Target imbalance can hide minority errors",
            "Target diagnostics found class imbalance.",
            "Aggregate metrics may hide poor minority-class performance.",
            "Track minority-class metrics and per-fold class counts.",
            ["target", "metric", "validation"],
            ["target_diagnostics.imbalance"],
            ["Track minority-class performance in validation diagnostics."],
        )
    if int(distribution.get("missing_target_count") or 0) > 0:
        _add(
            risks,
            "target",
            "high",
            "confirmed",
            "high",
            "Target has missing values",
            "Target diagnostics found missing target values.",
            "Rows without labels can corrupt training or metric computation if not handled.",
            "Exclude or explicitly handle unlabeled rows before model training.",
            ["target", "data_quality", "modeling"],
            ["target_diagnostics.distribution"],
            ["Handle missing target values before training."],
        )
    if distribution.get("is_constant") or distribution.get("near_constant"):
        _add(
            risks,
            "target",
            "critical",
            "confirmed",
            "high",
            "Target may be constant or near-constant",
            "Target diagnostics indicate little or no target variation.",
            "Model validation may be meaningless when the target has insufficient variation.",
            "Verify target extraction and competition task definition.",
            ["target", "schema", "validation"],
            ["target_diagnostics.distribution"],
            ["Verify target extraction before modeling."],
        )
    if distribution.get("heavy_tail"):
        _add(
            risks,
            "target",
            "medium",
            "confirmed",
            "medium",
            "Regression target has heavy-tail or outlier evidence",
            "Target diagnostics found heavy-tail or outlier behavior.",
            "Ordinary losses may overreact to extreme target values.",
            "Test target transforms or robust losses as validation hypotheses.",
            ["target", "metric", "modeling"],
            ["target_diagnostics.distribution"],
            ["Test target transforms or robust losses as validation hypotheses."],
        )


def _baseline_risks(risks: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    if not baseline or baseline.get("status") == "skipped":
        _add(
            risks,
            "baseline",
            "medium",
            "skipped",
            "high" if baseline else "medium",
            "No completed baseline evidence is available",
            str(baseline.get("reason") or "No completed baseline evidence is available."),
            "Advanced experiments lack a reproducible sanity floor.",
            "Run a simple fold-safe baseline before advanced modeling.",
            ["baseline", "model_selection"],
            ["baseline_evidence"],
            ["Run a simple fold-safe baseline before advanced modeling."],
        )
        return
    if baseline.get("status") != "completed":
        return
    policy = _as_dict(baseline.get("preprocessing_policy"))
    if not policy:
        _add(
            risks,
            "baseline",
            "medium",
            "suspected",
            "medium",
            "Baseline preprocessing policy is not auditable",
            "Baseline completed but no preprocessing policy was recorded.",
            "The baseline cannot be reliably reproduced or audited for fold safety.",
            "Record preprocessing policy and fold-safety checks.",
            ["baseline", "model_selection"],
            ["baseline_evidence"],
            ["Record baseline preprocessing policy."],
        )
    elif _as_dict(policy.get("safety_checks")).get("fits_preprocessing_inside_folds"):
        _add(
            risks,
            "baseline",
            "info",
            "informational",
            "high",
            "Completed baseline can serve as sanity floor",
            "Baseline completed with documented fold-safe preprocessing.",
            "Future experiments can be compared against a reproducible baseline.",
            "Compare future experiments against the completed EDA baseline.",
            ["baseline", "model_selection"],
            ["baseline_evidence.metric_value", "baseline_evidence.preprocessing_policy"],
            ["Compare future experiments against the completed EDA baseline."],
        )
    high_cardinality = _as_dict(policy.get("high_cardinality"))
    if high_cardinality.get("columns"):
        _add(
            risks,
            "high_cardinality",
            "medium",
            "confirmed",
            "medium",
            "High-cardinality categorical features need robust encoding",
            "Baseline preprocessing policy included high-cardinality categorical columns with caution.",
            "Sparse category levels can overfit validation and fail on unseen test categories.",
            "Use rare handling, frequency/hash encoding, or fold-fitted encoders; validate impact.",
            ["feature_engineering", "validation"],
            ["baseline_evidence.preprocessing_policy.high_cardinality"],
            ["Treat high-cardinality categoricals as hypotheses requiring robust encoding and validation."],
        )


def _ablation_risks(risks: list[dict[str, Any]], ablations: dict[str, Any]) -> None:
    if ablations.get("status") != "completed":
        return
    findings = [_as_dict(item) for item in ablations.get("feature_block_findings", [])]
    for finding in findings:
        block = finding.get("feature_block")
        status = finding.get("status")
        finding_type = finding.get("finding_type", "feature_block")
        ref = f"baseline_ablation_evidence.feature_block_findings.{block}"
        if finding_type == "configuration" and finding.get("materiality_vs_best_prior") == "negligible":
            _add(
                risks,
                "feature_engineering",
                "low",
                "informational",
                "medium",
                "Additional feature complexity produced negligible validation gain",
                "The composite baseline configuration was not materially better than a simpler prior configuration.",
                "Extra feature complexity can make experiments harder to reproduce without improving validation.",
                "Prefer the simpler competitive configuration until follow-up validation shows a material gain.",
                ["feature_engineering", "baseline"],
                ["baseline_ablation_evidence.complexity_tradeoffs"],
                ["Prefer the simpler feature configuration until the added complexity shows a material gain."],
            )
        if block == "high_cardinality_categorical" and status in {"hurt", "unstable"} and finding.get("materiality") in {"material", "material_negative", "small"}:
            _add(
                risks,
                "high_cardinality",
                "medium",
                "confirmed",
                "medium",
                "High-cardinality features showed unstable baseline contribution",
                "Baseline ablation did not show stable benefit from high-cardinality categorical features.",
                "Default inclusion can add variance or overfit sparse categories.",
                "Evaluate high-cardinality features in isolated fold-safe experiments.",
                ["feature_engineering", "baseline", "validation"],
                [ref],
                ["Treat high-cardinality categorical features as controlled experiments, not default features."],
            )
        if block == "missingness_indicators" and status == "helped" and finding.get("materiality") == "material":
            _add(
                risks,
                "missingness",
                "medium",
                "confirmed",
                "medium",
                "Missingness appears model-relevant",
                "Baseline ablation showed missingness indicators improved validation score.",
                "Missingness can encode important data-generation or target-related signal.",
                "Use fold-safe imputation and missingness indicators.",
                ["missingness", "feature_engineering", "baseline"],
                [ref],
                ["Include missingness indicators in the first feature engineering pass."],
            )


def _source_claim_risks(risks: list[dict[str, Any]], claims: dict[str, Any]) -> None:
    if claims.get("status") != "completed":
        return
    unsafe = [_as_dict(item) for item in claims.get("validated_claims", []) if _as_dict(item).get("validation_status") == "unsafe"]
    if unsafe:
        _add(risks, "notebook_source", "high", "confirmed", "high", "Source advice recommends leakage-prone preprocessing", "At least one collected source claim was rejected as unsafe by current EDA safety evidence.", "Following the claim could inflate validation performance or violate role policy.", "Reject the claim and use fold-safe preprocessing.", ["leakage", "notebook_source"], ["source_claim_validation.validated_claims", "leakage_evidence"], ["Reject the unsafe source advice and use fold-safe, metric-compatible practice."])
    contradicted = [_as_dict(item) for item in claims.get("validated_claims", []) if _as_dict(item).get("validation_status") == "contradicted"]
    if contradicted:
        _add(risks, "notebook_source", "medium", "confirmed", "medium", "Source claim conflicts with current-dataset evidence", "A material source claim was contradicted by schema, metric, validation, or ablation evidence.", "External advice can misdirect experimentation when presented as current-dataset fact.", "Treat contradicted claims as rejected and retain their evidence trail.", ["notebook_source", "validation"], ["source_claim_validation.contradicted_claims"], ["Reject this claim because current-dataset evidence conflicts with it."])


def _interaction_risks(risks: list[dict[str, Any]], interactions: dict[str, Any]) -> None:
    if interactions.get("status") != "completed":
        return
    sparse = [item for item in interactions.get("categorical_categorical", []) if _as_dict(item).get("reliability") in {"caution_sparse_combinations", "caution_test_mismatch"} and _as_dict(item).get("materiality") in {"material", "small"}]
    if sparse:
        _add(risks, "feature_engineering", "medium", "confirmed", "medium", "Sparse categorical interactions may overfit", "Interaction diagnostics found sparse or unseen categorical combinations with non-negligible apparent signal.", "Sparse crosses can overfit and fail on unseen test combinations.", "Use rare-combination handling and require stable fold-level gains.", ["feature_engineering", "validation"], ["interaction_diagnostics.categorical_categorical"], ["Avoid uncontrolled categorical crosses with sparse or unseen combinations."])
    if interactions.get("redundancy_groups"):
        _add(risks, "feature_engineering", "low", "informational", "medium", "Redundant feature groups can add avoidable complexity", "Interaction diagnostics identified strongly overlapping safe feature pairs.", "Overlapping features can complicate experiments without independent signal.", "Ablate representatives before increasing model complexity.", ["feature_engineering", "modeling"], ["interaction_diagnostics.redundancy_groups"], ["Ablate redundant feature groups before increasing model complexity."])


def _relationship_risks(
    risks: list[dict[str, Any]],
    relationships: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    tables = schema.get("tables") or []
    secondary_tables = [
        table for table in tables
        if _as_dict(table).get("table_type") in {"secondary", "depth_1", "depth_2"}
    ]
    rejected = relationships.get("rejected_relationships") or []
    if secondary_tables and rejected and not relationships.get("relationships"):
        _add(
            risks,
            "relationship",
            "medium",
            "suspected",
            "medium",
            "Secondary table relationships are ambiguous",
            "Relationship inference rejected candidate joins for secondary tables.",
            "Incorrect joins can duplicate rows or create aggregate leakage.",
            "Verify join keys before aggregation features.",
            ["relationship", "feature_engineering"],
            ["relationship_evidence.rejected_relationships"],
            ["Verify join keys before aggregation features."],
        )


def _notebook_risks(risks: list[dict[str, Any]], notebooks: dict[str, Any]) -> None:
    if not notebooks:
        _add(
            risks,
            "notebook_source",
            "info",
            "skipped",
            "low",
            "Notebook source patterns were not verified",
            "Notebook static analysis evidence is not available.",
            "No source-derived modeling or leaderboard patterns were checked.",
            "Run notebook static analysis when source notebooks are available.",
            ["notebook_source"],
            ["notebook_static_analysis"],
            [],
        )
        return
    if notebooks.get("suspicious_leaderboard_overfit_patterns"):
        _add(
            risks,
            "leaderboard",
            "high",
            "suspected",
            "medium",
            "Notebook source suggests leaderboard overfit risk",
            "Notebook static analysis found leaderboard-tuning patterns.",
            "Leaderboard-driven iteration can overfit public feedback.",
            "Audit notebook patterns and rely on validation-first model selection.",
            ["leaderboard", "notebook_source", "validation"],
            ["notebook_static_analysis.suspicious_leaderboard_overfit_patterns"],
            ["Do not tune strategy directly on public leaderboard feedback."],
        )


def _add(
    risks: list[dict[str, Any]],
    risk_type: str,
    severity: str,
    status: str,
    confidence: str,
    title: str,
    finding: str,
    impact: str,
    mitigation: str | None,
    applies_to: list[str],
    evidence_refs: list[str],
    related_actions: list[str],
    limitations: list[str] | None = None,
) -> None:
    risks.append(
        {
            "risk_id": "",
            "risk_intent": "",
            "risk_type": risk_type,
            "severity": severity,
            "status": status,
            "confidence": confidence,
            "title": title,
            "finding": finding,
            "impact": impact,
            "mitigation": mitigation,
            "applies_to": _unique(applies_to),
            "evidence_refs": _unique(evidence_refs),
            "related_actions": _unique(related_actions),
            "limitations": _unique(limitations or []),
        }
    )


def _assign_ids(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for risk in risks:
        risk_type = str(risk["risk_type"])
        counts[risk_type] = counts.get(risk_type, 0) + 1
        payload = dict(risk)
        payload["risk_id"] = f"risk_{risk_type}_{counts[risk_type]:03d}"
        result.append(EdaRisk(**payload).model_dump(mode="json"))
    return result


def _sort_key(risk: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        SEVERITY_ORDER.get(str(risk.get("severity")), 99),
        STATUS_ORDER.get(str(risk.get("status")), 99),
        RISK_TYPE_ORDER.get(str(risk.get("risk_type")), 99),
        str(risk.get("risk_intent") or ""),
        _normalize_text(str(risk.get("title") or "")),
    )


def _merge_risk(current: dict[str, Any], incoming: dict[str, Any]) -> None:
    if SEVERITY_ORDER[str(incoming["severity"])] < SEVERITY_ORDER[str(current["severity"])]:
        current["severity"] = incoming["severity"]
    if STATUS_ORDER[str(incoming["status"])] < STATUS_ORDER[str(current["status"])]:
        current["status"] = incoming["status"]
    if CONFIDENCE_ORDER[str(incoming["confidence"])] < CONFIDENCE_ORDER[str(current["confidence"])]:
        current["confidence"] = incoming["confidence"]
    if _title_score(incoming) > _title_score(current):
        current["title"] = incoming["title"]
        current["finding"] = incoming["finding"]
    if _mitigation_score(incoming.get("mitigation")) > _mitigation_score(current.get("mitigation")):
        current["mitigation"] = incoming.get("mitigation")
    if _text_score(incoming.get("impact")) > _text_score(current.get("impact")):
        current["impact"] = incoming.get("impact")
    current["evidence_refs"] = _unique([*current.get("evidence_refs", []), *incoming.get("evidence_refs", [])])
    current["applies_to"] = _unique([*current.get("applies_to", []), *incoming.get("applies_to", [])])
    current["related_actions"] = _unique([*current.get("related_actions", []), *incoming.get("related_actions", [])])
    current["limitations"] = _unique([*current.get("limitations", []), *incoming.get("limitations", [])])


def _risk_payload(value: dict[str, Any] | EdaRisk) -> dict[str, Any]:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
    payload["risk_id"] = str(payload.get("risk_id") or "")
    payload["risk_type"] = str(payload.get("risk_type") or "unsupported")
    payload["severity"] = str(payload.get("severity") or "info")
    payload["status"] = str(payload.get("status") or "informational")
    payload["confidence"] = str(payload.get("confidence") or "low")
    payload["title"] = str(payload.get("title") or "").strip()
    payload["finding"] = str(payload.get("finding") or "").strip()
    payload["impact"] = str(payload.get("impact") or "").strip()
    mitigation = payload.get("mitigation")
    payload["mitigation"] = str(mitigation).strip() if mitigation is not None else None
    payload["applies_to"] = _unique([str(value) for value in payload.get("applies_to") or []])
    payload["evidence_refs"] = _unique([str(value) for value in payload.get("evidence_refs") or []])
    payload["related_actions"] = _unique([str(value) for value in payload.get("related_actions") or []])
    payload["limitations"] = _unique([str(value) for value in payload.get("limitations") or []])
    payload["risk_intent"] = _risk_intent(payload)
    return payload


def _risk_intent(risk: dict[str, Any]) -> str:
    explicit = str(risk.get("risk_intent") or "").strip()
    if explicit:
        return explicit
    title = _normalize_text(str(risk.get("title") or ""))
    risk_type = str(risk.get("risk_type") or "")
    refs = " ".join(str(ref) for ref in risk.get("evidence_refs") or []).lower()
    text = f"{title} {refs}"

    if "primary id" in text:
        return "primary_id_feature_leakage"
    if "sample submission" in text:
        return "sample_submission_feature_leakage"
    if "naive target encoding" in text:
        return "naive_target_encoding_leakage"
    if "out of fold target encoding" in text or "oof target encoding" in text:
        return "oof_target_encoding_policy"
    if "safe feature drift" in text or "feature drift severity" in text or (
        risk_type == "drift" and "train test" in text and "drift" in text
    ):
        return "safe_feature_drift"
    if "id artifact drift" in text or "index drift artifact" in text:
        return "id_artifact_drift"
    if "threshold" in text and risk_type == "metric":
        return "threshold_metric_risk"
    if "target distribution" in text or "preserve target distribution" in text:
        return "target_distribution_validation"
    if "rare classes" in text or "fold class count" in text:
        return "target_fold_class_count"
    if "target imbalance" in text or "minority errors" in text:
        return "target_imbalance_metric"
    if "high cardinality" in text and "target association" in text:
        return "high_cardinality_target_association_reliability"
    if "high cardinality" in text and (
        "encoding" in text
        or "rare handling" in text
        or "robust" in text
        or "categorical" in text
        or "caution" in text
        or "contribution" in text
    ):
        return "high_cardinality_encoding"
    if "missingness" in text and ("informative" in text or "target associated" in text):
        return "missingness_informative"
    if "missingness" in text and ("shift" in text or "differs" in text):
        return "missingness_shift"
    if "completed baseline" in text or "sanity floor" in text:
        return "baseline_completed"
    if "no completed baseline" in text or "baseline evidence is available" in text:
        return "baseline_skipped"
    return f"{risk_type}:{title}"


def _title_score(risk: dict[str, Any]) -> tuple[int, int, int]:
    title = str(risk.get("title") or "")
    normalized = _normalize_text(title)
    general_tokens = sum(
        token in normalized
        for token in ("features", "columns", "validation", "metric", "baseline", "drift")
    )
    column_name_penalty = 1 if "`" in title or "[" in title else 0
    length = len(title)
    return (general_tokens, -column_name_penalty, -abs(length - 60))


def _mitigation_score(value: Any) -> tuple[int, int]:
    text = str(value or "")
    normalized = _normalize_text(text)
    action_tokens = sum(
        token in normalized
        for token in ("use", "exclude", "tune", "track", "check", "run", "record", "evaluate", "validate")
    )
    return (action_tokens, len(text))


def _text_score(value: Any) -> tuple[int, int]:
    text = str(value or "")
    return (1 if text else 0, len(text))


def _target_missingness_signal(target: dict[str, Any]) -> bool:
    return any(float(_as_dict(item).get("absolute_difference") or 0.0) >= 0.2 for item in target.get("target_by_missingness") or [])


def _driver_names_suggest_sparse_text(drivers: Any) -> bool:
    for driver in drivers or []:
        if isinstance(driver, str):
            name = driver.lower()
        else:
            name = str(_as_dict(driver).get("column") or _as_dict(driver).get("feature") or "").lower()
        if any(token in name for token in ("text", "code", "category", "cat", "description", "comment")):
            return True
    return False


def _metric_target_type(validation: dict[str, Any]) -> str:
    method = str(_as_dict(validation.get("primary_validation")).get("method") or "")
    return "multiclass" if "stratified" in method else ""


def _primary_evidence_category(refs: list[str]) -> str:
    if not refs:
        return "none"
    return str(refs[0]).split(".", 1)[0]


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = ["build_eda_risk_register", "deduplicate_eda_risks", "risk_summary", "validate_eda_risk_register"]
