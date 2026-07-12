from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.modules import baseline_runner as baseline
from kaggle_researcher.eda.schemas import (
    InferredSchema,
    LeakageCheckResult,
    MetricEvidence,
    ValidationEvidence,
)


ABLATION_PLANS = [
    ("abl_001_safe_numeric", "Safe numeric baseline", ["safe_numeric"], "reference_configuration"),
    (
        "abl_002_numeric_low_card_cat",
        "Numeric plus low-cardinality categoricals",
        ["safe_numeric", "low_cardinality_categorical"],
        "atomic_increment",
    ),
    (
        "abl_003_numeric_low_card_cat_missingness",
        "Numeric, low-cardinality categoricals, and missingness indicators",
        ["safe_numeric", "low_cardinality_categorical", "missingness_indicators"],
        "atomic_increment",
    ),
    (
        "abl_004_add_high_cardinality",
        "Add high-cardinality categoricals",
        [
            "safe_numeric",
            "low_cardinality_categorical",
            "missingness_indicators",
            "high_cardinality_categorical",
        ],
        "atomic_increment",
    ),
    (
        "abl_005_add_text_code_simple",
        "Add simple text/code features",
        [
            "safe_numeric",
            "low_cardinality_categorical",
            "missingness_indicators",
            "text_code_simple",
        ],
        "atomic_increment",
    ),
    ("abl_006_all_safe_features", "All safe features", ["all_safe_features"], "composite_configuration"),
]
BOUNDED_NEGLIGIBLE_DELTA = 0.002
BOUNDED_MATERIAL_DELTA = 0.005
SCALE_NEGLIGIBLE_RELATIVE_DELTA = 0.0025
SCALE_MATERIAL_RELATIVE_DELTA = 0.01
GENERIC_NEGLIGIBLE_DELTA = 0.001
GENERIC_MATERIAL_DELTA = 0.005


def run_baseline_ablations(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    metric_evidence: MetricEvidence,
    leakage_evidence: list[LeakageCheckResult],
    reader: DatasetReader,
    output_dir: Path,
    *,
    baseline_evidence: dict[str, Any] | None = None,
    feature_diagnostics: dict[str, Any] | None = None,
    max_rows: int = 100_000,
    max_ablations: int = 12,
    random_seed: int = 42,
    n_folds: int = 5,
    max_runtime_sec: int | None = None,
) -> dict[str, Any]:
    """Run lightweight fold-safe feature-block baseline ablations."""

    del output_dir, n_folds, max_runtime_sec
    warnings: list[str] = []
    limitations = [
        "Baseline ablations are lightweight sanity checks, not hyperparameter tuning.",
        f"Baseline ablations use at most {max_rows} train rows.",
    ]
    task_type = metric_evidence.task_type or "unknown"
    metric_name = metric_evidence.metric_name or "unknown"
    if task_type not in baseline.SUPPORTED_TASK_TYPES:
        return _skipped(metric_name, metric_evidence, validation_evidence, f"{task_type} ablations are not supported.")
    train_table = inferred_schema.train_base_table
    target_column = inferred_schema.target_column
    if train_table is None or target_column is None:
        return _skipped(metric_name, metric_evidence, validation_evidence, "Train base table and target column are required.")

    train_schema = _safe_schema(reader, train_table, warnings)
    if target_column not in {column["name"] for column in train_schema}:
        return _skipped(metric_name, metric_evidence, validation_evidence, "Target column is not present in train base.", warnings=warnings)
    feature_columns, excluded_columns, excluded_details = baseline._feature_columns(
        train_schema,
        inferred_schema=inferred_schema,
        validation_evidence=validation_evidence,
        leakage_evidence=leakage_evidence,
    )
    if not feature_columns:
        return _skipped(metric_name, metric_evidence, validation_evidence, "No safe feature columns are available after exclusions.", warnings=warnings)

    frame = _safe_read_frame(reader, train_table, [target_column, *feature_columns], max_rows, warnings, limitations)
    if frame is None or frame.height < 2:
        return _skipped(metric_name, metric_evidence, validation_evidence, "Not enough rows for baseline ablations.", warnings=warnings, limitations=limitations)
    try:
        splits = baseline._build_splits(
            frame,
            target_column=target_column,
            validation_evidence=validation_evidence,
            task_type=task_type,
            random_seed=random_seed,
            warnings=warnings,
        )
    except RuntimeError as exc:
        return _skipped(metric_name, metric_evidence, validation_evidence, str(exc), warnings=warnings, limitations=limitations)

    blocks = _feature_blocks(frame, feature_columns, feature_diagnostics or {})
    ablations: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None

    for ablation_id, name, required_blocks, ablation_kind in ABLATION_PLANS[:max_ablations]:
        missing_blocks = [block for block in required_blocks if not blocks.get(block)]
        if missing_blocks:
            ablation = _skipped_ablation(
                ablation_id,
                name,
                required_blocks,
                f"Required feature block(s) are empty: {', '.join(missing_blocks)}.",
                metric_name,
                excluded_columns,
                ablation_kind=ablation_kind,
            )
            ablations.append(ablation)
            continue
        ablation_frame, selected_columns, generated_columns = _frame_for_blocks(frame, blocks, required_blocks)
        if not selected_columns:
            ablation = _skipped_ablation(
                ablation_id,
                name,
                required_blocks,
                "No feature columns are available for this ablation.",
                metric_name,
                excluded_columns,
                ablation_kind=ablation_kind,
            )
            ablations.append(ablation)
            continue
        raw_columns = [column for column in selected_columns if column not in set(generated_columns)]
        complexity = _complexity_assessment(raw_columns, generated_columns, required_blocks)
        if completed and not _has_effective_feature_change(complexity, completed[-1].get("complexity_assessment", {})):
            ablations.append(
                _skipped_ablation(
                    ablation_id,
                    name,
                    required_blocks,
                    "no_effective_features",
                    metric_name,
                    excluded_columns,
                    ablation_kind=ablation_kind,
                    complexity_assessment=complexity,
                )
            )
            continue
        ablation = _run_one_ablation(
            ablation_frame,
            splits,
            target_column=target_column,
            feature_columns=selected_columns,
            generated_columns=generated_columns,
            excluded_columns=excluded_columns,
            ablation_id=ablation_id,
            name=name,
            feature_blocks=required_blocks,
            ablation_kind=ablation_kind,
            complexity_assessment=complexity,
            metric_evidence=metric_evidence,
            task_type=task_type,
            random_seed=random_seed,
        )
        if ablation["status"] == "completed":
            if reference is None:
                reference = ablation
            completed.append(ablation)
        ablations.append(ablation)

    if not completed:
        return {
            "status": "skipped",
            "reason": "No ablations could be completed.",
            "metric_name": metric_name,
            "greater_is_better": metric_evidence.greater_is_better,
            "validation_policy": baseline._validation_policy(validation_evidence),
            "fold_policy": _fold_policy(validation_evidence, splits, random_seed),
            "baseline_reference": {},
            "ablations": ablations,
            "best_ablation": {},
            "feature_block_findings": _not_testable_findings(ablations),
            "recommended_actions": [],
            "warnings": baseline._unique(warnings),
            "limitations": baseline._unique(limitations),
        }

    _add_comparisons(completed, metric_evidence)
    best = _best_ablation(completed, metric_evidence)
    best["is_best_overall"] = True
    findings = _feature_block_findings(ablations, metric_evidence)
    best_summary = _best_ablation_summary(best, completed, metric_evidence)
    complexity_tradeoffs = _complexity_tradeoffs(completed)
    return {
        "status": "completed",
        "metric_name": metric_name,
        "greater_is_better": metric_evidence.greater_is_better,
        "validation_policy": baseline._validation_policy(validation_evidence),
        "fold_policy": _fold_policy(validation_evidence, splits, random_seed),
        "baseline_reference": _ablation_reference(reference or completed[0]),
        "baseline_metric_value": (baseline_evidence or {}).get("metric_value"),
        "ablations": ablations,
        "best_ablation": best_summary,
        "feature_block_findings": findings,
        "complexity_tradeoffs": complexity_tradeoffs,
        "recommended_actions": _recommended_actions(findings),
        "warnings": baseline._unique(warnings),
        "limitations": baseline._unique(limitations),
    }


def _run_one_ablation(
    frame: pl.DataFrame,
    splits: list[tuple[list[int], list[int]]],
    *,
    target_column: str,
    feature_columns: list[str],
    generated_columns: list[str],
    excluded_columns: list[str],
    ablation_id: str,
    name: str,
    feature_blocks: list[str],
    ablation_kind: str,
    complexity_assessment: dict[str, Any],
    metric_evidence: MetricEvidence,
    task_type: str,
    random_seed: int,
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    metric_values: list[float] = []
    warnings: list[str] = []
    model_type = ""
    prediction_kind = baseline._prediction_kind(task_type, metric_evidence)
    for fold_idx, (train_indices, valid_indices) in enumerate(splits):
        train_fold = frame[train_indices]
        valid_fold = frame[valid_indices]
        try:
            model, model_type = baseline._fit_model(
                train_fold,
                feature_columns=feature_columns,
                target_column=target_column,
                task_type=task_type,
                random_seed=random_seed,
            )
            predictions = baseline._predict(model, valid_fold, feature_columns, prediction_kind)
            metric_value = baseline._compute_metric(
                metric_evidence.metric_name or "",
                task_type,
                valid_fold[target_column].to_list(),
                predictions,
            )
        except Exception as exc:
            warnings.append(f"Ablation fold {fold_idx} failed: {exc}")
            metric_value = None
        if metric_value is not None:
            metric_values.append(metric_value)
        fold_results.append(
            {
                "fold": fold_idx,
                "train_rows": len(train_indices),
                "valid_rows": len(valid_indices),
                "metric_value": metric_value,
            }
        )
    if not metric_values:
        return {
            "ablation_id": ablation_id,
            "name": name,
            "status": "failed",
            "feature_blocks": feature_blocks,
            "ablation_kind": ablation_kind,
            "feature_columns": feature_columns,
            "generated_feature_columns": generated_columns,
            "excluded_columns": excluded_columns,
            "metric_name": metric_evidence.metric_name,
            "fold_results": fold_results,
            "delta_direction": "positive_is_better",
            "complexity_assessment": complexity_assessment,
            "preprocessing_policy": _ablation_preprocessing_policy(feature_blocks),
            "reliability": "not_applicable",
            "warnings": warnings,
            "limitations": ["No folds produced a metric value."],
        }
    metric_value = round(sum(metric_values) / len(metric_values), 6)
    metric_std = _std(metric_values)
    return {
        "ablation_id": ablation_id,
        "name": name,
        "status": "completed",
        "model_type": model_type,
        "feature_blocks": feature_blocks,
        "ablation_kind": ablation_kind,
        "feature_columns": feature_columns,
        "generated_feature_columns": generated_columns,
        "excluded_columns": excluded_columns,
        "metric_name": metric_evidence.metric_name,
        "metric_value": metric_value,
        "metric_std": metric_std,
        "fold_results": fold_results,
        "delta_vs_reference": None,
        "delta_vs_previous": None,
        "delta_vs_best_prior": None,
        "comparison_reference_ids": {},
        "fold_comparison": {},
        "fold_delta_mean": None,
        "fold_delta_std": None,
        "fold_wins": 0,
        "fold_losses": 0,
        "fold_ties": 0,
        "materiality": "negligible",
        "stability": "insufficient_folds",
        "comparison_interpretation": "Comparison is pending paired fold analysis.",
        "delta_direction": "positive_is_better",
        "complexity_assessment": complexity_assessment,
        "preprocessing_policy": _ablation_preprocessing_policy(feature_blocks),
        "reliability": _reliability(metric_values, metric_std),
        "warnings": warnings + _block_warnings(feature_blocks),
        "limitations": [],
    }


def _feature_blocks(
    frame: pl.DataFrame,
    feature_columns: list[str],
    feature_diagnostics: dict[str, Any],
) -> dict[str, list[str]]:
    del feature_diagnostics
    numeric = [column for column in feature_columns if frame[column].dtype.is_numeric()]
    categorical = [column for column in feature_columns if column not in numeric]
    high_cardinality = baseline._high_cardinality_columns(frame, categorical)
    text_like = baseline._text_like_columns(frame, categorical)
    low_cardinality = [
        column
        for column in categorical
        if column not in set(high_cardinality)
        and column not in set(text_like)
        and frame[column].n_unique() <= 50
        and frame[column].n_unique() / max(frame.height, 1) <= 0.5
    ]
    missing_sources = [column for column in feature_columns if frame[column].null_count() > 0]
    return {
        "safe_numeric": numeric,
        "low_cardinality_categorical": low_cardinality,
        "missingness_indicators": missing_sources,
        "high_cardinality_categorical": high_cardinality,
        "text_code_simple": text_like,
        "all_safe_features": list(feature_columns),
    }


def _frame_for_blocks(
    frame: pl.DataFrame,
    blocks: dict[str, list[str]],
    selected_blocks: list[str],
) -> tuple[pl.DataFrame, list[str], list[str]]:
    if selected_blocks == ["all_safe_features"]:
        return frame, list(blocks["all_safe_features"]), []
    working = frame
    selected: list[str] = []
    generated: list[str] = []
    for block in selected_blocks:
        if block == "missingness_indicators":
            for column in blocks[block]:
                name = _generated_name("missing", column)
                if name not in working.columns:
                    working = working.with_columns(pl.col(column).is_null().cast(pl.Int8).alias(name))
                generated.append(name)
                selected.append(name)
        elif block == "text_code_simple":
            for column in blocks[block]:
                for name, values in _text_features(working[column], column).items():
                    if name not in working.columns:
                        working = working.with_columns(pl.Series(name, values))
                    generated.append(name)
                    selected.append(name)
        else:
            selected.extend(blocks[block])
    return working, baseline._unique(selected), baseline._unique(generated)


def _text_features(series: pl.Series, column: str) -> dict[str, list[int]]:
    values = series.to_list()
    prefix = _generated_name("text", column)
    text_values = ["" if value is None else str(value) for value in values]
    return {
        f"{prefix}_length": [len(value) for value in text_values],
        f"{prefix}_tokens": [len(value.split()) for value in text_values],
        f"{prefix}_digits": [sum(char.isdigit() for char in value) for value in text_values],
        f"{prefix}_punct": [sum(not char.isalnum() and not char.isspace() for char in value) for value in text_values],
        f"{prefix}_missing": [1 if value is None else 0 for value in values],
    }


def _feature_block_findings(
    ablations: list[dict[str, Any]],
    metric_evidence: MetricEvidence,
) -> list[dict[str, Any]]:
    targets = {
        "low_cardinality_categorical": "abl_002_numeric_low_card_cat",
        "missingness_indicators": "abl_003_numeric_low_card_cat_missingness",
        "high_cardinality_categorical": "abl_004_add_high_cardinality",
        "text_code_simple": "abl_005_add_text_code_simple",
    }
    results: list[dict[str, Any]] = []
    by_id = {item["ablation_id"]: item for item in ablations}
    for block, ablation_id in targets.items():
        ablation = by_id.get(ablation_id)
        if ablation is None or ablation.get("status") != "completed":
            results.append(_finding(block, "not_testable", None, "low", ablation_id, "Feature block was not testable in baseline ablations."))
            continue
        delta = ablation.get("delta_vs_best_prior")
        materiality = str(ablation.get("materiality") or "negligible")
        stability = str(ablation.get("stability") or "insufficient_folds")
        status = _finding_status(delta, materiality, stability)
        confidence = _finding_confidence(status, materiality, stability)
        recommendation = _finding_recommendation(block, status)
        results.append(_finding(
            block,
            status,
            delta,
            confidence,
            ablation_id,
            recommendation,
            ablation=ablation,
            finding_type="feature_block",
        ))
    composite = by_id.get("abl_006_all_safe_features")
    if composite is not None:
        results.append(_configuration_finding(composite))
    return results


def _finding(
    block: str,
    status: str,
    delta: float | None,
    confidence: str,
    ablation_id: str,
    recommendation: str,
    *,
    ablation: dict[str, Any] | None = None,
    finding_type: str = "feature_block",
) -> dict[str, Any]:
    ablation = ablation or {}
    return {
        "finding_type": finding_type,
        "feature_block": block,
        "status": status,
        "delta_metric": round(delta, 6) if isinstance(delta, (int, float)) else None,
        "delta_vs_reference": ablation.get("delta_vs_reference"),
        "delta_vs_previous": ablation.get("delta_vs_previous"),
        "delta_vs_best_prior": ablation.get("delta_vs_best_prior"),
        "fold_delta_mean": ablation.get("fold_delta_mean"),
        "fold_delta_std": ablation.get("fold_delta_std"),
        "fold_wins": ablation.get("fold_wins", 0),
        "fold_losses": ablation.get("fold_losses", 0),
        "fold_ties": ablation.get("fold_ties", 0),
        "materiality": ablation.get("materiality"),
        "stability": ablation.get("stability"),
        "confidence": confidence,
        "evidence_refs": [f"baseline_ablation_evidence.ablations.{ablation_id}"],
        "recommendation": recommendation,
    }


def _configuration_finding(ablation: dict[str, Any]) -> dict[str, Any]:
    if ablation.get("status") != "completed":
        return {
            "finding_type": "configuration",
            "configuration": "all_safe_features",
            "status": "not_testable",
            "recommendation": "Composite configuration was not testable in baseline ablations.",
            "evidence_refs": [f"baseline_ablation_evidence.ablations.{ablation.get('ablation_id') or 'abl_006_all_safe_features'}"],
        }
    materiality = str(ablation.get("materiality") or "negligible")
    stability = str(ablation.get("stability") or "insufficient_folds")
    delta = ablation.get("delta_vs_best_prior")
    if stability in {"mixed", "unstable"}:
        status = "unstable"
    elif isinstance(delta, (int, float)) and delta > 0 and materiality in {"material", "small"}:
        status = "best_overall" if _is_best(ablation) else "competitive"
    else:
        status = "not_better"
    return {
        "finding_type": "configuration",
        "configuration": "all_safe_features",
        "status": status,
        "delta_vs_reference": ablation.get("delta_vs_reference"),
        "delta_vs_best_prior": delta,
        "materiality_vs_best_prior": materiality,
        "stability_vs_best_prior": stability,
        "recommendation": _configuration_recommendation(materiality, stability),
        "evidence_refs": [f"baseline_ablation_evidence.ablations.{ablation.get('ablation_id')}"],
    }


def _recommended_actions(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in findings:
        if item.get("finding_type") == "configuration":
            if item.get("materiality_vs_best_prior") == "negligible":
                actions.append(_action("P1", "Prefer the simpler feature configuration until added complexity shows a material gain.", "The more complex configuration was only negligibly better than a prior configuration.", item.get("evidence_refs", []), "low", ["feature_engineering", "baseline"]))
            continue
        block = item.get("feature_block")
        status = item.get("status")
        ref = f"baseline_ablation_evidence.feature_block_findings.{block}"
        if status == "helped" and item.get("materiality") == "material" and item.get("stability") == "stable":
            actions.append(_action("P1", "Prioritize the feature block that produced a stable material validation improvement.", "Paired fold comparison showed a material and stable gain.", [ref], "low", ["feature_engineering", "baseline"]))
        elif status == "unstable" or item.get("materiality") == "small":
            actions.append(_action("P2", "Retest the feature block in a controlled experiment before adopting it by default.", "The aggregate improvement was small or inconsistent across folds.", [ref], "medium", ["feature_engineering", "baseline"]))
        elif block == "high_cardinality_categorical" and status in {"hurt", "neutral"}:
            actions.append(_action("P1", "Treat high-cardinality categorical features as controlled experiments, not default features.", "Baseline ablation did not show stable benefit from this block.", [ref], "medium", ["feature_engineering", "risk"]))
    return actions


def _action(priority: str, action: str, why: str, refs: list[str], risk: str, applies_to: list[str]) -> dict[str, Any]:
    return {
        "priority": priority,
        "action": action,
        "why": why,
        "evidence_refs": refs,
        "risk": risk,
        "applies_to": applies_to,
    }


def _delta(value: Any, reference: Any, metric_evidence: MetricEvidence) -> float | None:
    if value is None or reference is None:
        return None
    raw = float(value) - float(reference)
    if metric_evidence.greater_is_better is False:
        raw = -raw
    return round(raw, 6)


def _add_comparisons(completed: list[dict[str, Any]], metric_evidence: MetricEvidence) -> None:
    """Populate aggregate and paired-fold comparisons in declared ablation order."""

    reference = completed[0]
    best_prior: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None
    for index, ablation in enumerate(completed):
        if index == 0:
            ablation.update({
                "delta_vs_reference": 0.0,
                "delta_vs_previous": None,
                "delta_vs_best_prior": None,
                "comparison_reference_ids": {
                    "reference_ablation_id": ablation.get("ablation_id"),
                    "previous_ablation_id": None,
                    "best_prior_ablation_id": None,
                },
                "fold_comparison": {"paired_with_ablation_id": None, "fold_deltas": []},
                "comparison_interpretation": "Reference configuration for subsequent ablations.",
                "materiality": "negligible",
                "stability": "insufficient_folds",
            })
        else:
            best_prior = _best_ablation(completed[:index], metric_evidence)
            ablation["delta_vs_reference"] = _delta(ablation.get("metric_value"), reference.get("metric_value"), metric_evidence)
            ablation["delta_vs_previous"] = _delta(ablation.get("metric_value"), previous.get("metric_value"), metric_evidence) if previous else None
            ablation["delta_vs_best_prior"] = _delta(ablation.get("metric_value"), best_prior.get("metric_value"), metric_evidence)
            ablation["comparison_reference_ids"] = {
                "reference_ablation_id": reference.get("ablation_id"),
                "previous_ablation_id": previous.get("ablation_id") if previous else None,
                "best_prior_ablation_id": best_prior.get("ablation_id"),
            }
            comparison = _paired_fold_comparison(ablation, best_prior, metric_evidence)
            ablation["fold_comparison"] = comparison
            for field in ("fold_delta_mean", "fold_delta_std", "fold_wins", "fold_losses", "fold_ties"):
                ablation[field] = comparison.get(field)
            materiality = classify_ablation_materiality(
                metric_name=metric_evidence.metric_name,
                metric_family=metric_evidence.metric_family,
                greater_is_better=metric_evidence.greater_is_better is not False,
                delta=ablation["delta_vs_best_prior"] or 0.0,
                reference_metric=best_prior.get("metric_value"),
                fold_delta_std=comparison.get("fold_delta_std"),
            )
            stability = classify_ablation_stability(
                delta=ablation["delta_vs_best_prior"] or 0.0,
                fold_delta_mean=comparison.get("fold_delta_mean"),
                fold_delta_std=comparison.get("fold_delta_std"),
                fold_wins=int(comparison.get("fold_wins") or 0),
                fold_losses=int(comparison.get("fold_losses") or 0),
                fold_ties=int(comparison.get("fold_ties") or 0),
            )
            ablation["materiality"] = materiality
            ablation["stability"] = stability
            ablation["comparison_interpretation"] = build_ablation_interpretation(
                materiality=materiality,
                stability=stability,
                delta_vs_best_prior=ablation["delta_vs_best_prior"],
                fold_wins=int(comparison.get("fold_wins") or 0),
                fold_losses=int(comparison.get("fold_losses") or 0),
                fold_ties=int(comparison.get("fold_ties") or 0),
                effective_feature_change=bool(_as_dict(ablation.get("complexity_assessment")).get("effective_feature_change")),
            )
        previous = ablation


def _paired_fold_comparison(candidate: dict[str, Any], comparator: dict[str, Any], metric_evidence: MetricEvidence) -> dict[str, Any]:
    candidate_scores = {item.get("fold"): item.get("metric_value") for item in candidate.get("fold_results", [])}
    comparator_scores = {item.get("fold"): item.get("metric_value") for item in comparator.get("fold_results", [])}
    deltas = [
        _delta(candidate_scores[fold], comparator_scores[fold], metric_evidence)
        for fold in sorted(set(candidate_scores) & set(comparator_scores))
        if candidate_scores[fold] is not None and comparator_scores[fold] is not None
    ]
    values = [float(delta) for delta in deltas if delta is not None]
    epsilon = _tie_epsilon(metric_evidence, comparator.get("metric_value"))
    wins = sum(value > epsilon for value in values)
    losses = sum(value < -epsilon for value in values)
    ties = len(values) - wins - losses
    non_tied = wins + losses
    return {
        "paired_with_ablation_id": comparator.get("ablation_id"),
        "fold_deltas": [round(value, 6) for value in values],
        "fold_delta_mean": round(sum(values) / len(values), 6) if values else None,
        "fold_delta_std": _std(values) if values else None,
        "fold_delta_median": _median(values),
        "fold_wins": wins,
        "fold_losses": losses,
        "fold_ties": ties,
        "win_rate": round(wins / len(values), 6) if values else None,
        "sign_consistency": round(max(wins, losses) / non_tied, 6) if non_tied else 1.0,
        "tie_epsilon": epsilon,
    }


def _best_ablation(ablations: list[dict[str, Any]], metric_evidence: MetricEvidence) -> dict[str, Any]:
    reverse = metric_evidence.greater_is_better is not False
    return sorted(ablations, key=lambda item: float(item.get("metric_value") or 0.0), reverse=reverse)[0]


def classify_ablation_materiality(
    *,
    metric_name: str | None,
    metric_family: str | None,
    greater_is_better: bool,
    delta: float,
    reference_metric: float | None,
    fold_delta_std: float | None,
) -> str:
    """Classify an oriented delta using metric-aware, conservative thresholds."""

    del greater_is_better, fold_delta_std
    magnitude = abs(delta)
    name = (metric_name or "").lower()
    family = (metric_family or "").lower()
    if _is_scale_dependent_metric(name, family):
        denominator = abs(float(reference_metric)) if reference_metric not in (None, 0) else None
        relative = magnitude / denominator if denominator else magnitude
        negligible = SCALE_NEGLIGIBLE_RELATIVE_DELTA
        material = SCALE_MATERIAL_RELATIVE_DELTA
    elif _is_bounded_metric(name, family):
        relative = magnitude
        negligible = BOUNDED_NEGLIGIBLE_DELTA
        material = BOUNDED_MATERIAL_DELTA
    else:
        relative = magnitude
        negligible = GENERIC_NEGLIGIBLE_DELTA
        material = GENERIC_MATERIAL_DELTA
    if relative < negligible:
        return "negligible"
    if relative < material:
        return "small"
    return "material" if delta > 0 else "material_negative"


def classify_ablation_stability(
    *,
    delta: float,
    fold_delta_mean: float | None,
    fold_delta_std: float | None,
    fold_wins: int,
    fold_losses: int,
    fold_ties: int,
) -> str:
    total = fold_wins + fold_losses + fold_ties
    if total < 3:
        return "insufficient_folds"
    non_tied = fold_wins + fold_losses
    if non_tied == 0:
        return "stable"
    dominant = max(fold_wins, fold_losses) / non_tied
    mean = abs(float(fold_delta_mean or 0.0))
    std = float(fold_delta_std or 0.0)
    if dominant >= 0.7 and std <= mean and ((delta > 0 and fold_wins > fold_losses) or (delta < 0 and fold_losses > fold_wins)):
        return "stable"
    if std > mean or dominant < 0.6:
        return "unstable"
    return "mixed"


def build_ablation_interpretation(
    *,
    materiality: str,
    stability: str,
    delta_vs_best_prior: float | None,
    fold_wins: int,
    fold_losses: int,
    fold_ties: int,
    effective_feature_change: bool,
) -> str:
    if not effective_feature_change:
        return "No effective feature change was introduced."
    if materiality == "material_negative" and stability == "stable":
        return "Material degradation across most folds."
    if materiality == "material" and stability == "stable":
        return "Material and stable improvement over the best prior configuration."
    if materiality == "negligible":
        return "Aggregate improvement is within ordinary CV variation."
    if stability in {"mixed", "unstable"}:
        return "Small improvement, but fold-level evidence is mixed." if (delta_vs_best_prior or 0) >= 0 else "Aggregate degradation is inconsistent across folds."
    return f"Paired folds recorded {fold_wins} wins, {fold_losses} losses, and {fold_ties} ties versus the best prior configuration."


def _best_ablation_summary(best: dict[str, Any], completed: list[dict[str, Any]], metric_evidence: MetricEvidence) -> dict[str, Any]:
    summary = _ablation_reference(best)
    summary.update({
        "delta_vs_reference": best.get("delta_vs_reference"),
        "delta_vs_best_prior": best.get("delta_vs_best_prior"),
        "materiality_vs_best_prior": best.get("materiality"),
        "stability_vs_best_prior": best.get("stability"),
        "is_materially_better_than_simpler_prior": True,
        "simpler_competitive_ablation_id": None,
    })
    best_complexity = _as_dict(best.get("complexity_assessment"))
    simpler = [
        item for item in completed
        if item.get("ablation_id") != best.get("ablation_id")
        and int(_as_dict(item.get("complexity_assessment")).get("n_effective_features") or 0) < int(best_complexity.get("n_effective_features") or 0)
    ]
    if not simpler:
        return summary
    competitive = _best_ablation(simpler, metric_evidence)
    comparison = _paired_fold_comparison(best, competitive, metric_evidence)
    delta = _delta(best.get("metric_value"), competitive.get("metric_value"), metric_evidence)
    materiality = classify_ablation_materiality(
        metric_name=metric_evidence.metric_name,
        metric_family=metric_evidence.metric_family,
        greater_is_better=metric_evidence.greater_is_better is not False,
        delta=delta or 0.0,
        reference_metric=competitive.get("metric_value"),
        fold_delta_std=comparison.get("fold_delta_std"),
    )
    if materiality in {"negligible", "small"}:
        summary.update({
            "simpler_competitive_ablation_id": competitive.get("ablation_id"),
            "simpler_competitive_metric_value": competitive.get("metric_value"),
            "delta_vs_simpler_competitive": delta,
            "materiality_vs_simpler_competitive": materiality,
            "is_materially_better_than_simpler_prior": False,
        })
    return summary


def _complexity_tradeoffs(completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item.get("ablation_id"): item for item in completed}
    tradeoffs: list[dict[str, Any]] = []
    for item in completed[1:]:
        previous_id = _as_dict(item.get("comparison_reference_ids")).get("previous_ablation_id")
        previous = by_id.get(previous_id)
        if previous is None:
            continue
        current_complexity = _as_dict(item.get("complexity_assessment"))
        prior_complexity = _as_dict(previous.get("complexity_assessment"))
        added = max(0, int(current_complexity.get("n_effective_features") or 0) - int(prior_complexity.get("n_effective_features") or 0))
        materiality = str(item.get("materiality") or "negligible")
        tradeoff = {
            "simpler_ablation_id": previous_id,
            "candidate_ablation_id": item.get("ablation_id"),
            "added_feature_count": added,
            "metric_gain": item.get("delta_vs_previous"),
            "recommendation": "adopt_candidate" if materiality == "material" and item.get("stability") == "stable" else "test_further" if materiality == "small" else "prefer_simpler",
        }
        item["complexity_tradeoff"] = tradeoff
        tradeoffs.append(tradeoff)
    return tradeoffs


def _complexity_assessment(raw_columns: list[str], generated_columns: list[str], feature_blocks: list[str]) -> dict[str, Any]:
    return {
        "raw_feature_columns": list(raw_columns),
        "generated_feature_columns": list(generated_columns),
        "n_raw_features": len(raw_columns),
        "n_generated_features": len(generated_columns),
        "n_effective_features": len(raw_columns) + len(generated_columns),
        "added_raw_columns": list(raw_columns),
        "added_generated_columns": list(generated_columns),
        "removed_columns": [],
        "added_feature_blocks": list(feature_blocks),
        "effective_feature_change": True,
        "limitation": "Feature counts describe raw model inputs; transformed pipeline feature counts are not exposed.",
    }


def _has_effective_feature_change(candidate: dict[str, Any], previous: dict[str, Any]) -> bool:
    candidate_features = set(candidate.get("raw_feature_columns", [])) | set(candidate.get("generated_feature_columns", []))
    previous_features = set(previous.get("raw_feature_columns", [])) | set(previous.get("generated_feature_columns", []))
    candidate["added_raw_columns"] = sorted(set(candidate.get("raw_feature_columns", [])) - set(previous.get("raw_feature_columns", [])))
    candidate["added_generated_columns"] = sorted(set(candidate.get("generated_feature_columns", [])) - set(previous.get("generated_feature_columns", [])))
    candidate["removed_columns"] = sorted(previous_features - candidate_features)
    candidate["effective_feature_change"] = candidate_features != previous_features
    return bool(candidate["effective_feature_change"])


def _finding_status(delta: Any, materiality: str, stability: str) -> str:
    if delta is None:
        return "not_testable"
    if materiality == "negligible":
        return "neutral"
    if stability in {"mixed", "unstable", "insufficient_folds"}:
        return "unstable"
    if materiality == "material_negative":
        return "hurt"
    return "helped" if float(delta) > 0 else "neutral"


def _finding_confidence(status: str, materiality: str, stability: str) -> str:
    if status == "helped" and materiality == "material" and stability == "stable":
        return "high"
    if status in {"helped", "hurt"} and stability == "stable":
        return "medium"
    return "low"


def _configuration_recommendation(materiality: str, stability: str) -> str:
    if materiality == "negligible":
        return "Prefer the simpler competitive configuration initially."
    if materiality == "small" or stability in {"mixed", "unstable"}:
        return "Retest this composite configuration before adopting its added complexity."
    return "Prioritize this configuration only when its marginal gain remains stable."


def _is_best(ablation: dict[str, Any]) -> bool:
    return bool(ablation.get("is_best_overall"))


def _is_bounded_metric(name: str, family: str) -> bool:
    return any(token in name for token in ("accuracy", "auc", "f1", "precision", "recall", "average_precision", "ndcg", "map")) or "classification" in family or "ranking" in family


def _is_scale_dependent_metric(name: str, family: str) -> bool:
    return any(token in name for token in ("rmse", "mae", "mse", "rmsle", "mape")) or "regression" in family


def _tie_epsilon(metric_evidence: MetricEvidence, reference_metric: Any) -> float:
    name = (metric_evidence.metric_name or "").lower()
    family = (metric_evidence.metric_family or "").lower()
    if _is_bounded_metric(name, family):
        return 0.0005
    if _is_scale_dependent_metric(name, family) and reference_metric not in (None, 0):
        return max(abs(float(reference_metric)) * 0.0005, 1e-9)
    return 1e-6


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    result = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return round(result, 6)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ablation_reference(ablation: dict[str, Any]) -> dict[str, Any]:
    return {
        "ablation_id": ablation.get("ablation_id"),
        "name": ablation.get("name"),
        "feature_blocks": ablation.get("feature_blocks", []),
        "metric_value": ablation.get("metric_value"),
    }


def _skipped_ablation(
    ablation_id: str,
    name: str,
    feature_blocks: list[str],
    reason: str,
    metric_name: str,
    excluded_columns: list[str],
    *,
    ablation_kind: str = "atomic_increment",
    complexity_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ablation_id": ablation_id,
        "name": name,
        "status": "skipped",
        "reason": reason,
        "feature_blocks": feature_blocks,
        "ablation_kind": ablation_kind,
        "feature_columns": [],
        "generated_feature_columns": [],
        "excluded_columns": excluded_columns,
        "metric_name": metric_name,
        "fold_results": [],
        "delta_direction": "positive_is_better",
        "delta_vs_reference": None,
        "delta_vs_previous": None,
        "delta_vs_best_prior": None,
        "complexity_assessment": complexity_assessment or {
            "raw_feature_columns": [],
            "generated_feature_columns": [],
            "n_raw_features": 0,
            "n_generated_features": 0,
            "n_effective_features": 0,
            "added_raw_columns": [],
            "added_generated_columns": [],
            "removed_columns": [],
            "added_feature_blocks": list(feature_blocks),
            "effective_feature_change": False,
        },
        "preprocessing_policy": _ablation_preprocessing_policy(feature_blocks),
        "reliability": "not_applicable",
        "warnings": [],
        "limitations": [reason],
    }


def _ablation_preprocessing_policy(feature_blocks: list[str]) -> dict[str, Any]:
    return {
        "fit_scope": "inside_cv_folds",
        "leakage_safe": True,
        "target_used_in_preprocessing": False,
        "uses_target_encoding": False,
        "uses_test_labels": False,
        "feature_blocks": list(feature_blocks),
    }


def _fold_policy(
    validation_evidence: ValidationEvidence,
    splits: list[tuple[list[int], list[int]]],
    random_seed: int,
) -> dict[str, Any]:
    policy = baseline._validation_policy(validation_evidence)
    return {
        **policy,
        "n_folds": len(splits),
        "same_folds_across_ablations": True,
        "random_state": random_seed,
        "fold_row_counts": [
            {"fold": index, "train_rows": len(train_idx), "valid_rows": len(valid_idx)}
            for index, (train_idx, valid_idx) in enumerate(splits)
        ],
    }


def _not_testable_findings(ablations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _finding(block, "not_testable", 0.0, "low", ablation.get("ablation_id", ""), "Feature block was not testable in baseline ablations.")
        for ablation in ablations
        for block in ablation.get("feature_blocks", [])
        if block not in {"safe_numeric", "all_safe_features"}
    ]


def _finding_recommendation(block: str, status: str) -> str:
    if status == "helped":
        return "Prioritize this block in early experiments."
    if status == "hurt":
        return "Defer this block unless a stronger fold-safe treatment is tested."
    if status == "unstable":
        return "Evaluate this block in isolated fold-safe experiments before default use."
    if status == "neutral":
        return "Treat this block as optional until stronger validation evidence appears."
    return "Feature block was not testable in baseline ablations."


def _reliability(metric_values: list[float], metric_std: float) -> str:
    if len(metric_values) < 2:
        return "caution_small_data"
    if metric_std >= 0.05:
        return "caution_high_variance"
    return "reliable"


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return round(sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 6)


def _block_warnings(feature_blocks: list[str]) -> list[str]:
    if "high_cardinality_categorical" in feature_blocks:
        return ["High-cardinality categorical ablation is included with caution and uses no target encoding."]
    return []


def _generated_name(kind: str, column: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in column).strip("_").lower()
    return f"__{kind}_{safe}"


def _safe_schema(reader: DatasetReader, table: str, warnings: list[str]) -> list[dict[str, str]]:
    try:
        return reader.read_schema(table)
    except ReaderError as exc:
        warnings.append(str(exc))
        return []


def _safe_read_frame(
    reader: DatasetReader,
    table: str,
    columns: list[str],
    max_rows: int,
    warnings: list[str],
    limitations: list[str],
) -> pl.DataFrame | None:
    try:
        return reader.read_columns(table, columns=baseline._unique(columns), n_rows=max_rows)
    except ReaderError as exc:
        warnings.append(str(exc))
        limitations.append(f"Could not read ablation columns from {table}.")
        return None


def _skipped(
    metric_name: str,
    metric_evidence: MetricEvidence,
    validation_evidence: ValidationEvidence,
    reason: str,
    *,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "metric_name": metric_name,
        "greater_is_better": metric_evidence.greater_is_better,
        "validation_policy": baseline._validation_policy(validation_evidence),
        "baseline_reference": {},
        "ablations": [],
        "best_ablation": {},
        "feature_block_findings": [],
        "recommended_actions": [],
        "warnings": baseline._unique(warnings or []),
        "limitations": baseline._unique([*(limitations or []), reason]),
    }


__all__ = ["run_baseline_ablations"]
