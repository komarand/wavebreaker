from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.modules.column_policy import ColumnRolePolicy
from kaggle_researcher.eda.schemas import InferredSchema, MetricEvidence, TableProfile


DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_NUMERIC_COLUMNS = 30
DEFAULT_MAX_CATEGORICAL_COLUMNS = 20
DEFAULT_MAX_PAIR_CANDIDATES = 300
DEFAULT_MAX_REPORTED_PER_TYPE = 20
DEFAULT_MIN_GROUP_ROWS = 20
MATERIAL_SCORE = 0.05
SMALL_SCORE = 0.01
HIGH_REDUNDANCY = 0.95


def diagnose_interactions(
    inferred_schema: InferredSchema,
    table_profiles: list[TableProfile],
    metric_evidence: MetricEvidence | dict[str, Any],
    reader: DatasetReader,
    *,
    feature_diagnostics: dict[str, Any] | None = None,
    target_diagnostics: dict[str, Any] | None = None,
    drift_evidence: dict[str, Any] | None = None,
    baseline_ablation_evidence: dict[str, Any] | None = None,
    leakage_evidence: list[dict[str, Any]] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_numeric_columns: int = DEFAULT_MAX_NUMERIC_COLUMNS,
    max_categorical_columns: int = DEFAULT_MAX_CATEGORICAL_COLUMNS,
    max_pair_candidates: int = DEFAULT_MAX_PAIR_CANDIDATES,
    max_reported_interactions_per_type: int = DEFAULT_MAX_REPORTED_PER_TYPE,
    min_group_rows: int = DEFAULT_MIN_GROUP_ROWS,
    random_state: int = 42,
) -> dict[str, Any]:
    """Produce compact, role-safe pairwise interaction hypotheses.

    Scores are statistical proxies, not accepted feature engineering decisions.  The
    module deliberately caps inputs and reports only experiment-ready evidence.
    """

    del target_diagnostics, random_state
    metric = _as_dict(metric_evidence)
    base_profile = _base_profile(inferred_schema, table_profiles)
    if base_profile is None or not inferred_schema.train_base_table:
        return _not_testable("Train base profile is unavailable.")
    policy = ColumnRolePolicy(inferred_schema, metric_evidence)
    train_table = inferred_schema.train_base_table
    profiles = {item.name: item for item in base_profile.columns}
    safe, excluded = _safe_candidates(
        list(profiles), profiles, policy, train_table, leakage_evidence or [],
    )
    numeric, categorical, selection = _select_candidates(
        safe,
        profiles,
        feature_diagnostics or {},
        drift_evidence or {},
        baseline_ablation_evidence or {},
        max_numeric_columns=max_numeric_columns,
        max_categorical_columns=max_categorical_columns,
    )
    selected = [*numeric, *categorical]
    target = inferred_schema.target_column if inferred_schema.target_column in profiles else None
    frame = _read(reader, train_table, [*selected, *( [target] if target else [])], max_rows)
    if frame is None or frame.height < 3:
        return _not_testable("Not enough readable train rows for interaction diagnostics.", excluded=excluded)
    test_frame = _read(reader, inferred_schema.test_base_table, selected, max_rows) if inferred_schema.test_base_table else None
    target_values = _target_values(frame, target)
    numeric_numeric = _numeric_numeric(frame, numeric, target_values, max_pair_candidates, min_group_rows)
    numeric_categorical = _numeric_categorical(frame, numeric, categorical, target_values, max_pair_candidates, min_group_rows)
    categorical_categorical = _categorical_categorical(frame, test_frame, categorical, target_values, max_pair_candidates, min_group_rows)
    missingness = _missingness_interactions(frame, selected, target_values, min_group_rows)
    redundancy = _redundancy_groups(numeric_numeric)
    _apply_drift_caution([*numeric_numeric, *numeric_categorical, *categorical_categorical], drift_evidence or {})
    numeric_numeric = _reportable(numeric_numeric, max_reported_interactions_per_type)
    numeric_categorical = _reportable(numeric_categorical, max_reported_interactions_per_type)
    categorical_categorical = _reportable(categorical_categorical, max_reported_interactions_per_type)
    missingness = _reportable(missingness, max_reported_interactions_per_type)
    hypotheses, experiments = _hypotheses_and_experiments(
        numeric_numeric, numeric_categorical, categorical_categorical, missingness, redundancy,
    )
    warnings: list[str] = []
    if target is None:
        warnings.append("Target-conditioned interaction scores were not testable because no target column is available.")
    return {
        "status": "completed",
        "task_type": metric.get("task_type") or "unknown",
        "target_column": target,
        "candidate_columns": selected,
        "excluded_columns": excluded,
        "candidate_selection": selection,
        "numeric_numeric": numeric_numeric,
        "numeric_categorical": numeric_categorical,
        "categorical_categorical": categorical_categorical,
        "missingness_interactions": missingness,
        "redundancy_groups": redundancy,
        "interaction_hypotheses": hypotheses,
        "recommended_experiments": experiments,
        "interaction_score_policy": {
            "method": "statistical_proxy",
            "metric_name": metric.get("metric_name"),
            "same_folds": False,
            "positive_means_better": True,
            "limitations": ["Scores are bounded statistical screening proxies; validate every hypothesis with the primary fold policy."],
        },
        "warnings": warnings,
        "limitations": [
            "Only capped candidate pairs are evaluated; text-like and role columns are excluded.",
            "No crossed, ratio, or encoded features are created automatically.",
        ],
    }


def _safe_candidates(
    columns: list[str], profiles: dict[str, Any], policy: ColumnRolePolicy, table: str, leakage: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    excluded = [
        item for item in policy.excluded_columns(columns, table=table, context="model_feature")
        # Schema role inference can mark binary input features as target-like; the
        # declared target remains the only target input exclusion here.
        if not (item.get("reason") == "target_column" and item.get("column") != policy.target_column)
    ]
    excluded_names = {item["column"] for item in excluded}
    unsafe = _leakage_columns(leakage)
    for column in columns:
        profile = profiles[column]
        reason = None
        if column in unsafe:
            reason = "leakage_risk"
        elif bool(getattr(profile, "is_constant", False)):
            reason = "constant"
        elif int(getattr(profile, "missing_count", 0) or 0) and float(getattr(profile, "missing_pct", 0) or 0) >= 1:
            reason = "all_missing"
        elif not (_is_numeric(profile) or _is_categorical(profile)):
            reason = "unsupported"
        if reason and column not in excluded_names:
            excluded.append({"column": column, "reason": reason})
            excluded_names.add(column)
    return [column for column in columns if column not in excluded_names], sorted(excluded, key=lambda item: (item["column"], item["reason"]))


def _select_candidates(
    safe: list[str], profiles: dict[str, Any], feature_diagnostics: dict[str, Any], drift: dict[str, Any], ablations: dict[str, Any], *, max_numeric_columns: int, max_categorical_columns: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    del ablations
    numeric = [column for column in safe if _is_numeric(profiles[column])]
    categorical = [column for column in safe if _is_categorical(profiles[column])]
    association = _association_scores(feature_diagnostics)
    shifted = _shifted_columns(drift)
    def score(column: str) -> tuple[float, str]:
        profile = profiles[column]
        missing = float(getattr(profile, "missing_pct", 0) or 0)
        unique = float(getattr(profile, "unique_ratio", 0) or 0)
        value = association.get(column, 0.0) + (0.1 if 0 < missing <= 0.5 else 0.0) - unique * 0.2 - (0.2 if column in shifted else 0.0)
        return value, "target_association_or_safe_variance"
    numeric_ranked = sorted(numeric, key=lambda column: (-score(column)[0], column))
    categorical_ranked = sorted(categorical, key=lambda column: (-score(column)[0], float(getattr(profiles[column], "unique_ratio", 0) or 0), column))
    chosen_numeric, chosen_categorical = numeric_ranked[:max_numeric_columns], categorical_ranked[:max_categorical_columns]
    selected = set(chosen_numeric) | set(chosen_categorical)
    missingness = [column for column in safe if float(getattr(profiles[column], "missing_pct", 0) or 0) > 0 and column in selected]
    return chosen_numeric, chosen_categorical, {
        "numeric_columns": chosen_numeric,
        "categorical_columns": chosen_categorical,
        "missingness_columns": missingness,
        "selection_reasons": {column: score(column)[1] for column in [*chosen_numeric, *chosen_categorical]},
        "excluded_due_to_caps": [column for column in [*numeric_ranked, *categorical_ranked] if column not in selected],
    }


def _numeric_numeric(frame: pl.DataFrame, columns: list[str], target: list[float] | None, cap: int, min_group_rows: int) -> list[dict[str, Any]]:
    rows = []
    for left, right in list(combinations(columns, 2))[:cap]:
        pairs = [(float(a), float(b), target[index] if target else None) for index, (a, b) in enumerate(zip(frame[left].to_list(), frame[right].to_list())) if a is not None and b is not None and (target is None or target[index] is not None)]
        if len(pairs) < min_group_rows:
            continue
        xs, ys = [item[0] for item in pairs], [item[1] for item in pairs]
        pearson = _correlation(xs, ys)
        spearman = _correlation(_ranks(xs), _ranks(ys))
        target_score = 0.0
        if target is not None:
            values, target_values = [a * b for a, b, _ in pairs], [float(c) for _, _, c in pairs]
            target_score = max(0.0, abs(_correlation(values, target_values)) - max(abs(_correlation(xs, target_values)), abs(_correlation(ys, target_values))))
        absolute = max(abs(pearson), abs(spearman))
        redundancy = "high" if absolute >= HIGH_REDUNDANCY else "medium" if absolute >= 0.8 else "low" if absolute >= 0.6 else "none"
        transforms = [] if redundancy == "high" else _numeric_transforms(xs, ys)
        rows.append({
            "left_column": left, "right_column": right, "pearson": round(pearson, 6), "spearman": round(spearman, 6), "absolute_correlation": round(absolute, 6),
            "redundancy_severity": redundancy, "target_interaction_score": round(target_score, 6), "interaction_reliability": "reliable" if len(pairs) >= min_group_rows * 2 else "caution_small_sample",
            "materiality": _materiality(target_score), "suggested_transforms": transforms,
            "evidence_refs": ["interaction_diagnostics.numeric_numeric"], "warnings": [],
        })
    return rows


def _numeric_categorical(frame: pl.DataFrame, numeric: list[str], categorical: list[str], target: list[float] | None, cap: int, min_group_rows: int) -> list[dict[str, Any]]:
    rows = []
    for number, category in list(((n, c) for n in numeric for c in categorical))[:cap]:
        groups: dict[str, list[tuple[float, float | None]]] = {}
        for index, (raw_number, raw_category) in enumerate(zip(frame[number].to_list(), frame[category].to_list())):
            if raw_number is None or raw_category is None:
                continue
            groups.setdefault(str(raw_category), []).append((float(raw_number), target[index] if target else None))
        eligible = {key: values for key, values in groups.items() if len(values) >= min_group_rows}
        if len(eligible) < 2:
            continue
        target_means = [_mean([float(item[1]) for item in values if item[1] is not None]) for values in eligible.values()] if target else []
        score = _normalized_range(target_means) if target_means else 0.0
        sparse = len(eligible) < len(groups) or len(groups) > 20
        rows.append({
            "numeric_column": number, "categorical_column": category, "n_categories_analyzed": len(eligible), "minimum_group_rows": min_group_rows,
            "interaction_score": round(score, 6), "target_behavior_varies_by_category": score >= SMALL_SCORE,
            "reliability": "caution_sparse_categories" if sparse else "reliable", "materiality": _materiality(score),
            "finding": "Numeric target behavior varies across category groups." if score >= SMALL_SCORE else "No material category-specific numeric behavior was detected.",
            "suggested_experiment": "Test category-specific numeric bins or a tree interaction." if score >= SMALL_SCORE else None,
            "evidence_refs": ["interaction_diagnostics.numeric_categorical"],
        })
    return rows


def _categorical_categorical(frame: pl.DataFrame, test: pl.DataFrame | None, columns: list[str], target: list[float] | None, cap: int, min_group_rows: int) -> list[dict[str, Any]]:
    rows = []
    for left, right in list(combinations(columns, 2))[:cap]:
        combinations_train = [(str(a), str(b)) for a, b in zip(frame[left].to_list(), frame[right].to_list()) if a is not None and b is not None]
        counts = Counter(combinations_train)
        if not counts:
            continue
        rare_rate = sum(value for value in counts.values() if value < min_group_rows) / len(combinations_train)
        association = _cramers_v(frame[left].to_list(), frame[right].to_list())
        score = 0.0
        if target:
            target_by_combo: dict[tuple[str, str], list[float]] = {}
            for a, b, value in zip(frame[left].to_list(), frame[right].to_list(), target):
                if a is not None and b is not None and value is not None:
                    target_by_combo.setdefault((str(a), str(b)), []).append(float(value))
            means = [_mean(values) for pair, values in target_by_combo.items() if counts[pair] >= min_group_rows]
            score = _normalized_range(means)
        unseen = _unseen_combination_rate(test, left, right, set(counts))
        reliability = "reliable"
        if rare_rate > 0.2:
            reliability = "caution_sparse_combinations"
        elif unseen > 0.05:
            reliability = "caution_test_mismatch"
        rows.append({
            "left_column": left, "right_column": right, "combination_count": len(counts), "rare_combination_rate": round(rare_rate, 6),
            "association_score": round(association, 6), "target_interaction_score": round(score, 6), "unseen_test_combination_rate": round(unseen, 6),
            "reliability": reliability, "materiality": _materiality(score),
            "finding": "Category combination behavior warrants a controlled crossed-feature experiment." if score >= SMALL_SCORE and reliability == "reliable" else "Combination evidence is sparse or not material.",
            "suggested_experiment": "Test a crossed categorical feature with rare-combination handling." if score >= SMALL_SCORE and reliability == "reliable" else None,
            "evidence_refs": ["interaction_diagnostics.categorical_categorical"],
        })
    return rows


def _missingness_interactions(frame: pl.DataFrame, columns: list[str], target: list[float] | None, min_group_rows: int) -> list[dict[str, Any]]:
    if target is None:
        return []
    rows = []
    missing_columns = [column for column in columns if any(value is None for value in frame[column].to_list())]
    for column in missing_columns:
        missing_target = [target[index] for index, value in enumerate(frame[column].to_list()) if value is None and target[index] is not None]
        present_target = [target[index] for index, value in enumerate(frame[column].to_list()) if value is not None and target[index] is not None]
        if len(missing_target) < min_group_rows or len(present_target) < min_group_rows:
            continue
        difference = abs(_mean(missing_target) - _mean(present_target)) / max(_std(target), 1.0)
        rows.append({
            "interaction_type": "missingness_target", "columns": [column], "interaction_score": round(difference, 6), "target_rate_difference": round(_mean(missing_target) - _mean(present_target), 6),
            "train_test_shift": 0.0, "reliability": "reliable", "materiality": _materiality(difference),
            "finding": "Missingness has a material target association." if difference >= SMALL_SCORE else "Missingness target association is negligible.",
            "suggested_experiment": "Test a fold-safe missingness indicator in isolation." if difference >= SMALL_SCORE else None,
            "evidence_refs": ["interaction_diagnostics.missingness_interactions"],
        })
    return rows


def _redundancy_groups(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for index, item in enumerate(sorted((pair for pair in pairs if pair["redundancy_severity"] in {"medium", "high"}), key=lambda pair: (-pair["absolute_correlation"], pair["left_column"], pair["right_column"])), 1):
        groups.append({"group_id": f"redundancy_{index:03d}", "columns": [item["left_column"], item["right_column"]], "reason": "high_numeric_correlation", "strength": item["redundancy_severity"], "recommended_action": "Keep one representative or use regularized/tree models; validate removal.", "evidence_refs": ["interaction_diagnostics.numeric_numeric"]})
    return groups[:DEFAULT_MAX_REPORTED_PER_TYPE]


def _hypotheses_and_experiments(*collections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hypotheses, experiments = [], []
    candidates = [item for collection in collections[:-1] for item in collection if item.get("materiality") in {"material", "small"} and item.get("reliability", item.get("interaction_reliability")) == "reliable"]
    for index, item in enumerate(candidates[:10], 1):
        columns = item.get("columns") or [item.get("left_column") or item.get("numeric_column"), item.get("right_column") or item.get("categorical_column")]
        columns = [column for column in columns if column]
        kind = "missingness" if item.get("interaction_type") else "numeric_numeric" if "pearson" in item else "numeric_categorical" if "numeric_column" in item else "categorical_categorical"
        interaction_id, experiment_id = f"interaction_{index:03d}", f"exp_interaction_{index:03d}"
        hypothesis = f"Test whether the interaction between {' and '.join(columns)} may improve validation beyond individual feature effects."
        hypotheses.append({"interaction_id": interaction_id, "interaction_type": kind, "columns": columns, "hypothesis": hypothesis, "expected_effect": "improve_signal", "priority": "P1" if item.get("materiality") == "material" else "P2", "confidence": "high" if item.get("materiality") == "material" else "medium", "materiality": item.get("materiality"), "reliability": "reliable", "evidence_refs": item.get("evidence_refs", []), "recommended_experiment_id": experiment_id})
        experiments.append({"experiment_id": experiment_id, "priority": hypotheses[-1]["priority"], "hypothesis": hypothesis, "base_feature_blocks": ["safe_numeric", "low_cardinality_categorical"], "added_features": [{"type": "missingness_interaction" if kind == "missingness" else "product" if kind == "numeric_numeric" else "binned_by_category" if kind == "numeric_categorical" else "cross", "columns": columns}], "validation_ref": "validation_evidence.primary_validation", "metric_ref": "metric_evidence.metric_name", "success_criteria": {"minimum_materiality": "small", "must_be_stable_across_folds": True}, "risks": ["fold_safe_only"], "evidence_refs": item.get("evidence_refs", [])})
    return hypotheses, experiments


def _reportable(items: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[float, str, str]:
        columns = item.get("columns") or [""]
        left = str(item.get("left_column", item.get("numeric_column", columns[0])))
        right = str(item.get("right_column", item.get("categorical_column", "")))
        return (-float(item.get("target_interaction_score", item.get("interaction_score", 0)) or 0), left, right)
    return sorted(items, key=sort_key)[:cap]


def _apply_drift_caution(items: list[dict[str, Any]], drift: dict[str, Any]) -> None:
    shifted = _shifted_columns(drift)
    for item in items:
        columns = [item.get("left_column"), item.get("right_column"), item.get("numeric_column"), item.get("categorical_column")]
        if any(column in shifted for column in columns if column):
            current = item.get("reliability", item.get("interaction_reliability"))
            if "reliability" in item:
                item["reliability"] = "caution_drift" if current == "reliable" else current
            else:
                item["interaction_reliability"] = "caution_drift" if current == "reliable" else current


def _numeric_transforms(left: list[float], right: list[float]) -> list[str]:
    transforms = ["product"]
    if sum(abs(value) < 1e-12 for value in right) / len(right) < 0.05:
        transforms.append("ratio")
    if _std(left) > 0 and _std(right) > 0:
        transforms.extend(["sum", "difference"])
    return transforms


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right): return 0.0
    left_std, right_std = _std(left), _std(right)
    if left_std == 0 or right_std == 0: return 0.0
    return sum((a - _mean(left)) * (b - _mean(right)) for a, b in zip(left, right)) / (len(left) * left_std * right_std)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1]); ranks = [0.0] * len(values); index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and order[end + 1][1] == order[index][1]: end += 1
        rank = (index + end) / 2 + 1
        for position in range(index, end + 1): ranks[order[position][0]] = rank
        index = end + 1
    return ranks


def _cramers_v(left: list[Any], right: list[Any]) -> float:
    pairs = [(str(a), str(b)) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 2: return 0.0
    rows, cols = sorted({a for a, _ in pairs}), sorted({b for _, b in pairs}); counts = Counter(pairs); row_counts = Counter(a for a, _ in pairs); col_counts = Counter(b for _, b in pairs); n = len(pairs)
    chi = sum((counts[(a, b)] - row_counts[a] * col_counts[b] / n) ** 2 / (row_counts[a] * col_counts[b] / n) for a in rows for b in cols if row_counts[a] and col_counts[b])
    denominator = n * max(1, min(len(rows) - 1, len(cols) - 1))
    return sqrt(chi / denominator) if denominator else 0.0


def _target_values(frame: pl.DataFrame, target: str | None) -> list[float] | None:
    if not target or target not in frame.columns: return None
    values = frame[target].to_list(); mapping = {value: index for index, value in enumerate(sorted({str(value) for value in values if value is not None}))}
    result = []
    for value in values:
        try: result.append(float(value) if value is not None else None)
        except (TypeError, ValueError): result.append(float(mapping[str(value)]) if value is not None else None)
    return result


def _unseen_combination_rate(test: pl.DataFrame | None, left: str, right: str, train: set[tuple[str, str]]) -> float:
    if test is None or left not in test.columns or right not in test.columns: return 0.0
    pairs = [(str(a), str(b)) for a, b in zip(test[left].to_list(), test[right].to_list()) if a is not None and b is not None]
    return sum(pair not in train for pair in pairs) / len(pairs) if pairs else 0.0


def _base_profile(schema: InferredSchema, profiles: list[TableProfile]) -> TableProfile | None:
    return next((item for item in profiles if item.path == schema.train_base_table or item.table_name == schema.train_base_table), None)


def _read(reader: DatasetReader, table: str | None, columns: list[str], max_rows: int) -> pl.DataFrame | None:
    if not table or not columns: return None
    try: return reader.read_columns(table, columns=list(dict.fromkeys(columns)), n_rows=max_rows)
    except ReaderError: return None


def _is_numeric(profile: Any) -> bool: return any(token in str(profile.dtype).lower() for token in ("int", "float", "decimal"))
def _is_categorical(profile: Any) -> bool: return any(token in str(profile.dtype).lower() for token in ("str", "utf8", "categorical", "bool"))
def _mean(values: list[float]) -> float: return sum(values) / len(values) if values else 0.0
def _std(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]; return sqrt(sum((value - _mean(clean)) ** 2 for value in clean) / len(clean)) if clean else 0.0
def _normalized_range(values: list[float]) -> float: return (max(values) - min(values)) / max(_std(values), 1.0) if values else 0.0
def _materiality(score: float) -> str: return "material" if score >= MATERIAL_SCORE else "small" if score >= SMALL_SCORE else "negligible"
def _as_dict(value: Any) -> dict[str, Any]: return value.model_dump(mode="json") if hasattr(value, "model_dump") else value if isinstance(value, dict) else {}
def _not_testable(reason: str, *, excluded: list[dict[str, str]] | None = None) -> dict[str, Any]: return {"status": "not_testable", "reason": reason, "candidate_columns": [], "excluded_columns": excluded or [], "numeric_numeric": [], "numeric_categorical": [], "categorical_categorical": [], "missingness_interactions": [], "redundancy_groups": [], "interaction_hypotheses": [], "recommended_experiments": [], "warnings": [], "limitations": [reason]}
def _leakage_columns(items: list[dict[str, Any]]) -> set[str]: return {str(column) for item in items if item.get("status") in {"failed", "warning"} for column in _as_dict(item.get("evidence")).get("columns", [])}
def _association_scores(features: dict[str, Any]) -> dict[str, float]:
    scores = {}; 
    for section in ("numeric_feature_diagnostics", "categorical_feature_diagnostics"):
        for item in _as_dict(features.get(section)).get("columns", []): scores[str(item.get("column"))] = abs(float(item.get("target_association") or 0))
    return scores
def _shifted_columns(drift: dict[str, Any]) -> set[str]: return {str(item.get("column")) for section in ("numeric_psi", "categorical_drift", "missingness_drift") for item in _as_dict(drift.get(section)).get("columns", []) if float(item.get("psi", item.get("abs_diff", 0)) or 0) >= 0.1}


__all__ = ["diagnose_interactions"]
