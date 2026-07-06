from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from kaggle_researcher.eda.schemas import InferredSchema, TableProfile


GROUP_TOKENS = ("query", "group", "session", "user", "customer", "client", "patient", "entity")
TIME_TOKENS = ("week", "period", "date", "time", "month", "year", "timestamp")


def infer_class_balance(df: Any, target_col: str) -> dict[str, Any]:
    values = _non_missing_values(_column_values(df, target_col))
    counts = Counter(values)
    n_rows = len(values)
    classes = [
        {"class": label, "count": count, "pct": _ratio(count, n_rows)}
        for label, count in sorted(counts.items(), key=lambda item: str(item[0]))
    ]
    result: dict[str, Any] = {
        "target_col": target_col,
        "n_rows": n_rows,
        "n_classes": len(counts),
        "classes": classes,
    }
    if set(counts).issubset({0, 1, 0.0, 1.0, "0", "1"}) and n_rows:
        result["positive_rate"] = sum(_is_positive(value) for value in values) / n_rows
    return result


def infer_regression_target_stats(df: Any, target_col: str) -> dict[str, Any]:
    values = [float(value) for value in _non_missing_values(_column_values(df, target_col))]
    if not values:
        return {"target_col": target_col, "n_rows": 0}
    sorted_values = sorted(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "target_col": target_col,
        "n_rows": len(values),
        "mean": mean,
        "std": variance**0.5,
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "q50": _quantile(sorted_values, 0.5),
    }


def infer_candidate_group_columns(
    schema: InferredSchema,
    profiles: list[TableProfile],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name in schema.candidate_group_columns:
        candidates.append(
            {"name": name, "source": "schema", "reason": "schema group candidate"}
        )
    for key in schema.global_roles.get("candidate_join_keys", []):
        if _contains_token(str(key), GROUP_TOKENS):
            candidates.append(
                {
                    "name": str(key),
                    "source": "schema",
                    "reason": "join key looks group-like",
                }
            )
    for profile in profiles:
        for column in profile.columns:
            if _contains_token(column.name, GROUP_TOKENS):
                candidates.append(
                    {
                        "name": column.name,
                        "source": profile.path,
                        "reason": "column name looks group-like",
                    }
                )
    return _dedupe_candidates(candidates)


def infer_candidate_time_columns(
    schema: InferredSchema,
    profiles: list[TableProfile],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name in schema.candidate_time_columns:
        candidates.append({"name": name, "source": "schema", "reason": "schema time candidate"})
    for name in schema.candidate_date_columns:
        candidates.append({"name": name, "source": "schema", "reason": "schema date candidate"})
    for profile in profiles:
        for column in profile.columns:
            if _contains_token(column.name, TIME_TOKENS) or column.date_min is not None:
                candidates.append(
                    {
                        "name": column.name,
                        "source": profile.path,
                        "reason": "column name or profile looks time-like",
                    }
                )
    return _dedupe_candidates(candidates)


def summarize_column_distribution(
    df: Any,
    col: str,
    target_col: str | None = None,
) -> list[dict[str, Any]]:
    values = _column_values(df, col)
    targets = _column_values(df, target_col) if target_col is not None else None
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for index, value in enumerate(values):
        if _is_missing(value):
            continue
        grouped[value].append(targets[index] if targets is not None else None)
    total = sum(len(items) for items in grouped.values())
    rows: list[dict[str, Any]] = []
    for value in sorted(grouped, key=_natural_sort_key):
        item_targets = grouped[value]
        row: dict[str, Any] = {
            "value": value,
            "n_rows": len(item_targets),
            "pct": _ratio(len(item_targets), total),
        }
        if targets is not None:
            numeric_targets = [
                float(target)
                for target in item_targets
                if not _is_missing(target)
            ]
            row["target_mean"] = (
                sum(numeric_targets) / len(numeric_targets)
                if numeric_targets
                else None
            )
        rows.append(row)
    return rows


def _column_values(df: Any, col: str | None) -> list[Any]:
    if col is None:
        return []
    if hasattr(df, "get_column"):
        return df.get_column(col).to_list()
    series = df[col]
    if hasattr(series, "to_list"):
        return series.to_list()
    if hasattr(series, "tolist"):
        return series.tolist()
    return list(series)


def _non_missing_values(values: list[Any]) -> list[Any]:
    return [value for value in values if not _is_missing(value)]


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _is_positive(value: Any) -> bool:
    return str(value) == "1" or value == 1 or value == 1.0


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _contains_token(name: str, tokens: tuple[str, ...]) -> bool:
    normalized = name.lower()
    return any(token in normalized for token in tokens)


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        key = str(candidate["name"]).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _natural_sort_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, (int, float)):
        return ("number", value)
    text = str(value)
    try:
        return ("number", float(text))
    except ValueError:
        return ("text", text)


__all__ = [
    "infer_candidate_group_columns",
    "infer_candidate_time_columns",
    "infer_class_balance",
    "infer_regression_target_stats",
    "summarize_column_distribution",
]
