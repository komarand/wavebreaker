from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.modules.column_policy import ColumnRolePolicy
from kaggle_researcher.eda.schemas import (
    InferredSchema,
    LeakageCheckResult,
    MetricEvidence,
    ValidationEvidence,
)


SUPPORTED_TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
}
UNSUPPORTED_TASK_TYPES = {"ranking", "survival", "forecasting_tabular"}
DATE_NAME_TOKENS = ("date", "timestamp", "_dt", "time")
GROUP_ID_TOKENS = ("id", "key", "group", "query", "customer", "client", "user", "session")
NUMERIC_DTYPE_TOKENS = ("int", "float", "decimal")


def run_baseline(
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    metric_evidence: MetricEvidence,
    leakage_evidence: list[LeakageCheckResult],
    reader: DatasetReader,
    output_dir: Path,
    max_rows: int = 1_000_000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run an honest train-base-only baseline with fold-local preprocessing."""

    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    warnings: list[str] = []
    limitations: list[str] = []
    task_type = _task_type(metric_evidence)
    metric_name = _metric_name(metric_evidence)

    if task_type in UNSUPPORTED_TASK_TYPES:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason=f"{task_type} baseline is not supported in the MVP baseline runner.",
            extra={
                "preprocessing_policy": _preprocessing_policy(
                    None, [], _role_exclusions(inferred_schema), validation_evidence
                )
            },
        )
    if task_type not in SUPPORTED_TASK_TYPES:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason="Unknown or unsupported task_type for baseline runner.",
            extra={
                "preprocessing_policy": _preprocessing_policy(
                    None, [], _role_exclusions(inferred_schema), validation_evidence
                )
            },
        )

    train_table = inferred_schema.train_base_table
    target_column = inferred_schema.target_column
    if train_table is None or target_column is None:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason="Train base table and target column are required.",
            extra={
                "preprocessing_policy": _preprocessing_policy(
                    None, [], _role_exclusions(inferred_schema), validation_evidence
                )
            },
        )

    train_schema = _safe_schema(reader, train_table, warnings)
    schema_names = {column["name"] for column in train_schema}
    if target_column not in schema_names:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason=f"Target column '{target_column}' is not present in train base.",
            warnings=warnings,
            extra={
                "preprocessing_policy": _preprocessing_policy(
                    None, [], _role_exclusions(inferred_schema), validation_evidence
                )
            },
        )

    feature_columns, excluded_columns = _feature_columns(
        train_schema,
        inferred_schema=inferred_schema,
        validation_evidence=validation_evidence,
        leakage_evidence=leakage_evidence,
    )
    if not feature_columns:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason="No safe feature columns are available after exclusions.",
            warnings=warnings,
            limitations=limitations,
            extra={
                "feature_columns": [],
                "excluded_columns": excluded_columns,
                "validation_policy": _validation_policy(validation_evidence),
                "preprocessing_policy": _preprocessing_policy(
                    None,
                    [],
                    excluded_columns,
                    validation_evidence,
                ),
            },
        )

    frame = _safe_read_train_frame(
        reader,
        train_table,
        columns=[target_column, *feature_columns],
        max_rows=max_rows,
        warnings=warnings,
        limitations=limitations,
    )
    if frame is None or frame.height < 2:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason="Not enough train rows for baseline validation.",
            warnings=warnings,
            limitations=limitations,
            extra={
                "feature_columns": feature_columns,
                "excluded_columns": excluded_columns,
                "validation_policy": _validation_policy(validation_evidence),
                "preprocessing_policy": _preprocessing_policy(
                    frame,
                    feature_columns,
                    excluded_columns,
                    validation_evidence,
                ),
            },
        )

    metric_available = bool(metric_evidence.local_metric_available)
    if not metric_available:
        warnings.append(
            f"Metric '{metric_name}' is not locally available; baseline metric was skipped."
        )

    try:
        splits = _build_splits(
            frame,
            target_column=target_column,
            validation_evidence=validation_evidence,
            task_type=task_type,
            random_seed=random_seed,
            warnings=warnings,
        )
    except RuntimeError as exc:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason=str(exc),
            warnings=warnings,
            limitations=limitations,
            extra={
                "feature_columns": feature_columns,
                "excluded_columns": excluded_columns,
                "validation_policy": _validation_policy(validation_evidence),
                "preprocessing_policy": _preprocessing_policy(
                    frame,
                    feature_columns,
                    excluded_columns,
                    validation_evidence,
                ),
            },
        )

    fold_results: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    metric_values: list[float] = []
    model_type = ""
    prediction_kind = _prediction_kind(task_type, metric_evidence)

    for fold_idx, (train_indices, valid_indices) in enumerate(splits):
        train_fold = frame[train_indices]
        valid_fold = frame[valid_indices]
        try:
            model, model_type = _fit_model(
                train_fold,
                feature_columns=feature_columns,
                target_column=target_column,
                task_type=task_type,
                random_seed=random_seed,
            )
            predictions = _predict(model, valid_fold, feature_columns, prediction_kind)
        except Exception as exc:
            warnings.append(f"Baseline fold {fold_idx} failed: {exc}")
            continue

        y_true = valid_fold[target_column].to_list()
        metric_value = (
            _compute_metric(metric_name, task_type, y_true, predictions)
            if metric_available
            else None
        )
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
        for row_idx, true_value, pred_value in zip(
            valid_indices,
            y_true,
            _serializable_predictions(predictions),
            strict=True,
        ):
            oof_rows.append(
                {
                    "row_index": int(row_idx),
                    "fold": fold_idx,
                    "target": _normalise_value(true_value),
                    "prediction": pred_value,
                }
            )

    if not fold_results:
        return _skipped(
            task_type=task_type,
            metric_name=metric_name,
            reason="All baseline folds failed.",
            warnings=warnings,
            limitations=limitations,
            extra={
                "feature_columns": feature_columns,
                "excluded_columns": excluded_columns,
                "validation_policy": _validation_policy(validation_evidence),
                "preprocessing_policy": _preprocessing_policy(
                    frame,
                    feature_columns,
                    excluded_columns,
                    validation_evidence,
                ),
            },
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / "baseline_oof_predictions.csv"
    pl.DataFrame(oof_rows).sort("row_index").write_csv(oof_path)

    aggregate_metric = _aggregate_metric(metric_values, metric_evidence)
    return {
        "status": "completed",
        "task_type": task_type,
        "metric_name": metric_name,
        "metric_family": metric_evidence.metric_family,
        "metric_available": metric_available,
        "metric_value": aggregate_metric,
        "greater_is_better": metric_evidence.greater_is_better,
        "model_type": model_type,
        "validation_policy": {
            **_validation_policy(validation_evidence),
            "n_folds": len(fold_results),
        },
        "fold_results": fold_results,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "preprocessing_policy": _preprocessing_policy(
            frame,
            feature_columns,
            excluded_columns,
            validation_evidence,
        ),
        "train_table": train_table,
        "train_rows": frame.height,
        "sampled": _is_sampled(reader, train_table, frame.height, max_rows),
        "sample_rows": frame.height,
        "artifacts": {"oof_predictions": str(oof_path)},
        "warnings": _unique(warnings),
        "limitations": _unique(limitations),
    }


def _fit_model(
    train_frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    random_seed: int,
) -> tuple[Any, str]:
    try:
        model = _lightgbm_pipeline(train_frame, feature_columns, task_type, random_seed)
        model_type = "lightgbm"
    except Exception:
        model = _sklearn_pipeline(train_frame, feature_columns, task_type, random_seed)
        model_type = _sklearn_model_type(task_type, train_frame.height)

    data = train_frame.select(feature_columns).to_pandas()
    target = train_frame[target_column].to_pandas()
    model.fit(data, target)
    return model, model_type


def _lightgbm_pipeline(
    train_frame: pl.DataFrame,
    feature_columns: list[str],
    task_type: str,
    random_seed: int,
) -> Any:
    from lightgbm import LGBMClassifier, LGBMRegressor

    estimator: Any
    if task_type == "regression":
        estimator = LGBMRegressor(n_estimators=50, random_state=random_seed, verbose=-1)
    else:
        estimator = LGBMClassifier(n_estimators=50, random_state=random_seed, verbose=-1)
    return _pipeline(train_frame, feature_columns, estimator)


def _sklearn_pipeline(
    train_frame: pl.DataFrame,
    feature_columns: list[str],
    task_type: str,
    random_seed: int,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task_type == "regression":
        estimator = (
            LinearRegression()
            if train_frame.height < 20
            else HistGradientBoostingRegressor(random_state=random_seed)
        )
    else:
        estimator = (
            LogisticRegression(max_iter=500, random_state=random_seed)
            if train_frame.height < 20
            else HistGradientBoostingClassifier(random_state=random_seed)
        )
    return _pipeline(train_frame, feature_columns, estimator)


def _pipeline(train_frame: pl.DataFrame, feature_columns: list[str], estimator: Any) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_features = [
        column
        for column in feature_columns
        if train_frame[column].dtype.is_numeric()
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            ("model", estimator),
        ]
    )


def _one_hot_encoder() -> Any:
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _predict(
    model: Any,
    frame: pl.DataFrame,
    feature_columns: list[str],
    prediction_kind: str,
) -> list[Any]:
    data = frame.select(feature_columns).to_pandas()
    if prediction_kind == "probability":
        probabilities = model.predict_proba(data)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1].tolist()
        return probabilities.tolist()
    return model.predict(data).tolist()


def _compute_metric(
    metric_name: str,
    task_type: str,
    y_true: list[Any],
    predictions: list[Any],
) -> float | None:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    normalized = _normalize_metric_name(metric_name)
    if not y_true or not predictions:
        return None
    try:
        if normalized in {"auc", "roc_auc", "gini", "normalized_gini"}:
            if len(set(y_true)) < 2:
                return None
            auc = float(roc_auc_score(y_true, predictions))
            return round(2 * auc - 1, 6) if normalized in {"gini", "normalized_gini"} else round(auc, 6)
        if normalized in {"logloss", "log_loss", "cross_entropy"}:
            return round(float(log_loss(y_true, predictions)), 6)
        if normalized == "accuracy":
            return round(float(accuracy_score(y_true, predictions)), 6)
        if normalized in {"f1", "macro_f1", "f1_macro"}:
            average = "binary" if task_type == "binary_classification" else "macro"
            return round(float(f1_score(y_true, predictions, average=average)), 6)
        if normalized in {"rmse", "root_mean_squared_error"}:
            return round(float(mean_squared_error(y_true, predictions) ** 0.5), 6)
        if normalized in {"mse", "mean_squared_error"}:
            return round(float(mean_squared_error(y_true, predictions)), 6)
        if normalized in {"mae", "mean_absolute_error"}:
            return round(float(mean_absolute_error(y_true, predictions)), 6)
        if normalized in {"r2", "r_squared"}:
            return round(float(r2_score(y_true, predictions)), 6)
    except Exception:
        return None
    return None


def _build_splits(
    frame: pl.DataFrame,
    *,
    target_column: str,
    validation_evidence: ValidationEvidence,
    task_type: str,
    random_seed: int,
    warnings: list[str],
) -> list[tuple[list[int], list[int]]]:
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split

    method = str(validation_evidence.primary_validation.get("method") or "").lower()
    n_rows = frame.height
    indices = list(range(n_rows))
    if n_rows < 2:
        raise RuntimeError("At least two rows are required for baseline validation.")

    group_column = validation_evidence.primary_validation.get("group_column")
    if (
        method in {"group_kfold", "stratified_group_kfold", "ranking_group_cv"}
        and group_column in frame.columns
    ):
        groups = frame[str(group_column)].to_list()
        n_groups = len(set(groups))
        if n_groups >= 2:
            n_splits = min(5, n_groups)
            return [
                (train_idx.tolist(), valid_idx.tolist())
                for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(indices, groups=groups)
            ]
        warnings.append("Group validation requested, but fewer than two groups are available.")

    if method in {"stratified_kfold", "stratifiedkfold"} and task_type != "regression":
        target_values = frame[target_column].to_list()
        class_counts = Counter(target_values)
        min_class_count = min(class_counts.values()) if class_counts else 0
        if min_class_count >= 2:
            n_splits = min(5, min_class_count)
            return [
                (train_idx.tolist(), valid_idx.tolist())
                for train_idx, valid_idx in StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_seed,
                ).split(indices, target_values)
            ]
        warnings.append("StratifiedKFold requested, but class counts are too small; using KFold.")

    if method in {"temporal_holdout", "expanding_window"}:
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=max(1, int(round(n_rows * 0.2))),
            shuffle=False,
        )
        return [(list(train_idx), list(valid_idx))]

    n_splits = min(5, n_rows)
    if n_splits < 2:
        raise RuntimeError("At least two folds are required for baseline validation.")
    return [
        (train_idx.tolist(), valid_idx.tolist())
        for train_idx, valid_idx in KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        ).split(indices)
    ]


def _feature_columns(
    train_schema: list[dict[str, str]],
    *,
    inferred_schema: InferredSchema,
    validation_evidence: ValidationEvidence,
    leakage_evidence: list[LeakageCheckResult],
) -> tuple[list[str], list[str]]:
    excluded = {
        inferred_schema.target_column,
        inferred_schema.primary_id_column,
        inferred_schema.prediction_column,
    }
    policy = ColumnRolePolicy(inferred_schema)
    excluded.update(
        item["column"]
        for item in policy.excluded_columns(
            [column["name"] for column in train_schema],
            table=inferred_schema.train_base_table,
            context="model_feature",
        )
    )
    excluded.update(_critical_leakage_columns(leakage_evidence, train_schema))

    method = str(validation_evidence.primary_validation.get("method") or "").lower()
    if "group" in method or method == "ranking_group_cv":
        group_column = validation_evidence.primary_validation.get("group_column")
        if group_column is not None:
            excluded.add(str(group_column))
        for candidate in [*validation_evidence.group_columns, *validation_evidence.query_columns]:
            if isinstance(candidate, dict) and candidate.get("name") is not None:
                excluded.add(str(candidate["name"]))

    feature_columns: list[str] = []
    excluded_columns: list[str] = []
    for column in train_schema:
        name = column["name"]
        if name in excluded or _is_raw_date_string(column):
            excluded_columns.append(name)
        else:
            feature_columns.append(name)
    return feature_columns, _unique(excluded_columns)


def _critical_leakage_columns(
    leakage_evidence: list[LeakageCheckResult],
    train_schema: list[dict[str, str]],
) -> set[str]:
    train_columns = {column["name"] for column in train_schema}
    excluded: set[str] = set()
    for item in leakage_evidence:
        payload = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        if payload.get("severity") != "critical":
            continue
        evidence = payload.get("evidence") or {}
        for key in ("target_column", "id_column", "column"):
            value = evidence.get(key)
            if value in train_columns:
                excluded.add(str(value))
        for value in evidence.get("columns", []):
            if isinstance(value, dict):
                for key in ("column", "name"):
                    column_name = value.get(key)
                    if column_name in train_columns:
                        excluded.add(str(column_name))
            elif value in train_columns:
                excluded.add(str(value))
    return excluded


def _safe_schema(
    reader: DatasetReader,
    table: str,
    warnings: list[str],
) -> list[dict[str, str]]:
    try:
        return reader.read_schema(table)
    except ReaderError as exc:
        warnings.append(str(exc))
        return []


def _safe_read_train_frame(
    reader: DatasetReader,
    table: str,
    columns: list[str],
    max_rows: int,
    warnings: list[str],
    limitations: list[str],
) -> pl.DataFrame | None:
    try:
        limitations.append(f"Baseline uses at most {max_rows} rows from train base.")
        return reader.read_columns(table, columns=_unique(columns), n_rows=max_rows)
    except ReaderError as exc:
        warnings.append(str(exc))
        limitations.append(f"Could not read baseline columns from {table}.")
        return None


def _is_sampled(reader: DatasetReader, table: str, frame_height: int, max_rows: int) -> bool:
    try:
        row_count = reader.count_rows(table)
    except ReaderError:
        return frame_height >= max_rows
    return row_count is not None and row_count > frame_height


def _is_raw_date_string(column: dict[str, str]) -> bool:
    name = column["name"].lower()
    dtype = column["dtype"].lower()
    return (
        any(token in name for token in DATE_NAME_TOKENS)
        and not any(token in dtype for token in NUMERIC_DTYPE_TOKENS)
    )


def _prediction_kind(task_type: str, metric_evidence: MetricEvidence) -> str:
    if task_type in {"binary_classification", "multiclass_classification"}:
        if metric_evidence.requires_probabilities:
            return "probability"
        return "label"
    return "score"


def _aggregate_metric(metric_values: list[float], metric_evidence: MetricEvidence) -> float | None:
    if not metric_values:
        return None
    return round(sum(metric_values) / len(metric_values), 6)


def _validation_policy(validation_evidence: ValidationEvidence) -> dict[str, Any]:
    primary = validation_evidence.primary_validation or {}
    return {
        "method": primary.get("method", "kfold"),
        "group_column": primary.get("group_column"),
    }


def _preprocessing_policy(
    frame: pl.DataFrame | None,
    feature_columns: list[str],
    excluded_columns: list[str],
    validation_evidence: ValidationEvidence,
) -> dict[str, Any]:
    numeric_columns = [
        column
        for column in feature_columns
        if frame is not None and column in frame.columns and frame[column].dtype.is_numeric()
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    high_cardinality_columns = []
    if frame is not None and frame.height:
        high_cardinality_columns = [
            column
            for column in categorical_columns
            if frame[column].n_unique() > 100
            or frame[column].n_unique() / frame.height >= 0.5
        ]
    return {
        "imputation": {
            "numeric": "median",
            "categorical": "most_frequent",
            "fit_scope": "training_fold_only",
        },
        "categorical_encoding": {
            "method": "one_hot",
            "handle_unknown": "ignore",
            "fit_scope": "training_fold_only",
        },
        "high_cardinality_handling": {
            "columns": high_cardinality_columns,
            "policy": "one_hot_with_unknown_ignore",
            "target_encoding": "disabled",
        },
        "excluded_columns": list(excluded_columns),
        "validation_split_policy": _validation_policy(validation_evidence),
    }


def _role_exclusions(inferred_schema: InferredSchema) -> list[str]:
    return _unique(
        [
            column
            for column in (
                inferred_schema.target_column,
                inferred_schema.primary_id_column,
                inferred_schema.prediction_column,
            )
            if column
        ]
    )


def _task_type(metric_evidence: MetricEvidence) -> str:
    return metric_evidence.task_type or "unknown"


def _metric_name(metric_evidence: MetricEvidence) -> str:
    return metric_evidence.metric_name or "unknown"


def _sklearn_model_type(task_type: str, n_rows: int) -> str:
    if task_type == "regression":
        return "sklearn_linear_regression" if n_rows < 20 else "sklearn_hist_gradient_boosting_regressor"
    return "sklearn_logistic_regression" if n_rows < 20 else "sklearn_hist_gradient_boosting_classifier"


def _normalize_metric_name(metric_name: str) -> str:
    return metric_name.strip().lower().replace("-", "_").replace(" ", "_")


def _serializable_predictions(predictions: list[Any]) -> list[Any]:
    result = []
    for value in predictions:
        if isinstance(value, list):
            result.append([_normalise_value(item) for item in value])
        else:
            result.append(_normalise_value(value))
    return result


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _skipped(
    *,
    task_type: str,
    metric_name: str,
    reason: str,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "task_type": task_type,
        "metric_name": metric_name,
        "reason": reason,
        "warnings": _unique(warnings or []),
        "limitations": _unique(limitations or []),
        **(extra or {}),
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

__all__ = ["run_baseline"]
