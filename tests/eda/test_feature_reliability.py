from __future__ import annotations

from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader
from kaggle_researcher.eda.modules.drift_analyzer import analyze_drift
from kaggle_researcher.eda.modules.feature_diagnostics import diagnose_features
from kaggle_researcher.eda.modules.feature_probe import probe_feature_families
from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.modules.metric_analyzer import analyze_metric
from kaggle_researcher.eda.modules.schema_inferer import infer_schema
from kaggle_researcher.eda.modules.strategy_hints import build_eda_strategy_hints
from kaggle_researcher.eda.modules.table_profiler import profile_tables
from kaggle_researcher.eda.modules.validation_analyzer import analyze_validation
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaTaskPlan
from kaggle_researcher.eda.summary import build_eda_summary


def test_low_cardinality_ordinal_numeric_is_not_low_information(tmp_path: Path) -> None:
    rows = [
        f"{idx},{1 if idx % 3 == 2 else 0},{idx % 3},{float(idx) / 10},cat{idx % 2}"
        for idx in range(90)
    ]
    diagnostics = _diagnostics(tmp_path, rows)

    ordinal = _numeric(diagnostics, "ord_level")
    assert ordinal["feature_value_type"] == "ordinal_low_cardinality"
    assert ordinal["target_association_reliability"] != "not_reliable"
    assert ordinal not in diagnostics["numeric_feature_diagnostics"]["low_information"]


def test_count_zero_inflated_numeric_has_cautious_outlier_reliability(tmp_path: Path) -> None:
    counts = [0] * 50 + [1] * 20 + [2] * 10 + [20] * 5
    rows = [
            f"{idx},{1 if value >= 2 else 0},{idx % 3},{value},cat{idx % 4}"
        for idx, value in enumerate(counts)
    ]
    diagnostics = _diagnostics(tmp_path, rows, header="row_id,target,ord_level,count_feature,category")

    count_feature = _numeric(diagnostics, "count_feature")
    assert count_feature["feature_value_type"] == "count_zero_inflated"
    assert count_feature["outlier_reliability"] in {"caution_count_feature", "caution_zero_inflated"}
    assert count_feature not in diagnostics["numeric_feature_diagnostics"]["outlier_heavy"]
    assert count_feature in diagnostics["numeric_feature_diagnostics"]["rare_large_value_features"]


def test_continuous_numeric_outliers_remain_reliable(tmp_path: Path) -> None:
    values = [float(idx) / 10 for idx in range(80)] + [1000.0, 1200.0]
    rows = [
        f"{idx},{1 if idx % 2 else 0},{idx % 3},{value},cat{idx % 4}"
        for idx, value in enumerate(values)
    ]
    diagnostics = _diagnostics(tmp_path, rows, header="row_id,target,ord_level,continuous_feature,category")

    continuous = _numeric(diagnostics, "continuous_feature")
    assert continuous["feature_value_type"] == "continuous"
    assert continuous["outlier_reliability"] == "reliable"
    assert continuous in diagnostics["numeric_feature_diagnostics"]["outlier_heavy"]


def test_near_unique_categorical_target_association_is_caution(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 3},{float(idx) / 10},code_{idx}"
        for idx in range(80)
    ]
    diagnostics = _diagnostics(tmp_path, rows)

    category = _categorical(diagnostics, "category")
    assert category["feature_value_type"] == "code_like"
    assert category["target_association_reliability"] == "not_reliable"
    assert category not in diagnostics["categorical_feature_diagnostics"]["high_target_association_candidates"]
    assert category in diagnostics["categorical_feature_diagnostics"]["target_association_cautions"]


def test_sparse_categorical_requires_rare_handling(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 3},{float(idx) / 10},group_{idx // 2}"
        for idx in range(80)
    ]
    diagnostics = _diagnostics(tmp_path, rows)

    category = _categorical(diagnostics, "category")
    assert category["rare_category_rate"] >= 0.2
    assert category["encoding_reliability"] in {"requires_rare_handling", "requires_fold_fitted_encoding", "high_leakage_risk_if_target_encoded"}
    assert category["target_association_reliability"] in {"caution_sparse_categories", "not_reliable", "caution_high_cardinality"}


def test_high_unseen_category_rate_sets_shift_caution_and_hint(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 3},{float(idx) / 10},train_{idx % 5}"
        for idx in range(60)
    ]
    diagnostics, payload = _diagnostics_with_payload(tmp_path, rows, test_category_prefix="new")

    category = _categorical(diagnostics, "category")
    assert category["unseen_category_rate"] >= 0.05
    assert category["shift_reliability"] == "caution_high_unseen_rate"
    hints = build_eda_strategy_hints(
        {
            "validation_evidence": payload["validation"],
            "metric_evidence": payload["metric"],
            "feature_diagnostics": diagnostics,
            "leakage_evidence": [],
        }
    )
    assert any("robust encoding" in item["action"] for item in hints["feature_engineering"])


def test_text_like_column_classification_is_consistent(tmp_path: Path) -> None:
    rows = [
        f'{idx},{idx % 2},{idx % 3},{float(idx) / 10},"customer wrote a longer note number {idx}"'
        for idx in range(50)
    ]
    diagnostics = _diagnostics(tmp_path, rows)

    category = _categorical(diagnostics, "category")
    text = _text(diagnostics, "category")
    assert category["feature_value_type"] == "text_like"
    assert text["feature_value_type"] == "text_like"
    assert any("length/count/token" in recommendation for recommendation in text["recommendations"])


def test_code_like_column_classification_is_consistent(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 3},{float(idx) / 10},AB{idx:04d}"
        for idx in range(50)
    ]
    diagnostics = _diagnostics(tmp_path, rows)

    category = _categorical(diagnostics, "category")
    text = _text(diagnostics, "category")
    assert category["feature_value_type"] == "code_like"
    assert text["feature_value_type"] == "code_like"
    assert any("prefix/suffix" in recommendation for recommendation in text["recommendations"])


def test_target_encoding_probe_split_is_explicit(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 3},{float(idx) / 10},cat{idx % 4}"
        for idx in range(40)
    ]
    _, payload = _diagnostics_with_payload(tmp_path, rows)
    probes = probe_feature_families(
        payload["schema"],
        payload["profiles"],
        {},
        [],
        {},
        payload["metric"].model_dump(mode="json"),
        validation_evidence=payload["validation"].model_dump(mode="json"),
    )

    naive = _probe(probes, "naive_target_encoding_or_woe")
    oof = _probe(probes, "oof_target_encoding_or_woe")
    assert naive["status"] == "unsafe"
    assert naive["evidence"]["safe_policy"] is False
    assert "Do not use naive target encoding" in naive["recommendation"]
    assert oof["evidence"]["safe_policy"] is True
    assert "out-of-fold" in oof["recommendation"]


def test_summary_renders_reliability_cautions(tmp_path: Path) -> None:
    rows = [
        f"{idx},{idx % 2},{idx % 3},{float(idx) / 10},code_{idx}"
        for idx in range(80)
    ]
    diagnostics = _diagnostics(tmp_path, rows)
    summary = build_eda_summary(
        EdaEvidencePack(
            competition_id="feature_reliability",
            created_at="2026-07-11T12:00:00+03:00",
            run_id="feature_reliability_run",
            feature_diagnostics=diagnostics,
        )
    )

    assert "Numeric features needing cautious interpretation" in summary
    assert "Target-association caution" in summary
    assert "target_association_reliability" not in summary


def _diagnostics(
    tmp_path: Path,
    rows: list[str],
    *,
    header: str = "row_id,target,ord_level,continuous_feature,category",
) -> dict:
    diagnostics, _ = _diagnostics_with_payload(tmp_path, rows, header=header)
    return diagnostics


def _diagnostics_with_payload(
    tmp_path: Path,
    rows: list[str],
    *,
    header: str = "row_id,target,ord_level,continuous_feature,category",
    test_category_prefix: str | None = None,
) -> tuple[dict, dict]:
    _write_dataset(tmp_path, rows, header=header, test_category_prefix=test_category_prefix)
    reader = DatasetReader(tmp_path)
    inventory = build_file_inventory(tmp_path)
    schema = infer_schema(inventory, reader, task_type_hint="binary_classification", metric_hint="roc_auc")
    profiles = profile_tables(inventory, schema, reader, sample_rows=1000)
    metric = analyze_metric(
        EdaTaskPlan(competition_id="feature_reliability", task_type="binary_classification", metric={"name": "roc_auc"}),
        schema,
        profiles,
    )
    validation = analyze_validation(schema, profiles, metric, reader)
    drift = analyze_drift(schema, validation, reader)
    diagnostics = diagnose_features(schema, profiles, metric, drift, reader, max_rows=1000)
    return diagnostics, {
        "schema": schema,
        "profiles": profiles,
        "metric": metric,
        "validation": validation,
    }


def _write_dataset(
    tmp_path: Path,
    rows: list[str],
    *,
    header: str,
    test_category_prefix: str | None,
) -> None:
    (tmp_path / "train.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    columns = header.split(",")
    test_columns = [column for column in columns if column != "target"]
    test_rows = []
    for index, row in enumerate(rows[:20], start=1000):
        values = dict(zip(columns, row.split(","), strict=False))
        values["row_id"] = str(index)
        if test_category_prefix is not None and "category" in values:
            values["category"] = f"{test_category_prefix}_{index}"
        test_rows.append(",".join(values[column] for column in test_columns))
    (tmp_path / "test.csv").write_text(",".join(test_columns) + "\n" + "\n".join(test_rows) + "\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("row_id,target\n1000,0\n", encoding="utf-8")


def _numeric(diagnostics: dict, column: str) -> dict:
    return next(item for item in diagnostics["numeric_feature_diagnostics"]["columns"] if item["column"] == column)


def _categorical(diagnostics: dict, column: str) -> dict:
    return next(item for item in diagnostics["categorical_feature_diagnostics"]["columns"] if item["column"] == column)


def _text(diagnostics: dict, column: str) -> dict:
    return next(item for item in diagnostics["text_feature_diagnostics"]["columns"] if item["column"] == column)


def _probe(probes: list[dict], feature_family: str) -> dict:
    return next(item for item in probes if item["feature_family"] == feature_family)
