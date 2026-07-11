from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.presets import CompetitionPreset
from kaggle_researcher.eda.schemas import (
    ColumnRole,
    FileInventoryResult,
    InferredSchema,
    TableSchema,
)


READABLE_TABULAR_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}
PROFILE_ROWS = 200_000

TARGET_WORDS = {
    "target",
    "label",
    "class",
    "y",
    "response",
    "outcome",
    "score",
    "rating",
    "relevance",
    "demand",
    "sales",
    "price",
    "loss",
    "default",
    "clicked",
    "conversion",
}
PREDICTION_WORDS = {
    "prediction",
    "pred",
    "target",
    "label",
    "score",
    "probability",
    "prob",
    "class",
    "saleprice",
    "price",
    "relevance",
    "rating",
}
GROUP_WORDS = {"group", "fold", "customer", "client", "user", "session", "query"}
TIME_WORDS = {"period", "week", "month", "year", "quarter"}
DATE_WORDS = {"date", "timestamp", "datetime", "dt"}
METADATA_WORDS = {"index", "idx", "fold", "split"}
TRAIN_STEMS = {"train", "training"}
TEST_STEMS = {"test", "testing"}
SUBMISSION_STEMS = {
    "sample_submission",
    "submission",
    "submissions",
    "sample",
    "baseline_submission",
}


@dataclass
class ColumnStats:
    n_rows: int | None = None
    n_unique: int | None = None
    unique_ratio: float | None = None
    non_null_count: int | None = None


@dataclass
class TableEvidence:
    table: TableSchema
    column_names: list[str]
    columns_lower: dict[str, str]
    dtype_by_column: dict[str, str]
    n_rows: int | None = None
    stats: dict[str, ColumnStats] = field(default_factory=dict)


@dataclass
class Candidate:
    name: str
    score: float
    confidence: str
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


def infer_schema(
    file_inventory: FileInventoryResult,
    reader: DatasetReader,
    preset: CompetitionPreset | None = None,
    *,
    task_type_hint: str | None = None,
    metric_hint: str | None = None,
) -> InferredSchema:
    evidence: list[TableEvidence] = []
    warnings: list[str] = []

    for dataset_file in file_inventory.files:
        if not dataset_file.can_read or dataset_file.extension not in READABLE_TABULAR_EXTENSIONS:
            continue
        try:
            columns = reader.read_schema(dataset_file.path)
        except ReaderError as exc:
            warnings.append(f"{dataset_file.path}: {exc}")
            continue
        table_schema = _build_table_schema(dataset_file, columns, preset=preset)
        table_evidence = _build_table_evidence(table_schema, reader, warnings)
        evidence.append(table_evidence)

    _infer_table_roles(evidence, warnings)

    train_base_table, train_candidates = _select_train_base_table(evidence, warnings)
    test_base_table, test_candidates = _select_test_base_table(
        evidence,
        train_base_table=train_base_table,
        warnings=warnings,
    )
    sample_submission_table, submission_candidates = _select_sample_submission_table(
        evidence,
        test_base_table=test_base_table,
        warnings=warnings,
    )

    target_candidate, target_candidates = _infer_target_column(
        evidence,
        train_base_table=train_base_table,
        test_base_table=test_base_table,
        sample_submission_table=sample_submission_table,
        task_type_hint=task_type_hint,
        metric_hint=metric_hint,
        warnings=warnings,
    )
    primary_id_candidate, id_candidates, group_candidates = _infer_id_and_group_columns(
        evidence,
        train_base_table=train_base_table,
        test_base_table=test_base_table,
        sample_submission_table=sample_submission_table,
        target_column=target_candidate.name if target_candidate else None,
        warnings=warnings,
    )
    prediction_columns, prediction_candidates = _infer_prediction_columns(
        evidence,
        sample_submission_table=sample_submission_table,
        primary_id_column=primary_id_candidate.name if primary_id_candidate else None,
        target_column=target_candidate.name if target_candidate else None,
        task_type_hint=task_type_hint,
    )

    target_column = target_candidate.name if target_candidate else None
    primary_id_column = primary_id_candidate.name if primary_id_candidate else None
    prediction_column = prediction_columns[0] if prediction_columns else None
    candidate_time_columns = _unique_column_names(evidence, role="time")
    candidate_date_columns = _unique_column_names(evidence, role="date")
    candidate_group_columns = _unique([candidate.name for candidate in group_candidates])
    candidate_join_keys = _unique(
        [
            key
            for item in evidence
            for key in item.table.candidate_join_keys
        ]
    )

    _apply_global_column_roles(
        evidence,
        target_column=target_column,
        primary_id_column=primary_id_column,
        prediction_columns=prediction_columns,
        candidate_group_columns=candidate_group_columns,
    )

    if train_base_table is None:
        warnings.append("Train base table could not be inferred.")
    if test_base_table is None:
        warnings.append("Test base table could not be inferred.")
    if sample_submission_table is None:
        warnings.append("Sample submission table could not be inferred.")
    if target_column is None:
        warnings.append("Target column could not be inferred from train tables.")
    if primary_id_column is None:
        warnings.append("Primary id column could not be inferred.")

    confidence = _schema_confidence(
        train_base_table=train_base_table,
        test_base_table=test_base_table,
        target_column=target_column,
        primary_id_column=primary_id_column,
        warnings=warnings,
    )

    role_warnings = _unique([warning for warning in warnings if "could not" in warning.lower() or "ambiguous" in warning.lower()])
    global_roles: dict[str, Any] = {
        "target_column": target_column,
        "target_column_confidence": target_candidate.confidence if target_candidate else None,
        "target_column_reason": "; ".join(target_candidate.reasons) if target_candidate else None,
        "target_column_candidates": [candidate.as_dict() for candidate in target_candidates],
        "primary_id_column": primary_id_column,
        "primary_id_column_confidence": primary_id_candidate.confidence if primary_id_candidate else None,
        "primary_id_column_reason": "; ".join(primary_id_candidate.reasons) if primary_id_candidate else None,
        "primary_id_column_candidates": [candidate.as_dict() for candidate in id_candidates],
        "prediction_column": prediction_column,
        "prediction_columns": prediction_columns,
        "prediction_column_candidates": [candidate.as_dict() for candidate in prediction_candidates],
        "train_base_table": train_base_table,
        "test_base_table": test_base_table,
        "sample_submission_table": sample_submission_table,
        "train_base_table_candidates": [candidate.as_dict() for candidate in train_candidates],
        "test_base_table_candidates": [candidate.as_dict() for candidate in test_candidates],
        "sample_submission_table_candidates": [candidate.as_dict() for candidate in submission_candidates],
        "candidate_time_columns": candidate_time_columns,
        "candidate_date_columns": candidate_date_columns,
        "candidate_group_columns": candidate_group_columns,
        "candidate_group_column_details": [candidate.as_dict() for candidate in group_candidates],
        "candidate_join_keys": candidate_join_keys,
        "role_inference_warnings": role_warnings,
    }

    return InferredSchema(
        global_roles=global_roles,
        tables=[item.table for item in evidence],
        target_column=target_column,
        primary_id_column=primary_id_column,
        prediction_column=prediction_column,
        train_base_table=train_base_table,
        test_base_table=test_base_table,
        sample_submission_table=sample_submission_table,
        candidate_time_columns=candidate_time_columns,
        candidate_group_columns=candidate_group_columns,
        candidate_date_columns=candidate_date_columns,
        confidence=confidence,
        warnings=_unique(warnings),
    )


def _build_table_schema(
    dataset_file: Any,
    columns: list[dict[str, str]],
    *,
    preset: CompetitionPreset | None,
) -> TableSchema:
    table_role = _table_role(dataset_file.role_hint)
    table_type = _table_type(dataset_file.table_hint)
    column_roles = [
        _infer_column_role(column["name"], table_role=table_role, preset=preset)
        for column in columns
    ]
    candidate_join_keys = [
        role.name
        for role in column_roles
        if role.role in {"primary_id", "group"} or _is_id_like(role.name)
    ]
    candidate_time_columns = [role.name for role in column_roles if role.role == "time"]
    candidate_date_columns = [role.name for role in column_roles if role.role == "date"]
    warnings: list[str] = []
    confidence = "high" if column_roles else "low"
    if not column_roles:
        warnings.append("No columns were available for schema inference.")

    return TableSchema(
        table_name=Path(dataset_file.name).stem,
        path=dataset_file.path,
        role=table_role,
        table_type=table_type,
        n_columns=len(columns),
        columns=columns,
        column_roles=column_roles,
        candidate_join_keys=candidate_join_keys,
        candidate_time_columns=candidate_time_columns,
        candidate_date_columns=candidate_date_columns,
        confidence=confidence,
        warnings=warnings,
    )


def _build_table_evidence(
    table: TableSchema,
    reader: DatasetReader,
    warnings: list[str],
) -> TableEvidence:
    column_names = [column["name"] for column in table.columns]
    evidence = TableEvidence(
        table=table,
        column_names=column_names,
        columns_lower={column.lower(): column for column in column_names},
        dtype_by_column={column["name"]: str(column.get("dtype", "")) for column in table.columns},
    )
    try:
        evidence.n_rows = reader.count_rows(table.path)
    except ReaderError as exc:
        warnings.append(str(exc))
    stat_columns = list(column_names)
    if stat_columns:
        try:
            frame = reader.read_columns(table.path, columns=_unique(stat_columns), n_rows=PROFILE_ROWS)
        except ReaderError as exc:
            warnings.append(str(exc))
        else:
            n_rows = int(frame.height)
            for column in stat_columns:
                series = frame[column]
                non_null = int(series.drop_nulls().len())
                n_unique = int(series.n_unique())
                evidence.stats[column] = ColumnStats(
                    n_rows=n_rows,
                    n_unique=n_unique,
                    unique_ratio=(n_unique / n_rows) if n_rows else None,
                    non_null_count=non_null,
                )
    return evidence


def _infer_table_roles(evidence: list[TableEvidence], warnings: list[str]) -> None:
    for item in evidence:
        filename_role = _filename_table_role(item.table.path)
        if filename_role is not None:
            item.table.role = filename_role
        if item.table.role == "submission":
            item.table.table_type = "unknown" if item.table.table_type != "unknown" else item.table.table_type
        elif item.table.table_type in {"unknown", "secondary"} and item.table.role in {"train", "test"}:
            item.table.table_type = "base" if item.table.table_name.lower() in {"train", "test", "training", "testing"} else item.table.table_type

    train_like = [item for item in evidence if item.table.role == "train"]
    test_like = [item for item in evidence if item.table.role == "test"]
    if not test_like:
        return
    for item in evidence:
        if item.table.role != "unknown":
            continue
        candidate = _score_structural_submission(item, test_like, train_like)
        if candidate.score >= 6:
            item.table.role = "submission"
            item.table.table_type = "unknown"
            item.table.confidence = "high" if candidate.score >= 8 else "medium"
            item.table.warnings.append(
                "Inferred sample submission role from structure: "
                + "; ".join(candidate.reasons)
            )
    submission_count = sum(1 for item in evidence if item.table.role == "submission")
    if submission_count > 1:
        warnings.append("Multiple sample submission candidates were inferred; selected candidate is score-ranked.")


def _score_structural_submission(
    item: TableEvidence,
    test_like: list[TableEvidence],
    train_like: list[TableEvidence],
) -> Candidate:
    reasons: list[str] = []
    score = 0.0
    columns = set(item.column_names)
    if 2 <= len(columns) <= 5:
        score += 1.5
        reasons.append("small column count")
    if any(_is_prediction_like(column) or _is_target_like(column) for column in columns):
        score += 1.5
        reasons.append("contains prediction-like column")
    for test in test_like:
        overlap = columns & set(test.column_names)
        id_overlap = [column for column in overlap if _is_id_like(column)]
        if id_overlap:
            score += 2
            reasons.append(f"shares id-like column with test ({', '.join(id_overlap)})")
        if _row_counts_close(item.n_rows, test.n_rows):
            score += 1.5
            reasons.append("row count is close to a test table")
    for train in train_like:
        train_columns = set(train.column_names)
        feature_overlap = [
            column
            for column in columns & train_columns
            if not _is_id_like(column) and not _is_target_like(column) and not _is_prediction_like(column)
        ]
        train_only_features = [
            column
            for column in train_columns - columns
            if not _is_id_like(column) and not _is_target_like(column)
        ]
        if not feature_overlap and train_only_features:
            score += 1
            reasons.append("does not contain train feature structure")
    return Candidate(item.table.path, score, _candidate_confidence(score), reasons)


def _select_train_base_table(
    evidence: list[TableEvidence],
    warnings: list[str],
) -> tuple[str | None, list[Candidate]]:
    candidates: list[Candidate] = []
    train_tables = [item for item in evidence if item.table.role == "train"]
    for item in train_tables:
        score = 0.0
        reasons: list[str] = []
        if len(train_tables) == 1:
            score += 2
            reasons.append("only train-like table")
        if item.table.table_type == "base":
            score += 4
            reasons.append("explicit base hint")
        if _filename_table_role(item.table.path) == "train":
            score += 2
            reasons.append("train-like filename")
        if item.n_rows is not None:
            score += min(item.n_rows, 10_000) / 10_000
            reasons.append("has rows")
        score += min(item.table.n_columns, 50) / 25
        reasons.append("feature-compatible column count")
        candidates.append(Candidate(item.table.path, score, _candidate_confidence(score), reasons))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    selected = _clear_winner(candidates, minimum=3.0, margin=1.0)
    if selected is None and len(candidates) > 1:
        warnings.append("Ambiguous train base table candidates; not forcing a base table.")
    return (selected.name if selected else None), candidates


def _select_test_base_table(
    evidence: list[TableEvidence],
    *,
    train_base_table: str | None,
    warnings: list[str],
) -> tuple[str | None, list[Candidate]]:
    candidates: list[Candidate] = []
    train = _by_path(evidence, train_base_table)
    train_columns = set(train.column_names) if train else set()
    target_like = {column for column in train_columns if _is_target_like(column)}
    comparable_train_columns = train_columns - target_like
    for item in evidence:
        if item.table.role != "test":
            continue
        score = 0.0
        reasons: list[str] = []
        test_table_count = sum(1 for candidate in evidence if candidate.table.role == "test")
        if test_table_count == 1:
            score += 2
            reasons.append("only test-like table")
        if item.table.table_type == "base":
            score += 3
            reasons.append("explicit base hint")
        if _filename_table_role(item.table.path) == "test":
            score += 2
            reasons.append("test-like filename")
        if comparable_train_columns:
            overlap = comparable_train_columns & set(item.column_names)
            score += min(len(overlap), 30) / 3
            reasons.append(f"{len(overlap)} shared train/test columns")
            score += _dtype_overlap_score(train, item, overlap)
            if target_like and not (set(item.column_names) & target_like):
                score += 1
                reasons.append("train-only target-like columns absent")
        candidates.append(Candidate(item.table.path, score, _candidate_confidence(score), reasons))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    selected = _clear_winner(candidates, minimum=4.0, margin=1.0)
    if selected is None and len(candidates) > 1:
        warnings.append("Ambiguous test base table candidates; not forcing a base table.")
    return (selected.name if selected else None), candidates


def _select_sample_submission_table(
    evidence: list[TableEvidence],
    *,
    test_base_table: str | None,
    warnings: list[str],
) -> tuple[str | None, list[Candidate]]:
    test = _by_path(evidence, test_base_table)
    candidates: list[Candidate] = []
    for item in evidence:
        if item.table.role != "submission":
            continue
        score = 0.0
        reasons: list[str] = []
        if _filename_table_role(item.table.path) == "submission":
            score += 4
            reasons.append("submission-like filename")
        if 2 <= len(item.column_names) <= 10:
            score += 1.5
            reasons.append("compact submission schema")
        if test is not None:
            shared_id = [
                column
                for column in item.column_names
                if column in test.column_names and _is_id_like(column)
            ]
            if shared_id:
                score += 2
                reasons.append(f"shares id-like column with test ({', '.join(shared_id)})")
            if _row_counts_close(item.n_rows, test.n_rows):
                score += 1.5
                reasons.append("row count is close to test base")
        if any(_is_prediction_like(column) or _is_target_like(column) for column in item.column_names):
            score += 1
            reasons.append("contains prediction-like output column")
        candidates.append(Candidate(item.table.path, score, _candidate_confidence(score), reasons))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    selected = _clear_winner(candidates, minimum=4.0, margin=0.75)
    if selected is None and len(candidates) > 1:
        warnings.append("Ambiguous sample submission candidates; not forcing a submission table.")
    return (selected.name if selected else None), candidates


def _infer_target_column(
    evidence: list[TableEvidence],
    *,
    train_base_table: str | None,
    test_base_table: str | None,
    sample_submission_table: str | None,
    task_type_hint: str | None,
    metric_hint: str | None,
    warnings: list[str],
) -> tuple[Candidate | None, list[Candidate]]:
    train = _by_path(evidence, train_base_table)
    test = _by_path(evidence, test_base_table)
    sample = _by_path(evidence, sample_submission_table)
    if train is None:
        return None, []
    test_columns = set(test.column_names) if test else set()
    sample_prediction_names = {
        column for column in (sample.column_names if sample else []) if not _is_id_like(column)
    }
    candidates: list[Candidate] = []
    for column in train.column_names:
        reasons: list[str] = []
        score = 0.0
        name_score = _target_name_score(column)
        if column in test_columns:
            if name_score < 4:
                continue
            score += 1
            reasons.append("strong target-like column is present in train and test; possible leakage")
        else:
            score += 3
            reasons.append("present in train base and absent from test base")
        if _is_id_like(column) or _is_metadata_like(column):
            score -= 4
            reasons.append("id/metadata-like name")
        if name_score:
            score += name_score
            reasons.append("target-like column name")
        if column in sample_prediction_names or column.lower() in {name.lower() for name in sample_prediction_names}:
            score += 1
            reasons.append("matches a sample-submission prediction column")
        dtype = train.dtype_by_column.get(column, "")
        stats = train.stats.get(column)
        score += _target_cardinality_score(column, dtype, stats, task_type_hint, metric_hint, reasons)
        if score >= 3:
            candidates.append(Candidate(column, score, _candidate_confidence(score), reasons))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    selected = _clear_winner(candidates, minimum=5.0, margin=1.25)
    if selected is None and candidates and candidates[0].score >= 5.0:
        warnings.append(
            "Ambiguous target column candidates; not forcing target_column. "
            f"Candidates: {_candidate_names(candidates)}."
        )
    return selected, candidates


def _infer_id_and_group_columns(
    evidence: list[TableEvidence],
    *,
    train_base_table: str | None,
    test_base_table: str | None,
    sample_submission_table: str | None,
    target_column: str | None,
    warnings: list[str],
) -> tuple[Candidate | None, list[Candidate], list[Candidate]]:
    train = _by_path(evidence, train_base_table)
    test = _by_path(evidence, test_base_table)
    sample = _by_path(evidence, sample_submission_table)
    if train is None or test is None:
        return None, [], []
    sample_columns = sample.column_names if sample else []
    common = set(train.column_names) & set(test.column_names)
    candidates: list[Candidate] = []
    group_candidates: list[Candidate] = []
    for column in sorted(common):
        if column == target_column or _is_target_like(column):
            continue
        reasons: list[str] = []
        score = 0.0
        if _is_id_like(column):
            score += 3
            reasons.append("id-like column name")
        if column in sample_columns:
            score += 2
            reasons.append("appears in sample submission")
            if sample_columns and sample_columns[0] == column:
                score += 1
                reasons.append("first sample-submission column")
        train_unique = _unique_ratio(train, column)
        test_unique = _unique_ratio(test, column)
        if _high_unique(train_unique) and _high_unique(test_unique):
            score += 3
            reasons.append("high uniqueness in train and test")
        elif _medium_unique(train_unique) and _medium_unique(test_unique):
            score += 1
            reasons.append("moderate uniqueness in train and test")
        elif _is_group_like(column) or _is_id_like(column):
            group_score = 2 + (1 if column in sample_columns else 0)
            group_reasons = ["non-unique id/group-like shared key"]
            if column in sample_columns:
                group_reasons.append("also appears in sample submission")
            group_candidates.append(
                Candidate(column, group_score, _candidate_confidence(group_score), group_reasons)
            )
            score -= 1
            reasons.append("not unique enough for primary row id")
        if _looks_continuous_measure(column, train.dtype_by_column.get(column, ""), train_unique):
            score -= 2
            reasons.append("looks like continuous measurement")
        if score >= 3:
            candidates.append(Candidate(column, score, _candidate_confidence(score), reasons))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    group_candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    selected = _clear_winner(candidates, minimum=5.0, margin=1.0)
    if selected is None and candidates:
        warnings.append(
            "Ambiguous primary id column candidates; not forcing primary_id_column. "
            f"Candidates: {_candidate_names(candidates)}."
        )
    return selected, candidates, group_candidates


def _infer_prediction_columns(
    evidence: list[TableEvidence],
    *,
    sample_submission_table: str | None,
    primary_id_column: str | None,
    target_column: str | None,
    task_type_hint: str | None,
) -> tuple[list[str], list[Candidate]]:
    sample = _by_path(evidence, sample_submission_table)
    if sample is None:
        return [], []
    candidates: list[Candidate] = []
    for column in sample.column_names:
        if column == primary_id_column or _is_id_like(column):
            continue
        reasons = ["non-id sample-submission column"]
        score = 3.0
        if _is_prediction_like(column) or _is_target_like(column):
            score += 2
            reasons.append("prediction-like name")
        if target_column and column.lower() == target_column.lower():
            score += 1
            reasons.append("matches inferred target name")
        if task_type_hint == "multiclass_classification" and column.lower().startswith("class"):
            score += 1
            reasons.append("multiclass class-output column")
        candidates.append(Candidate(column, score, _candidate_confidence(score), reasons))
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.name))
    return [candidate.name for candidate in candidates if candidate.score >= 3], candidates


def _apply_global_column_roles(
    evidence: list[TableEvidence],
    *,
    target_column: str | None,
    primary_id_column: str | None,
    prediction_columns: list[str],
    candidate_group_columns: list[str],
) -> None:
    for item in evidence:
        roles: list[ColumnRole] = []
        for column in item.column_names:
            existing = next((role for role in item.table.column_roles if role.name == column), None)
            if item.table.role == "train" and column == target_column:
                roles.append(ColumnRole(name=column, role="target", confidence="high", reason="Selected by cross-table target inference."))
            elif column == primary_id_column:
                roles.append(ColumnRole(name=column, role="primary_id", confidence="high", reason="Selected by cross-table ID inference."))
            elif item.table.role == "submission" and column in prediction_columns:
                roles.append(ColumnRole(name=column, role="prediction", confidence="high", reason="Selected as sample-submission prediction output."))
            elif column in candidate_group_columns:
                roles.append(ColumnRole(name=column, role="group", confidence="medium", reason="Shared non-unique id/group-like key."))
            elif existing is not None:
                roles.append(existing)
            else:
                roles.append(_infer_generic_column_role(column, table_role=item.table.role))
        item.table.column_roles = roles
        item.table.candidate_join_keys = _unique(
            [
                role.name
                for role in roles
                if role.role in {"primary_id", "group"} or _is_id_like(role.name)
            ]
        )
        item.table.candidate_time_columns = [role.name for role in roles if role.role == "time"]
        item.table.candidate_date_columns = [role.name for role in roles if role.role == "date"]


def _infer_column_role(
    column_name: str,
    table_role: str,
    preset: CompetitionPreset | None,
) -> ColumnRole:
    generic_role = _infer_generic_column_role(column_name, table_role=table_role)
    return _apply_preset_column_hint(
        generic_role,
        column_name=column_name,
        table_role=table_role,
        preset=preset,
    )


def _infer_generic_column_role(column_name: str, *, table_role: str) -> ColumnRole:
    normalized = column_name.lower()
    if table_role == "train" and _target_name_score(column_name) >= 3:
        return ColumnRole(
            name=column_name,
            role="target",
            confidence="medium",
            reason="Column has a generic target-like name in a train table.",
        )
    if _is_id_like(column_name):
        return ColumnRole(
            name=column_name,
            role="primary_id",
            confidence="medium",
            reason="Column has a generic id-like name.",
        )
    if any(token in normalized for token in TIME_WORDS):
        return ColumnRole(
            name=column_name,
            role="time",
            confidence="medium",
            reason="Column name contains a generic time-period signal.",
        )
    if any(token in normalized for token in DATE_WORDS):
        return ColumnRole(
            name=column_name,
            role="date",
            confidence="high",
            reason="Column name contains date/timestamp signal.",
        )
    if table_role == "submission" and (_is_prediction_like(column_name) or normalized.startswith("class_")):
        return ColumnRole(
            name=column_name,
            role="prediction",
            confidence="medium",
            reason="Column looks like a sample-submission prediction column.",
        )
    if _is_group_like(column_name):
        return ColumnRole(
            name=column_name,
            role="group",
            confidence="medium",
            reason="Column name contains a group-like token.",
        )
    return ColumnRole(
        name=column_name,
        role="unknown",
        confidence="low",
        reason="No semantic role heuristic matched.",
    )


def _apply_preset_column_hint(
    generic_role: ColumnRole,
    *,
    column_name: str,
    table_role: str,
    preset: CompetitionPreset | None,
) -> ColumnRole:
    if preset is None or generic_role.role != "unknown":
        return generic_role

    normalized = column_name.lower()
    target_names = _preset_names(preset, "preferred_target_columns")
    id_names = _preset_names(preset, "preferred_id_columns")
    time_names = _preset_names(preset, "preferred_time_columns")
    prediction_names = _preset_names(preset, "preferred_prediction_columns")

    if normalized in target_names and table_role == "train":
        return ColumnRole(
            name=column_name,
            role="target",
            confidence="high",
            reason="Column matches preset target-column hints and appears in a train table.",
        )
    if normalized in id_names:
        return ColumnRole(
            name=column_name,
            role="primary_id",
            confidence="high",
            reason="Column matches preset primary id hints.",
        )
    if normalized in time_names:
        return ColumnRole(
            name=column_name,
            role="time",
            confidence="high",
            reason="Column matches preset time-column hints.",
        )
    if table_role == "submission" and normalized in prediction_names:
        return ColumnRole(
            name=column_name,
            role="prediction",
            confidence="high",
            reason="Column matches preset prediction-column hints in sample submission.",
        )
    return generic_role


def _filename_table_role(path: str) -> str | None:
    stem = Path(path).stem.lower()
    tokens = _tokens(stem)
    if stem in SUBMISSION_STEMS or stem.endswith("_submission") or "sample_submission" in stem:
        return "submission"
    if "submission" in tokens or "submissions" in tokens:
        return "submission"
    if stem in TRAIN_STEMS or stem.startswith("train_") or stem.endswith("_train") or "train" in tokens or "training" in tokens:
        return "train"
    if stem in TEST_STEMS or stem.startswith("test_") or stem.endswith("_test") or "test" in tokens or "testing" in tokens:
        return "test"
    return None


def _table_role(role_hint: str) -> str:
    if role_hint == "train":
        return "train"
    if role_hint == "test":
        return "test"
    if role_hint == "sample_submission":
        return "submission"
    if role_hint == "metadata":
        return "metadata"
    return "unknown"


def _table_type(table_hint: str) -> str:
    if table_hint == "base":
        return "base"
    if table_hint in {"secondary", "depth_0", "depth_1", "depth_2"}:
        return table_hint
    return "unknown"


def _tokens(value: str) -> list[str]:
    words = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    split_words: list[str] = []
    for word in words:
        split_words.extend(re.findall(r"[a-z]+|\d+", word))
    return words + split_words


def _is_id_like(column_name: str) -> bool:
    normalized = column_name.lower()
    tokens = _tokens(normalized)
    return (
        normalized == "id"
        or normalized.endswith("_id")
        or normalized.startswith("id_")
        or normalized in {"rowid", "row_id", "record_key", "record_id", "object_id"}
        or (len(tokens) >= 2 and tokens[-1] == "id")
        or normalized.endswith("id")
        and any(token in normalized for token in ("row", "object", "customer", "user", "item", "passenger", "record", "case"))
    )


def _is_target_like(column_name: str) -> bool:
    normalized = column_name.lower()
    tokens = set(_tokens(normalized))
    return (
        normalized in TARGET_WORDS
        or bool(tokens & TARGET_WORDS)
        or normalized.startswith(("is_", "has_"))
        or normalized.startswith("is") and len(normalized) > 2
    )


def _target_name_score(column_name: str) -> float:
    normalized = column_name.lower()
    tokens = set(_tokens(normalized))
    if normalized in {"target", "label", "y"}:
        return 4.0
    if normalized.startswith(("is_", "has_")):
        return 3.5
    if normalized.startswith("is") and len(normalized) > 2:
        return 2.5
    if normalized in TARGET_WORDS or tokens & TARGET_WORDS:
        return 3.0
    return 0.0


def _is_prediction_like(column_name: str) -> bool:
    normalized = column_name.lower()
    tokens = set(_tokens(normalized))
    return (
        normalized in PREDICTION_WORDS
        or bool(tokens & PREDICTION_WORDS)
        or normalized.startswith(("pred", "prob", "class_"))
    )


def _is_group_like(column_name: str) -> bool:
    normalized = column_name.lower()
    tokens = set(_tokens(normalized))
    return bool(tokens & GROUP_WORDS) or any(word in normalized for word in ("query_id", "session_id", "user_id", "customer_id"))


def _is_metadata_like(column_name: str) -> bool:
    normalized = column_name.lower()
    tokens = set(_tokens(normalized))
    return normalized in METADATA_WORDS or bool(tokens & METADATA_WORDS)


def _target_cardinality_score(
    column: str,
    dtype: str,
    stats: ColumnStats | None,
    task_type_hint: str | None,
    metric_hint: str | None,
    reasons: list[str],
) -> float:
    task = str(task_type_hint or "").lower()
    metric = str(metric_hint or "").lower()
    unique = stats.n_unique if stats else None
    unique_ratio = stats.unique_ratio if stats else None
    score = 0.0
    if task in {"binary_classification", "binary"} or metric in {"auc", "roc_auc", "logloss", "log_loss"}:
        if unique == 2:
            score += 2
            reasons.append("binary cardinality matches task/metric")
        elif unique is not None and 2 < unique <= 20:
            score += 0.5
            reasons.append("low/moderate classification cardinality")
    elif task in {"multiclass_classification", "multiclass"}:
        if unique is not None and 2 <= unique <= 100:
            score += 1.5
            reasons.append("multiclass-like cardinality")
    elif task == "regression" or metric in {"rmse", "mae", "rmsle", "mse"}:
        if _is_numeric_dtype(dtype) and unique is not None and unique >= 3 and not _high_unique(unique_ratio):
            score += 1.5
            reasons.append("numeric non-id regression-like cardinality")
    elif task == "ranking":
        if _is_target_like(column):
            score += 1
            reasons.append("ranking target-like name")
    else:
        if unique == 2:
            score += 1
            reasons.append("binary cardinality")
        elif _is_numeric_dtype(dtype) and unique is not None and unique >= 3 and not _high_unique(unique_ratio):
            score += 0.5
            reasons.append("numeric target-compatible cardinality")
    return score


def _is_numeric_dtype(dtype: str) -> bool:
    lowered = str(dtype).lower()
    return any(token in lowered for token in ("int", "float", "decimal", "numeric"))


def _unique_ratio(table: TableEvidence, column: str) -> float | None:
    stats = table.stats.get(column)
    if stats is None:
        return None
    return stats.unique_ratio


def _high_unique(value: float | None) -> bool:
    return value is not None and value >= 0.98


def _medium_unique(value: float | None) -> bool:
    return value is not None and value >= 0.75


def _looks_continuous_measure(column: str, dtype: str, unique_ratio: float | None) -> bool:
    if _is_id_like(column):
        return False
    return _is_numeric_dtype(dtype) and _high_unique(unique_ratio)


def _dtype_overlap_score(left: TableEvidence | None, right: TableEvidence, columns: set[str]) -> float:
    if left is None or not columns:
        return 0.0
    matches = sum(1 for column in columns if left.dtype_by_column.get(column) == right.dtype_by_column.get(column))
    return min(matches, 20) / 10


def _row_counts_close(left: int | None, right: int | None) -> bool:
    if left is None or right is None or left == 0 or right == 0:
        return False
    ratio = min(left, right) / max(left, right)
    return ratio >= 0.8


def _clear_winner(candidates: list[Candidate], *, minimum: float, margin: float) -> Candidate | None:
    if not candidates or candidates[0].score < minimum:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if candidates[0].score - candidates[1].score >= margin:
        return candidates[0]
    return None


def _candidate_confidence(score: float) -> str:
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def _candidate_names(candidates: list[Candidate]) -> str:
    return ", ".join(f"{candidate.name} ({candidate.score:.1f})" for candidate in candidates)


def _by_path(evidence: list[TableEvidence], path: str | None) -> TableEvidence | None:
    if path is None:
        return None
    return next((item for item in evidence if item.table.path == path), None)


def _unique_column_names(
    evidence: list[TableEvidence],
    role: str,
) -> list[str]:
    return _unique(
        [
            column_role.name
            for item in evidence
            for column_role in item.table.column_roles
            if column_role.role == role
        ]
    )


def _schema_confidence(
    *,
    train_base_table: str | None,
    test_base_table: str | None,
    target_column: str | None,
    primary_id_column: str | None,
    warnings: list[str],
) -> str:
    severe_warnings = [warning for warning in warnings if "ambiguous" in warning.lower()]
    if all([train_base_table, test_base_table, target_column, primary_id_column]) and not severe_warnings:
        return "high"
    if train_base_table and test_base_table and primary_id_column:
        return "medium"
    return "low"


def _preset_names(
    preset: CompetitionPreset,
    field_name: str,
) -> set[str]:
    return {str(name).lower() for name in getattr(preset, field_name)}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["infer_schema"]
