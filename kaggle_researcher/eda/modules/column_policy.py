from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kaggle_researcher.eda.schemas import InferredSchema, MetricEvidence


ID_TOKENS = ("id", "key", "row", "record", "object")
METADATA_TOKENS = ("fold", "split", "index", "idx", "metadata")


@dataclass(frozen=True)
class ColumnExclusion:
    column: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"column": self.column, "reason": self.reason}


class ColumnRolePolicy:
    """Central policy for deciding whether columns are analysis-safe."""

    def __init__(
        self,
        inferred_schema: InferredSchema,
        metric_evidence: MetricEvidence | dict[str, Any] | None = None,
    ) -> None:
        self.schema = inferred_schema
        self.metric = _as_dict(metric_evidence)
        global_roles = inferred_schema.global_roles or {}
        self.target_column = inferred_schema.target_column
        self.primary_id_column = inferred_schema.primary_id_column
        self.sample_submission_table = inferred_schema.sample_submission_table
        self.prediction_columns = set(global_roles.get("prediction_columns") or [])
        if inferred_schema.prediction_column:
            self.prediction_columns.add(inferred_schema.prediction_column)
        self.group_columns = set(inferred_schema.candidate_group_columns)
        self.group_columns.update(
            str(item.get("name"))
            for item in global_roles.get("candidate_group_column_details", [])
            if isinstance(item, dict) and item.get("name") is not None
        )
        self.time_columns = set(inferred_schema.candidate_time_columns)
        self.date_columns = set(inferred_schema.candidate_date_columns)
        self._column_roles_by_table = {
            table.path: {role.name: role.role for role in table.column_roles}
            for table in inferred_schema.tables
        }

    def is_primary_id(self, column: str, table: str | None = None) -> bool:
        return column == self.primary_id_column or self._role(column, table) == "primary_id"

    def is_target(self, column: str, table: str | None = None) -> bool:
        return column == self.target_column or self._role(column, table) == "target"

    def is_prediction(self, column: str, table: str | None = None) -> bool:
        return column in self.prediction_columns or self._role(column, table) == "prediction"

    def is_submission_only(self, column: str, table: str | None = None) -> bool:
        return table is not None and table == self.sample_submission_table

    def is_group(self, column: str, table: str | None = None) -> bool:
        return column in self.group_columns or self._role(column, table) == "group"

    def is_time_or_date(self, column: str, table: str | None = None) -> bool:
        return (
            column in self.time_columns
            or column in self.date_columns
            or self._role(column, table) in {"time", "date"}
        )

    def is_metadata(self, column: str, table: str | None = None) -> bool:
        normalized = column.strip().lower()
        if self._role(column, table) in {"time", "date"}:
            return False
        return normalized in METADATA_TOKENS or any(token == normalized for token in METADATA_TOKENS)

    def is_safe_model_feature(self, column: str, table: str | None = None) -> bool:
        return self.exclusion_reason(column, table=table, context="model_feature") is None

    def is_safe_drift_feature(self, column: str, table: str | None = None) -> bool:
        return self.exclusion_reason(column, table=table, context="drift") is None

    def is_safe_leakage_probe_feature(self, column: str, table: str | None = None) -> bool:
        return self.exclusion_reason(column, table=table, context="leakage_probe") is None

    def exclusion_reason(
        self,
        column: str,
        *,
        table: str | None = None,
        context: str = "model_feature",
    ) -> str | None:
        if self.is_submission_only(column, table):
            return "submission_only"
        if self.is_target(column, table):
            return "target_column"
        if self.is_prediction(column, table):
            return "prediction_column"
        if self.is_primary_id(column, table):
            return "primary_id"
        if self.is_group(column, table) and context in {"model_feature", "drift", "leakage_probe"}:
            return "group_column"
        if self.is_metadata(column, table):
            return "metadata_column"
        if context == "relationship" and self.is_target(column, table):
            return "target_column"
        return None

    def excluded_columns(
        self,
        columns: list[str],
        *,
        table: str | None = None,
        context: str,
    ) -> list[dict[str, str]]:
        excluded: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for column in columns:
            reason = self.exclusion_reason(column, table=table, context=context)
            if reason is None:
                continue
            key = (column, reason)
            if key in seen:
                continue
            seen.add(key)
            excluded.append(ColumnExclusion(column, reason).as_dict())
        return excluded

    def safe_columns(
        self,
        columns: list[str],
        *,
        table: str | None = None,
        context: str,
    ) -> list[str]:
        return [
            column
            for column in columns
            if self.exclusion_reason(column, table=table, context=context) is None
        ]

    def _role(self, column: str, table: str | None) -> str | None:
        if table is not None:
            return self._column_roles_by_table.get(table, {}).get(column)
        for roles in self._column_roles_by_table.values():
            role = roles.get(column)
            if role is not None and role != "unknown":
                return role
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


__all__ = ["ColumnExclusion", "ColumnRolePolicy"]
