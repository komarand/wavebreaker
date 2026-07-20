from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompetitionPreset(BaseModel):
    """Typed competition hints consumed opportunistically by generic EDA modules."""

    model_config = ConfigDict(frozen=True)

    preset_id: str
    competition_ids: tuple[str, ...] = Field(default_factory=tuple)
    preferred_id_columns: tuple[str, ...] = Field(default_factory=tuple)
    preferred_target_columns: tuple[str, ...] = Field(default_factory=tuple)
    preferred_time_columns: tuple[str, ...] = Field(default_factory=tuple)
    preferred_prediction_columns: tuple[str, ...] = Field(default_factory=tuple)
    table_name_patterns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    metric_aliases: dict[str, str] = Field(default_factory=dict)
    known_validation_hints: tuple[str, ...] = Field(default_factory=tuple)

    def canonical_metric_name(self, metric_name: str | None) -> str | None:
        if metric_name is None:
            return None
        normalized = _normalize_hint(metric_name)
        return self.metric_aliases.get(normalized, metric_name)

    def matches_competition_id(self, competition_id: str | None) -> bool:
        if competition_id is None:
            return False
        normalized = _normalize_hint(competition_id)
        return any(_normalize_hint(candidate) == normalized for candidate in self.competition_ids)


def _normalize_hint(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["CompetitionPreset"]
