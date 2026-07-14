from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from kaggle_researcher.contracts.ids import ExperimentId
from kaggle_researcher.contracts.normalization import normalize_contract_payload
from kaggle_researcher.contracts.validation import ConfidenceLevel, ReasoningBaseResult


class SkepticalReview(ReasoningBaseResult):
    contract_family: Literal["skeptical_review"] = "skeptical_review"
    schema_version: Literal["1.0"] = "1.0"
    confidence: ConfidenceLevel = "medium"
    unsupported_claims: list[str] = Field(default_factory=list)
    too_generic: list[str] = Field(default_factory=list)
    unnecessary_experiments: list[str] = Field(default_factory=list)
    approved_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    rejected_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    reviewed_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    revised_sections: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: object) -> object:
        return normalize_contract_payload(value, "ReviewResult")

    @model_validator(mode="after")
    def _validate_decisions(self) -> "SkepticalReview":
        approved = set(self.approved_experiment_ids)
        rejected = set(self.rejected_experiment_ids)
        overlap = approved & rejected
        if overlap:
            raise ValueError(f"approved and rejected experiment IDs overlap: {sorted(overlap)}")
        if self.reviewed_experiment_ids:
            unknown = (approved | rejected) - set(self.reviewed_experiment_ids)
            if unknown:
                raise ValueError(f"decision IDs are absent from reviewed_experiment_ids: {sorted(unknown)}")
        return self


ReviewResult = SkepticalReview

__all__ = ["ReviewResult", "SkepticalReview"]
