from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.ids import EvidenceId
from kaggle_researcher.contracts.normalization import normalize_contract_payload


ConfidenceLevel = Literal["low", "medium", "high"]


class ReasoningBaseResult(ContractModel):
    confidence: ConfidenceLevel
    evidence_ids: list[EvidenceId] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: object) -> object:
        return normalize_contract_payload(value, cls.__name__)


class ValidationPolicy(ContractModel):
    method: str = Field(min_length=1)
    reason: str | None = None
    n_splits: int | None = Field(default=None, ge=2)
    shuffle: bool | None = None
    group_column: str | None = None
    split_column: str | None = None


class ValidationResult(ReasoningBaseResult):
    contract_family: Literal["validation_result"] = "validation_result"
    schema_version: Literal["1.0"] = "1.0"
    recommended_cv: str
    validation_risk: ConfidenceLevel
    likely_split: str
    failure_modes: list[str] = Field(default_factory=list)
    reasoning: str
    primary_validation: ValidationPolicy
    secondary_validation: ValidationPolicy | None = None
    do_not_use: list[str] = Field(default_factory=list)
    policy_enforced: StrictBool = False
    policy_notes: list[str] = Field(default_factory=list)


class LeakageRiskResult(ReasoningBaseResult):
    contract_family: Literal["leakage_risk_result"] = "leakage_risk_result"
    schema_version: Literal["1.0"] = "1.0"
    risk_level: ConfidenceLevel
    possible_issues: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)


class MetricResult(ReasoningBaseResult):
    contract_family: Literal["metric_result"] = "metric_result"
    schema_version: Literal["1.0"] = "1.0"
    metric_explanation: str
    needs_calibration: StrictBool
    rank_averaging_useful: StrictBool
    threshold_search_needed: StrictBool
    surrogate_loss_suggestion: str


class LeaderboardAuditResult(ReasoningBaseResult):
    contract_family: Literal["leaderboard_audit_result"] = "leaderboard_audit_result"
    schema_version: Literal["1.0"] = "1.0"
    shake_up_risk: ConfidenceLevel
    submission_selection_rule: str
    public_lb_trust: str
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "ConfidenceLevel",
    "LeaderboardAuditResult",
    "LeakageRiskResult",
    "MetricResult",
    "ReasoningBaseResult",
    "ValidationPolicy",
    "ValidationResult",
]
