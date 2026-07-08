from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.eda.schemas import EdaEvidencePack, ResearchHypotheses
from kaggle_researcher.schemas import PlanData, RetrievedDocument


Priority = Literal["P0", "P1", "P2", "P3"]
Confidence = Literal["low", "medium", "high"]
FinalValidationMethod = Literal[
    "stratified_kfold",
    "kfold",
    "group_kfold",
    "stratified_group_kfold",
    "temporal_holdout",
    "temporal_cv",
    "ranking_group_cv",
    "custom_required",
]

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceRef = NonEmptyString


class FinalStrategyAction(BaseModel):
    action_id: NonEmptyString | None = None
    priority: Priority
    action: NonEmptyString
    reason: NonEmptyString

    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_hypothesis_ids: list[NonEmptyString] = Field(default_factory=list)

    source_claim: NonEmptyString | None = None
    source_refs: list[NonEmptyString] = Field(default_factory=list)
    eda_result_refs: list[EvidenceRef] = Field(default_factory=list)

    validation_strategy: FinalValidationMethod | None = None
    confidence: Confidence = "medium"
    limitations: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_strategy_links(self) -> "FinalStrategyAction":
        if not self.evidence_refs:
            raise ValueError("FinalStrategyAction.evidence_refs must not be empty")
        if not self.related_hypothesis_ids:
            raise ValueError("FinalStrategyAction.related_hypothesis_ids must not be empty")
        if not self.eda_result_refs:
            self.eda_result_refs = list(self.evidence_refs)
        return self


class FinalStrategySection(BaseModel):
    section_id: NonEmptyString
    title: NonEmptyString
    summary: NonEmptyString
    actions: list[FinalStrategyAction] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_hypothesis_ids: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_action_or_evidence(self) -> "FinalStrategySection":
        if not self.actions and not self.evidence_refs:
            raise ValueError(
                "FinalStrategySection must include actions or evidence_refs"
            )
        return self


class FinalStrategyResult(BaseModel):
    schema_version: str = "1.0"
    competition_id: NonEmptyString
    task_type: NonEmptyString | None = None
    metric: dict[str, Any] = Field(default_factory=dict)

    recommended_validation: FinalValidationMethod | None = None
    sections: list[FinalStrategySection] = Field(default_factory=list)
    actions: list[FinalStrategyAction] = Field(default_factory=list)

    source_to_hypothesis_links: list[dict[str, Any]] = Field(default_factory=list)
    hypothesis_to_eda_links: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[NonEmptyString] = Field(default_factory=list)
    models_used: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_actions(self) -> "FinalStrategyResult":
        if not self.actions and not any(section.actions for section in self.sections):
            raise ValueError("FinalStrategyResult must include at least one action")
        return self


async def synthesize_final_strategy(
    *,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    domain_patterns: list[dict[str, Any]],
    research_hypotheses: ResearchHypotheses,
    eda_evidence_pack: EdaEvidencePack,
    reasoning_outputs: dict[str, Any],
    client: DeepSeekClient,
    model: str,
) -> FinalStrategyResult:
    raise NotImplementedError("Final strategy synthesis is implemented in task 57.")


__all__ = [
    "FinalStrategyAction",
    "FinalStrategyResult",
    "FinalStrategySection",
    "FinalValidationMethod",
    "synthesize_final_strategy",
]
