from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from kaggle_researcher.contracts.eda import (
    EdaEvidencePack,
    EdaRisk,
    EdaTestableHypothesis,
    EdaTask,
    EdaTaskPlan,
    SafetyConstraint,
    ValidationRequirement,
)
from kaggle_researcher.contracts.errors import (
    AmbiguousReferenceError,
    ContractIssue,
    UnknownReferenceError,
)
from kaggle_researcher.contracts.evidence import EvidenceRegistry, build_evidence_registry
from kaggle_researcher.contracts.experiments import ExperimentItem, ExperimentPlan
from kaggle_researcher.contracts.ids import (
    EdaTaskId,
    ExperimentId,
    HypothesisId,
    RiskId,
    SafetyConstraintId,
    ValidationRequirementId,
)
from kaggle_researcher.contracts.research import ResearchHypotheses, ResearchHypothesis
from kaggle_researcher.contracts.review import SkepticalReview


@dataclass(frozen=True)
class HypothesisRegistry:
    by_id: Mapping[HypothesisId, ResearchHypothesis | EdaTestableHypothesis]

    @classmethod
    def from_contract(cls, value: ResearchHypotheses) -> "HypothesisRegistry":
        return cls(_unique_mapping(value.hypotheses, "hypothesis_id", "hypothesis"))

    @classmethod
    def from_contracts(
        cls, research: ResearchHypotheses, eda: EdaEvidencePack
    ) -> "HypothesisRegistry":
        result = dict(_unique_mapping(
            [*research.hypotheses, *eda.testable_hypotheses],
            "hypothesis_id",
            "hypothesis",
        ))
        # HypothesisResult is an observation about an existing producer-owned
        # hypothesis. It may fill a missing legacy producer, but never redefines it.
        for observation in eda.hypothesis_results:
            result.setdefault(observation.hypothesis_id, observation)
        return cls(MappingProxyType(result))

    def bounded_prompt_ids(self, limit: int = 200) -> list[str]:
        return [str(value) for value in list(self.by_id)[:limit]]


@dataclass(frozen=True)
class EdaTaskRegistry:
    by_id: Mapping[EdaTaskId, EdaTask]

    @classmethod
    def from_contract(cls, value: EdaTaskPlan) -> "EdaTaskRegistry":
        return cls(_unique_mapping(value.eda_tasks, "task_id", "EDA task"))


@dataclass(frozen=True)
class ExperimentRegistry:
    by_id: Mapping[ExperimentId, ExperimentItem]
    approved_ids: frozenset[ExperimentId]
    rejected_ids: frozenset[ExperimentId]
    experiment_to_hypotheses: Mapping[ExperimentId, tuple[HypothesisId, ...]]

    @classmethod
    def from_contract(
        cls, plan: ExperimentPlan, review: SkepticalReview | None = None
    ) -> "ExperimentRegistry":
        by_id = _unique_mapping(plan.experiments, "experiment_id", "experiment")
        all_ids = frozenset(by_id)
        approved = frozenset(review.approved_experiment_ids) if review else all_ids
        rejected = frozenset(review.rejected_experiment_ids) if review else frozenset()
        reviewed = frozenset(review.reviewed_experiment_ids) if review else frozenset()
        unknown = (approved | rejected | reviewed) - all_ids
        if unknown:
            raise UnknownReferenceError(
                "Review references unknown experiments",
                issues=[
                    ContractIssue(
                        "review.experiment_ids", item, "planned experiment ID",
                        "unknown reference", "unknown",
                    )
                    for item in sorted(unknown)
                ],
            )
        overlap = approved & rejected
        if overlap:
            raise AmbiguousReferenceError(
                "Experiments cannot be approved and rejected",
                issues=[
                    ContractIssue(
                        "review.experiment_ids", item, "one review decision",
                        "conflicting decisions", "experiment",
                    )
                    for item in sorted(overlap)
                ],
            )
        if review is not None and not approved and not rejected and not reviewed:
            approved = all_ids
        mapping = {
            identifier: tuple(item.source_hypothesis_ids)
            for identifier, item in by_id.items()
        }
        return cls(by_id, approved, rejected, MappingProxyType(mapping))

    @property
    def all_experiment_ids(self) -> frozenset[ExperimentId]:
        return frozenset(self.by_id)

    @property
    def approved_experiment_ids(self) -> frozenset[ExperimentId]:
        return self.approved_ids

    @property
    def rejected_experiment_ids(self) -> frozenset[ExperimentId]:
        return self.rejected_ids

    @property
    def experiments(self) -> Mapping[ExperimentId, ExperimentItem]:
        return self.by_id

    def bounded_prompt_payload(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "experiment_id": str(identifier),
                "experiment": self.by_id[identifier].experiment,
                "source_hypothesis_ids": [
                    str(value) for value in self.experiment_to_hypotheses[identifier]
                ],
            }
            for identifier in list(self.by_id)[:limit]
        ]


@dataclass(frozen=True)
class ReviewRegistry:
    reviewed_ids: frozenset[ExperimentId]
    approved_ids: frozenset[ExperimentId]
    rejected_ids: frozenset[ExperimentId]

    @classmethod
    def from_contract(cls, review: SkepticalReview | None) -> "ReviewRegistry":
        if review is None:
            return cls(frozenset(), frozenset(), frozenset())
        return cls(
            frozenset(review.reviewed_experiment_ids),
            frozenset(review.approved_experiment_ids),
            frozenset(review.rejected_experiment_ids),
        )


@dataclass(frozen=True)
class RiskRegistry:
    by_id: Mapping[RiskId, EdaRisk]

    @classmethod
    def from_contract(cls, value: EdaEvidencePack) -> "RiskRegistry":
        return cls(_unique_mapping(value.eda_risks, "risk_id", "EDA risk"))


@dataclass(frozen=True)
class ValidationRequirementRegistry:
    by_id: Mapping[ValidationRequirementId, ValidationRequirement]

    @classmethod
    def from_contract(cls, value: EdaEvidencePack) -> "ValidationRequirementRegistry":
        return cls(_unique_mapping(
            value.validation_requirements,
            "validation_requirement_id",
            "validation requirement",
        ))


@dataclass(frozen=True)
class SafetyConstraintRegistry:
    by_id: Mapping[SafetyConstraintId, SafetyConstraint]

    @classmethod
    def from_contract(cls, value: EdaEvidencePack) -> "SafetyConstraintRegistry":
        return cls(_unique_mapping(
            value.safety_constraints,
            "safety_constraint_id",
            "safety constraint",
        ))


@dataclass(frozen=True)
class ContractRegistries:
    hypotheses: HypothesisRegistry
    tasks: EdaTaskRegistry
    evidence: EvidenceRegistry
    experiments: ExperimentRegistry
    reviews: ReviewRegistry
    risks: RiskRegistry
    validation_requirements: ValidationRequirementRegistry
    safety_constraints: SafetyConstraintRegistry

    def namespace_for(self, value: str) -> str | None:
        namespaces = {
            "hypothesis": self.hypotheses.by_id,
            "experiment": self.experiments.by_id,
            "risk": self.risks.by_id,
            "validation_requirement": self.validation_requirements.by_id,
            "safety_constraint": self.safety_constraints.by_id,
            "evidence": (
                set(self.evidence.ids("eda_evidence"))
                | set(self.evidence.ids("reasoning_evidence"))
                | set(self.evidence.ids("source_claim"))
                | set(self.evidence.ids("synthetic_inference"))
            ),
        }
        return classify_namespace(value, **namespaces)


def build_contract_registries(
    *,
    research: Any,
    eda: Any,
    reasoning: Any | None = None,
) -> ContractRegistries:
    """Build the sole production registry bundle from typed stage results.

    `Any` appears only to avoid an import cycle with orchestration result dataclasses;
    every accessed attribute is a canonical typed contract field.
    """

    hypotheses = research.hypotheses
    task_plan = research.task_plan
    evidence_pack = eda.evidence_pack
    experiment_plan = reasoning.experiments if reasoning and reasoning.experiments else ExperimentPlan()
    review = reasoning.review if reasoning else None
    source_ids = [document.id for document in research.retrieved_documents]
    hypothesis_ids = [item.hypothesis_id for item in hypotheses.hypotheses]
    evidence = build_evidence_registry(
        evidence_pack,
        source_ids=source_ids,
        hypothesis_ids=hypothesis_ids,
        reasoning_ids=("metric_result", "validation_result", "leakage_result", "leaderboard_audit"),
    )
    return ContractRegistries(
        hypotheses=HypothesisRegistry.from_contracts(hypotheses, evidence_pack),
        tasks=EdaTaskRegistry.from_contract(task_plan),
        evidence=evidence,
        experiments=ExperimentRegistry.from_contract(experiment_plan, review),
        reviews=ReviewRegistry.from_contract(review),
        risks=RiskRegistry.from_contract(evidence_pack),
        validation_requirements=ValidationRequirementRegistry.from_contract(evidence_pack),
        safety_constraints=SafetyConstraintRegistry.from_contract(evidence_pack),
    )


def classify_namespace(value: str, **registries: Iterable[str]) -> str | None:
    matches = [name for name, identifiers in registries.items() if value in identifiers]
    if len(matches) > 1:
        raise AmbiguousReferenceError(f"Reference {value!r} exists in multiple namespaces: {matches}")
    return matches[0] if matches else None


def _unique_mapping(items: Iterable[Any], field: str, label: str) -> Mapping[Any, Any]:
    result: dict[Any, Any] = {}
    for item in items:
        identifier = getattr(item, field)
        if identifier is None:
            raise ValueError(f"Canonical {label} requires {field}")
        if identifier in result:
            raise ValueError(f"Duplicate {label} ID: {identifier!r}")
        result[identifier] = item
    return MappingProxyType(result)


__all__ = [
    "ContractRegistries", "EdaTaskRegistry", "ExperimentRegistry",
    "HypothesisRegistry", "ReviewRegistry", "RiskRegistry",
    "SafetyConstraintRegistry", "ValidationRequirementRegistry",
    "build_contract_registries", "classify_namespace",
]
