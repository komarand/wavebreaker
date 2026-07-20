from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.final_strategy_compilation import (
    FinalStrategyCompilationDiagnostics,
    UnresolvedFinalStrategyReferenceError,
)
from kaggle_researcher.contracts.ids import EvidenceId, ExperimentId, HypothesisId
from kaggle_researcher.contracts.normalization import normalize_contract_payload

if TYPE_CHECKING:
    from kaggle_researcher.contracts.review import ReviewResult
    from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult


PriorityLevel = Literal["P0", "P1", "P2", "P3"]


class ExperimentItem(ContractModel):
    # Raw planner responses may omit identity; ExperimentPlan is the canonical
    # serialized boundary and requires every identity to be assigned.
    experiment_id: ExperimentId | None = None
    source_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    priority: PriorityLevel
    experiment: str = Field(min_length=1)
    why: str
    cost: str
    expected_gain: str
    risk: str
    evidence_ids: list[EvidenceId] = Field(default_factory=list)
    dependencies: list[ExperimentId] = Field(default_factory=list)

    @field_validator("experiment_id", mode="before")
    @classmethod
    def _empty_draft_identity_is_missing(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="before")
    @classmethod
    def _normalize_registered_collections(cls, value: object) -> object:
        return normalize_contract_payload(value, cls.__name__)


class ExperimentPlan(ContractModel):
    contract_family: Literal["experiment_plan"] = "experiment_plan"
    schema_version: str = "1.0"
    experiments: list[ExperimentItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_graph(self) -> "ExperimentPlan":
        if any(item.experiment_id is None for item in self.experiments):
            raise ValueError("Canonical experiment plan requires every experiment_id")
        ids = [item.experiment_id for item in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment plan contains duplicate experiment_id values")
        known = set(ids)
        graph: dict[ExperimentId, list[ExperimentId]] = {}
        for item in self.experiments:
            assert item.experiment_id is not None
            if item.experiment_id in item.source_hypothesis_ids:
                raise ValueError("experiment_id is never a source hypothesis ID")
            if item.experiment_id in item.dependencies:
                raise ValueError(f"Experiment {item.experiment_id!r} depends on itself")
            unknown = set(item.dependencies) - known
            if unknown:
                raise ValueError(f"Experiment {item.experiment_id!r} has unknown dependencies: {sorted(unknown)}")
            graph[item.experiment_id] = item.dependencies
        _assert_experiment_graph_acyclic(graph)
        return self


def _assert_experiment_graph_acyclic(graph: Mapping[ExperimentId, list[ExperimentId]]) -> None:
    visiting: set[ExperimentId] = set()
    visited: set[ExperimentId] = set()
    def visit(node: ExperimentId) -> None:
        if node in visiting:
            raise ValueError(f"Experiment dependency cycle contains {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)
    for node in graph:
        visit(node)


FORBIDDEN_CONTEXT_LABELS = frozenset({
    "approved_experiments",
    "rejected_experiments",
    "experiment_plan",
    "reasoning_outputs",
    "final_strategy_context",
    "skeptical_review",
})


@dataclass(frozen=True)
class ReferenceRegistries:
    evidence_ids: frozenset[str] = frozenset()
    eda_evidence_ids: frozenset[str] = frozenset()
    source_ids: frozenset[str] = frozenset()
    hypothesis_ids: frozenset[str] = frozenset()
    experiment_ids: frozenset[str] = frozenset()
    approved_experiment_ids: frozenset[str] = frozenset()
    rejected_experiment_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ReferenceIssue:
    field_path: str
    expected_namespace: str
    invalid_value: str
    actual_namespace: str | None = None
    reason: str = "unknown_reference"


@dataclass(frozen=True)
class AppliedReferenceRepair:
    field_path: str
    original_id: str
    replacement_id: str


@dataclass(frozen=True)
class FinalReferenceRepairResult:
    result: FinalStrategyResult
    applied_repairs: tuple[AppliedReferenceRepair, ...] = ()


class CrossNamespaceReferenceError(UnresolvedFinalStrategyReferenceError):
    def __init__(self, issues: list[ReferenceIssue]) -> None:
        self.stage = "final_strategy"
        self.contract = "final_strategy_references"
        self.issues = tuple(issues)
        self.field_paths = tuple(issue.field_path for issue in issues)
        self.invalid_ids = tuple(issue.invalid_value for issue in issues)
        self.expected_namespaces = tuple(issue.expected_namespace for issue in issues)
        self.actual_namespaces = tuple(issue.actual_namespace for issue in issues)
        self.recoverable = True
        self.suggested_rerun_stage = "final_strategy"
        details = "; ".join(_describe_reference_issue(issue) for issue in issues[:8])
        super().__init__(
            f"Final strategy contains unresolved cross-namespace references: {details}",
            phase="reference_resolution",
            diagnostics=FinalStrategyCompilationDiagnostics(
                phase="reference_resolution",
                initial_reference_issues=len(issues),
                unresolved_references=len(issues),
            ),
        )


def build_experiment_registry(
    experiments: list[ExperimentItem] | list[dict[str, Any]],
    review: ReviewResult | dict[str, Any] | None = None,
) -> ExperimentRegistry:
    from kaggle_researcher.contracts.registries import ExperimentRegistry
    from kaggle_researcher.contracts.review import ReviewResult
    parsed = [
        item if isinstance(item, ExperimentItem) else ExperimentItem.model_validate(item)
        for item in experiments
    ]
    review_result = (
        review if isinstance(review, ReviewResult) else ReviewResult.model_validate(review)
        if review is not None else None
    )
    return ExperimentRegistry.from_contract(ExperimentPlan(experiments=parsed), review_result)


def reference_registries(
    registry: ExperimentRegistry,
    *,
    hypothesis_ids: set[str] | frozenset[str],
    evidence_ids: set[str] | frozenset[str] = frozenset(),
    eda_evidence_ids: set[str] | frozenset[str] = frozenset(),
    source_ids: set[str] | frozenset[str] = frozenset(),
) -> ReferenceRegistries:
    return ReferenceRegistries(
        evidence_ids=frozenset(evidence_ids),
        eda_evidence_ids=frozenset(eda_evidence_ids),
        source_ids=frozenset(source_ids),
        hypothesis_ids=frozenset(hypothesis_ids),
        experiment_ids=registry.all_experiment_ids,
        approved_experiment_ids=registry.approved_experiment_ids,
        rejected_experiment_ids=registry.rejected_experiment_ids,
    )


def validate_final_strategy_namespaces(
    result: FinalStrategyResult,
    registries: ReferenceRegistries,
) -> list[ReferenceIssue]:
    issues: list[ReferenceIssue] = []
    for path, action in _iter_actions(result):
        for reference_id in action.evidence_refs:
            if registries.evidence_ids and reference_id not in registries.evidence_ids:
                issues.append(ReferenceIssue(
                    f"{path}.evidence_refs", "evidence", reference_id,
                    _actual_namespace(reference_id, registries),
                    _reference_reason(reference_id, registries),
                ))
        for reference_id in action.eda_result_refs:
            if registries.eda_evidence_ids and reference_id not in registries.eda_evidence_ids:
                issues.append(ReferenceIssue(
                    f"{path}.eda_result_refs", "eda_evidence", reference_id,
                    _actual_namespace(reference_id, registries),
                    _reference_reason(reference_id, registries),
                ))
        for reference_id in action.source_refs:
            if reference_id not in registries.source_ids:
                issues.append(ReferenceIssue(
                    f"{path}.source_refs", "source", reference_id,
                    _actual_namespace(reference_id, registries), "unknown_source",
                ))
        for reference_id in action.experiment_ids:
            if reference_id in registries.rejected_experiment_ids:
                issues.append(ReferenceIssue(
                    f"{path}.experiment_ids", "approved_experiment", reference_id,
                    "rejected_experiment", "rejected_experiment",
                ))
            elif reference_id not in registries.approved_experiment_ids:
                issues.append(ReferenceIssue(
                    f"{path}.experiment_ids", "approved_experiment", reference_id,
                    _actual_namespace(reference_id, registries), "unknown_or_unapproved_experiment",
                ))
        hypothesis_ids = set(action.related_hypothesis_ids) | set(action.hypothesis_ids)
        for reference_id in sorted(hypothesis_ids - registries.hypothesis_ids):
            issues.append(ReferenceIssue(
                f"{path}.hypothesis_ids", "hypothesis", reference_id,
                _actual_namespace(reference_id, registries), "unknown_hypothesis",
            ))
    for section_index, section in enumerate(result.sections):
        for reference_id in section.evidence_refs:
            if registries.evidence_ids and reference_id not in registries.evidence_ids:
                issues.append(ReferenceIssue(
                    f"sections[{section_index}].evidence_refs", "evidence", reference_id,
                    _actual_namespace(reference_id, registries),
                    _reference_reason(reference_id, registries),
                ))
    return issues


def repair_final_experiment_references(
    result: FinalStrategyResult,
    registry: ExperimentRegistry,
) -> FinalReferenceRepairResult:
    repaired = result.model_copy(deep=True)
    hypothesis_to_approved: dict[str, list[str]] = {}
    for experiment_id in registry.approved_experiment_ids:
        for hypothesis_id in registry.experiment_to_hypotheses.get(experiment_id, ()):
            hypothesis_to_approved.setdefault(hypothesis_id, []).append(experiment_id)

    applied: list[AppliedReferenceRepair] = []
    for path, action in _iter_actions(repaired):
        normalized: list[str] = []
        for reference_id in action.experiment_ids:
            replacement = reference_id
            if reference_id not in registry.all_experiment_ids:
                candidates = sorted(hypothesis_to_approved.get(reference_id, ()))
                if len(candidates) == 1:
                    replacement = candidates[0]
                    applied.append(AppliedReferenceRepair(
                        f"{path}.experiment_ids", reference_id, replacement
                    ))
            if replacement not in normalized:
                normalized.append(replacement)
        action.experiment_ids = normalized
    return FinalReferenceRepairResult(repaired, tuple(applied))


def approved_experiment_summary(registry: ExperimentRegistry) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": experiment_id,
            "title": registry.experiments[experiment_id].experiment,
            "source_hypothesis_ids": list(registry.experiment_to_hypotheses[experiment_id]),
            "evidence_ids": list(registry.experiments[experiment_id].evidence_ids),
            "review_status": "approved",
        }
        for experiment_id in sorted(registry.approved_experiment_ids)
    ]


def _iter_actions(result: FinalStrategyResult):
    for index, action in enumerate(result.actions):
        yield f"actions[{index}]", action


def _actual_namespace(reference_id: str, registries: ReferenceRegistries) -> str | None:
    if reference_id in FORBIDDEN_CONTEXT_LABELS:
        return "context_label"
    if reference_id in registries.hypothesis_ids:
        return "hypothesis"
    if reference_id in registries.evidence_ids:
        return "evidence"
    if reference_id in registries.source_ids:
        return "source"
    if reference_id in registries.rejected_experiment_ids:
        return "rejected_experiment"
    if reference_id in registries.approved_experiment_ids:
        return "approved_experiment"
    if reference_id in registries.experiment_ids:
        return "unapproved_experiment"
    return None


def _reference_reason(reference_id: str, registries: ReferenceRegistries) -> str:
    if reference_id in FORBIDDEN_CONTEXT_LABELS:
        return "context_label_not_reference"
    if _actual_namespace(reference_id, registries) is not None:
        return "cross_namespace"
    return "unknown"


def _describe_reference_issue(issue: ReferenceIssue) -> str:
    prefix = f"{issue.field_path} contains {issue.invalid_value!r}"
    if issue.reason == "context_label_not_reference":
        return (
            f"{prefix}: this is a context collection name, not an evidence reference; "
            "use concrete experiment_ids when referring to approved experiments"
        )
    return (
        f"{prefix}, expected {issue.expected_namespace}, "
        f"actual {issue.actual_namespace or 'unknown'} ({issue.reason})"
    )


# Deprecated import location. The implementation and identity are canonical in
# contracts.registries; this module retains only the historical import surface.
def __getattr__(name: str) -> Any:
    if name == "ExperimentRegistry":
        from kaggle_researcher.contracts.registries import ExperimentRegistry
        return ExperimentRegistry
    raise AttributeError(name)
