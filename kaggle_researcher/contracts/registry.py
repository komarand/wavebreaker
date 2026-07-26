from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.eda_task_plan import EdaTaskPlan
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceReferenceManifest,
    PublishedEdaEvidenceBundle,
)
from kaggle_researcher.contracts.manifest import RunManifest
from kaggle_researcher.contracts.research_hypotheses import ResearchHypotheses
from kaggle_researcher.contracts.synthesis_context import FinalSynthesisContext
from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.contracts.final_strategy_protocol import (
    StrategyRenderingDraft,
    StrategySelectionDraft,
    StrategySkeleton,
)
from kaggle_researcher.contracts.errors import (
    ContractIssue,
    DuplicateContractRegistrationError,
    UnknownContractFamilyError,
    UnsupportedSchemaVersionError,
)
from kaggle_researcher.contracts.versions import CURRENT_CONTRACT_VERSIONS
from kaggle_researcher.eda.schemas import EdaEvidencePack
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ReviewResult,
    ValidationResult,
)


class ContractHeader(BaseModel):
    """Minimal strict dispatch header; contract-specific fields are ignored here."""

    model_config = ConfigDict(extra="ignore", strict=True, str_strip_whitespace=True)

    contract_family: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)


class ContractRegistry(Mapping[tuple[str, str], type[ContractModel]]):
    """Deterministically ordered registry of public serialized contracts."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], type[ContractModel]] = {}
        self._current: dict[str, str] = {}

    def __getitem__(self, key: tuple[str, str]) -> type[ContractModel]:
        return self._models[key]

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(sorted(self._models))

    def __len__(self) -> int:
        return len(self._models)

    def register(
        self,
        contract_family: str,
        schema_version: str,
        model: type[ContractModel],
        *,
        current: bool = False,
    ) -> None:
        key = (contract_family, schema_version)
        if key in self._models:
            raise DuplicateContractRegistrationError(
                "Duplicate contract registration",
                issues=(ContractIssue(
                    "contract_registry", key, "unique family/version registration",
                    "duplicate contract registration", issue_type="duplicate_registration",
                ),),
                contract=contract_family,
            )
        if not isinstance(model, type) or not issubclass(model, ContractModel):
            raise TypeError("Registered contract model must inherit ContractModel")
        self._models[key] = model
        if current or contract_family not in self._current:
            self._current[contract_family] = schema_version

    def resolve(self, contract_family: str, schema_version: str) -> type[ContractModel]:
        families = {family for family, _ in self._models}
        if contract_family not in families:
            raise UnknownContractFamilyError(
                f"Unknown contract family {contract_family!r}",
                issues=(ContractIssue(
                    "contract_family", contract_family, "registered contract family",
                    "unknown contract family", issue_type="unknown_family",
                ),),
                contract=contract_family,
            )
        try:
            return self._models[(contract_family, schema_version)]
        except KeyError as exc:
            supported = sorted(version for family, version in self._models if family == contract_family)
            raise UnsupportedSchemaVersionError(
                f"Unsupported {contract_family} schema version {schema_version!r}",
                issues=(ContractIssue(
                    "schema_version", schema_version, f"one of {supported}",
                    "unsupported schema version", issue_type="unsupported_version",
                ),),
                contract=contract_family,
            ) from exc

    def current_version(self, contract_family: str) -> str:
        if contract_family not in self._current:
            self.resolve(contract_family, "")
        return self._current[contract_family]

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({family for family, _ in self._models}))


def _build_contract_registry() -> ContractRegistry:
    registry = ContractRegistry()
    registrations: tuple[tuple[str, str, type[ContractModel]], ...] = (
        ("eda_evidence_pack", "1.0", EdaEvidencePack),
        ("eda_task_plan", "1.0", EdaTaskPlan),
        ("evidence_reference_manifest", "1.0", EvidenceReferenceManifest),
        ("experiment_plan", "1.0", ExperimentPlan),
        ("final_strategy", "1.0", FinalStrategyResult),
        ("final_strategy", "2.0", FinalStrategyResult),
        ("final_synthesis_context", "1.0", FinalSynthesisContext),
        ("leaderboard_audit_result", "1.0", LeaderboardAuditResult),
        ("leakage_risk_result", "1.0", LeakageRiskResult),
        ("metric_result", "1.0", MetricResult),
        ("published_eda_evidence_bundle", "1.0", PublishedEdaEvidenceBundle),
        ("research_hypotheses", "1.0", ResearchHypotheses),
        ("run_manifest", "1.0", RunManifest),
        ("skeptical_review", "1.0", ReviewResult),
        ("strategy_rendering_draft", "2.0", StrategyRenderingDraft),
        ("strategy_selection_draft", "2.0", StrategySelectionDraft),
        ("strategy_skeleton", "2.0", StrategySkeleton),
        ("validation_result", "1.0", ValidationResult),
    )
    for family, version, model in registrations:
        registry.register(
            family, version, model,
            current=CURRENT_CONTRACT_VERSIONS.get(family) == version,
        )
    return registry


CONTRACT_REGISTRY = _build_contract_registry()


def resolve_contract_model(contract_family: str, schema_version: str) -> type[ContractModel]:
    return CONTRACT_REGISTRY.resolve(contract_family, schema_version)


@dataclass(frozen=True)
class ContractDefinition:
    contract_id: str
    producer_stage: str
    consumer_stages: tuple[str, ...]
    model: type[BaseModel] | None
    schema_version: str | None
    reference_fields: tuple[str, ...] = ()
    artifact_name: str | None = None
    nullable_fields: tuple[str, ...] = ()
    collection_fields: tuple[str, ...] = ()
    renderer_consumers: tuple[str, ...] = ()
    migration_support: str = "none"


CONTRACT_DEFINITIONS = (
    ContractDefinition(
        "research_hypotheses", "research_scout", ("eda_engine", "final_strategy"),
        ResearchHypotheses, "1.0", ("hypotheses[].hypothesis_id", "hypotheses[].source_refs"),
        "research_hypotheses.json", ("created_at", "hypotheses[].rationale"),
        ("hypotheses", "eda_tasks", "structured_findings", "scout_limitations"),
        migration_support="legacy unversioned to 1.0",
    ),
    ContractDefinition(
        "eda_task_plan", "research_scout", ("eda_engine",), EdaTaskPlan, "1.0",
        ("eda_tasks[].task_id", "eda_tasks[].related_hypothesis_ids", "hypothesis_index"),
        "eda_task_plan.json", ("task_type",),
        ("eda_tasks", "recommended_module_sequence", "recommended_human_checklist", "blocking_tasks"),
        migration_support="legacy unversioned to 1.0",
    ),
    ContractDefinition(
        "eda_evidence_pack", "eda_engine", ("reasoning_context", "final_strategy", "artifact_validation"),
        EdaEvidencePack, "1.0", ("*.evidence_refs", "*.source_refs", "*.related_hypothesis_ids"),
        "eda_evidence_pack.json", collection_fields=("safety_constraints", "validation_requirements", "testable_hypotheses", "warnings", "limitations"),
        renderer_consumers=("eda_summary", "final_strategy"),
    ),
    ContractDefinition(
        "evidence_reference_manifest", "eda_engine",
        ("reasoning_context", "final_strategy", "artifact_validation"),
        EvidenceReferenceManifest, "1.0", ("entries[].ref",),
        "evidence_reference_manifest.json", collection_fields=("entries", "conflicts"),
    ),
    ContractDefinition(
        "published_eda_evidence_bundle", "eda_engine",
        ("reasoning_context", "final_strategy", "artifact_validation"),
        PublishedEdaEvidenceBundle, "1.0", ("evidence_manifest.entries[].ref",),
        "published_eda_evidence_bundle.json", collection_fields=("warnings", "limitations"),
    ),
    ContractDefinition(
        "validation_result", "validation_architect", ("experiment_planner", "leaderboard_auditor", "final_strategy"),
        ValidationResult, "1.0", ("evidence_ids",), "validation_result.json",
        ("secondary_validation",), ("evidence_ids", "failure_modes", "do_not_use", "policy_notes"),
        ("final_report",),
    ),
    ContractDefinition(
        "metric_result", "metric_specialist", ("experiment_planner", "final_strategy"),
        MetricResult, "1.0", ("evidence_ids",), "metric_result.json",
        collection_fields=("evidence_ids",), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "leakage_risk_result", "leakage_risk_analyst", ("experiment_planner", "final_strategy"),
        LeakageRiskResult, "1.0", ("evidence_ids",), "leakage_result.json",
        collection_fields=("evidence_ids", "possible_issues", "recommended_checks"), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "leaderboard_audit_result", "leaderboard_auditor", ("final_strategy",),
        LeaderboardAuditResult, "1.0", ("evidence_ids",), "leaderboard_audit.json",
        collection_fields=("evidence_ids", "warnings"), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "experiment_plan", "experiment_planner", ("skeptical_reviewer", "final_strategy"),
        ExperimentPlan, "1.0", ("experiments[].experiment_id", "experiments[].source_hypothesis_ids", "experiments[].evidence_ids"), "experiment_plan.json",
        collection_fields=("[].evidence_ids",), renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "skeptical_review", "skeptical_reviewer", ("final_strategy",), ReviewResult, "1.0",
        ("evidence_ids", "reviewed_experiment_ids", "approved_experiment_ids", "rejected_experiment_ids"), "skeptical_review.json",
        collection_fields=("evidence_ids", "unsupported_claims", "too_generic", "unnecessary_experiments", "approved_experiment_ids", "rejected_experiment_ids"),
        renderer_consumers=("final_report",),
    ),
    ContractDefinition(
        "final_strategy", "final_strategy", ("final_report", "artifact_validation"),
        FinalStrategyResult, "1.0",
        (
            "actions[].evidence_refs", "actions[].related_hypothesis_ids",
            "actions[].hypothesis_ids", "actions[].experiment_ids", "actions[].source_refs",
            "actions[].risk_ids", "actions[].validation_requirement_ids",
            "actions[].safety_constraint_ids", "acknowledged_risk_ids",
            "selected_validation_requirement_ids", "enforced_safety_constraint_ids",
        ),
        "final_strategy.json", ("task_type", "recommended_validation"),
        ("sections", "actions", "limitations"), ("final_report",),
    ),
    ContractDefinition(
        "run_manifest", "full_run", ("resume", "run_summary"), RunManifest, "1.0",
        ("stages.*.outputs",), "run_manifest.json",
    ),
    ContractDefinition(
        "full_run_result", "full_run", ("cli",), None, None,
        ("run_dir", "manifest_path", "final_strategy_path", "final_report_path"),
    ),
    ContractDefinition(
        "final_report", "final_report", ("artifact_validation", "human"), None, None,
        artifact_name="final_report.md",
    ),
)


def contract_by_id(contract_id: str) -> ContractDefinition:
    try:
        return next(item for item in CONTRACT_DEFINITIONS if item.contract_id == contract_id)
    except StopIteration as exc:
        raise KeyError(contract_id) from exc
