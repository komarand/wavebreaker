"""Canonical inter-stage contracts for the Kaggle research pipeline."""

from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    FinalStageResult,
    ReasoningStageResult,
    ResearchStageResult,
    load_eda_evidence_pack,
    load_eda_task_plan,
    load_experiment_plan,
    load_final_strategy,
    load_research_hypotheses,
    load_skeptical_review,
    load_validation_result,
    validate_research_artifact_bundle,
)
from kaggle_researcher.contracts.action_support import (
    ActionReferenceResolutionDiagnostics,
    ActionSupportDecision,
    FinalStrategyCompilationContext,
    FinalStrategyCompilationReport,
    UnsupportedFinalStrategyActionError,
    compile_final_strategy_action_support,
    enforce_action_evidence_support,
)
from kaggle_researcher.contracts.action_canonicalization import (
    ActionCanonicalizationDiagnostics,
    FinalStrategyActionCanonicalizationError,
    SectionActionMembership,
    canonicalize_final_strategy_actions,
)
from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.composite_reference_resolution import (
    CompositeReferenceResolutionDiagnostics,
    resolve_composite_action_references,
    resolve_final_strategy_composite_references,
)
from kaggle_researcher.contracts.bundle_validation import (
    validate_final_synthesis_bundle,
    validate_reasoning_artifact_bundle,
)
from kaggle_researcher.contracts.eda import EdaEvidencePack, EdaTask, EdaTaskPlan, HypothesisIndexEntry
from kaggle_researcher.contracts.errors import *
from kaggle_researcher.contracts.experiments import ExperimentItem, ExperimentPlan
from kaggle_researcher.contracts.final_strategy import (
    FinalStrategyAction,
    FinalStrategyResult,
    FinalStrategySection,
)
from kaggle_researcher.contracts.final_strategy_draft import (
    FinalStrategyActionDraft,
    FinalStrategyDraft,
    FinalStrategyDraftReferenceError,
    FinalStrategyDraftReferenceIssue,
    FinalStrategySectionDraft,
    FinalStrategySupportRef,
    normalize_legacy_final_strategy_to_draft,
)
from kaggle_researcher.contracts.ids import *
from kaggle_researcher.contracts.manifest import RunManifest, StageManifestEntry
from kaggle_researcher.contracts.hypothesis_reference_migration import (
    HypothesisReferenceMigrationDiagnostics,
    migrate_final_strategy_hypothesis_references,
    migrate_hypothesis_references,
)
from kaggle_researcher.contracts.migration import (
    EdaTaskPlanMigrationResult,
    HypothesisMigrationResult,
    MigrationResult,
    migrate_eda_task_plan_payload,
    migrate_research_hypotheses_payload,
)
from kaggle_researcher.contracts.research import ResearchHypothesis, ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import (
    ContractIssue as ResearchToEdaContractIssue,
    ResearchToEdaContractError,
    ResearchToEdaContractValidationResult,
    STABLE_ERROR_CODES as RESEARCH_TO_EDA_ISSUE_CODES,
    require_valid_research_to_eda_contract,
    validate_research_to_eda_contract,
)
from kaggle_researcher.contracts.review import ReviewResult, SkepticalReview
from kaggle_researcher.contracts.repair import BoundaryValidationResult, validate_with_one_repair
from kaggle_researcher.contracts.reference_catalog import (
    REFERENCE_NAMESPACES,
    ReferenceCatalog,
    ReferenceCatalogDiagnostic,
    ReferenceCatalogEntry,
    ReferenceNamespace,
    ReferenceResolution,
    build_final_strategy_reference_catalog,
)
from kaggle_researcher.contracts.validation import ValidationPolicy, ValidationResult
from kaggle_researcher.contracts.versions import (
    CURRENT_CONTRACT_VERSIONS,
    CURRENT_SCHEMA_VERSION,
    ContractFamily,
)


__all__ = [
    "CURRENT_CONTRACT_VERSIONS", "CURRENT_SCHEMA_VERSION", "ContractFamily",
    "CompositeReferenceResolutionDiagnostics",
    "ContractModel", "EdaEvidencePack", "EdaStageResult", "EdaTask", "EdaTaskPlan",
    "ActionReferenceResolutionDiagnostics", "ActionSupportDecision",
    "ActionCanonicalizationDiagnostics",
    "BoundaryValidationResult",
    "EdaTaskPlanMigrationResult", "ExperimentItem", "ExperimentPlan",
    "ExperimentPlanningContext", "FinalStageResult", "FinalStrategyAction",
    "FinalStrategyActionDraft", "FinalStrategyDraft",
    "FinalStrategyDraftReferenceError", "FinalStrategyDraftReferenceIssue",
    "FinalStrategyCompilationContext", "FinalStrategyCompilationReport",
    "FinalStrategyResult", "FinalStrategySection", "FinalStrategySectionDraft",
    "FinalStrategySupportRef", "FinalSynthesisContext",
    "FinalStrategyActionCanonicalizationError",
    "HypothesisIndexEntry", "HypothesisMigrationResult", "MigrationResult",
    "HypothesisReferenceMigrationDiagnostics",
    "ReasoningStageResult", "ResearchHypothesis", "ResearchHypotheses",
    "ResearchToEdaContractError", "ResearchToEdaContractIssue",
    "ResearchToEdaContractValidationResult",
    "RESEARCH_TO_EDA_ISSUE_CODES",
    "REFERENCE_NAMESPACES", "ReferenceCatalog", "ReferenceCatalogDiagnostic",
    "ReferenceCatalogEntry", "ReferenceNamespace", "ReferenceResolution",
    "ResearchStageResult", "ReviewResult", "RunManifest", "SkepticalReview",
    "StageManifestEntry", "ValidationPolicy", "ValidationResult",
    "SectionActionMembership",
    "UnsupportedFinalStrategyActionError",
    "load_eda_evidence_pack", "load_eda_task_plan", "load_experiment_plan",
    "load_final_strategy", "load_research_hypotheses", "load_skeptical_review",
    "load_validation_result", "migrate_eda_task_plan_payload",
    "migrate_research_hypotheses_payload", "validate_research_artifact_bundle",
    "migrate_final_strategy_hypothesis_references", "migrate_hypothesis_references",
    "validate_final_synthesis_bundle", "validate_reasoning_artifact_bundle",
    "validate_with_one_repair",
    "build_final_strategy_reference_catalog",
    "resolve_composite_action_references", "resolve_final_strategy_composite_references",
    "compile_final_strategy_action_support", "enforce_action_evidence_support",
    "canonicalize_final_strategy_actions",
    "normalize_legacy_final_strategy_to_draft",
    "require_valid_research_to_eda_contract", "validate_research_to_eda_contract",
]


def __getattr__(name: str):
    if name in {"ExperimentPlanningContext", "FinalSynthesisContext"}:
        from kaggle_researcher.contracts import synthesis_context
        return getattr(synthesis_context, name)
    raise AttributeError(name)
