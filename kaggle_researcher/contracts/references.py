from __future__ import annotations

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.action_support import (
    ActionReferenceResolutionDiagnostics,
    ActionSupportDecision,
    FinalStrategyCompilationContext,
    FinalStrategyCompilationReport,
    UnsupportedFinalStrategyActionError,
    compile_final_strategy_action_support,
    enforce_action_evidence_support,
)
from kaggle_researcher.contracts.composite_reference_resolution import (
    CompositeReferenceResolutionDiagnostics,
    resolve_composite_action_references,
    resolve_final_strategy_composite_references,
)
from kaggle_researcher.contracts.evidence import (
    AmbiguousEvidencePathError,
    EvidencePathResolutionError,
    EvidenceReference,
    EvidenceRegistry,
    ResolvedEvidence,
    ResolvedEvidencePath,
    SemanticCollectionSpec,
    build_evidence_registry,
    generate_allowed_evidence_refs,
    generate_semantic_evidence_refs,
    resolve_evidence_path,
    resolve_evidence_reference,
)
from kaggle_researcher.contracts.ids import EvidenceId, ExperimentId, HypothesisId
from kaggle_researcher.contracts.hypothesis_reference_migration import (
    HypothesisReferenceMigrationDiagnostics,
    migrate_final_strategy_hypothesis_references,
    migrate_hypothesis_references,
)
from kaggle_researcher.contracts.reference_catalog import (
    REFERENCE_NAMESPACES,
    ReferenceCatalog,
    ReferenceCatalogDiagnostic,
    ReferenceCatalogEntry,
    ReferenceNamespace,
    ReferenceResolution,
    build_final_strategy_reference_catalog,
)


class HypothesisRef(ContractModel):
    hypothesis_id: HypothesisId


class ExperimentRef(ContractModel):
    experiment_id: ExperimentId


class EvidenceRef(ContractModel):
    evidence_id: EvidenceId


__all__ = [
    "ActionReferenceResolutionDiagnostics", "ActionSupportDecision",
    "AmbiguousEvidencePathError", "CompositeReferenceResolutionDiagnostics",
    "EvidencePathResolutionError", "EvidenceRef",
    "EvidenceReference", "EvidenceRegistry", "ExperimentRef", "HypothesisRef",
    "FinalStrategyCompilationContext", "FinalStrategyCompilationReport",
    "HypothesisReferenceMigrationDiagnostics",
    "ResolvedEvidence", "ResolvedEvidencePath", "SemanticCollectionSpec",
    "REFERENCE_NAMESPACES", "ReferenceCatalog", "ReferenceCatalogDiagnostic",
    "ReferenceCatalogEntry", "ReferenceNamespace", "ReferenceResolution",
    "build_evidence_registry", "generate_allowed_evidence_refs",
    "build_final_strategy_reference_catalog",
    "migrate_final_strategy_hypothesis_references", "migrate_hypothesis_references",
    "resolve_composite_action_references", "resolve_final_strategy_composite_references",
    "UnsupportedFinalStrategyActionError", "compile_final_strategy_action_support",
    "enforce_action_evidence_support",
    "generate_semantic_evidence_refs", "resolve_evidence_path",
    "resolve_evidence_reference",
]
