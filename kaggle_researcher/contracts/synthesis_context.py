from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Literal, Mapping

from pydantic import Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.evidence import EvidenceNamespace
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceReferenceEntry,
    EvidenceReferenceKind,
    EvidenceReferenceManifest,
    PublishedEdaEvidenceBundle,
    publish_eda_evidence_bundle,
)
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.contracts.ids import (
    EvidenceId,
    ExperimentId,
    HypothesisId,
    RiskId,
    SafetyConstraintId,
    ValidationRequirementId,
)
from kaggle_researcher.contracts.registries import ContractRegistries
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.review import SkepticalReview
from kaggle_researcher.contracts.validation import (
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ValidationResult,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument


class ValidationArchitectContext(ContractModel):
    plan_data: PlanData
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    allowed_evidence_ids: list[EvidenceId] = Field(default_factory=list)


class ExperimentPlanningContext(ContractModel):
    validation: ValidationResult
    leakage: LeakageRiskResult
    metric: MetricResult
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    allowed_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    allowed_evidence_ids: list[EvidenceId] = Field(default_factory=list)
    safety_constraints: list[dict[str, Any]] = Field(default_factory=list)
    validation_requirements: list[dict[str, Any]] = Field(default_factory=list)
    baseline_summary: dict[str, Any] = Field(default_factory=dict)


class SkepticalReviewContext(ContractModel):
    experiments: ExperimentPlan
    allowed_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    allowed_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    allowed_evidence_refs: list[EvidenceId] = Field(default_factory=list)


class FinalSynthesisContext(ContractModel):
    contract_family: Literal["final_synthesis_context"] = "final_synthesis_context"
    schema_version: Literal["1.0"] = "1.0"
    competition_desc: str
    plan_data: PlanData
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    domain_patterns: list[dict[str, Any]] = Field(default_factory=list)
    research_hypotheses: ResearchHypotheses
    published_eda_bundle: PublishedEdaEvidenceBundle
    eda_summary_text: str | None = None
    metric: MetricResult
    validation: ValidationResult
    leakage: LeakageRiskResult
    leaderboard: LeaderboardAuditResult
    experiment_plan: ExperimentPlan = Field(default_factory=ExperimentPlan)
    review: SkepticalReview | None = None
    approved_experiments: list[dict[str, Any]] = Field(default_factory=list)
    rejected_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    unresolved_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    allowed_experiment_ids: list[ExperimentId] = Field(default_factory=list)
    allowed_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    allowed_risk_ids: list[RiskId] = Field(default_factory=list)
    allowed_validation_requirement_ids: list[ValidationRequirementId] = Field(default_factory=list)
    allowed_safety_constraint_ids: list[SafetyConstraintId] = Field(default_factory=list)
    optional_stage_failure_messages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @property
    def eda_evidence_pack(self) -> EdaEvidencePack:
        return self.published_eda_bundle.evidence_pack

    @property
    def evidence_manifest(self) -> EvidenceReferenceManifest:
        return self.published_eda_bundle.evidence_manifest

    @property
    def pack_hash(self) -> str:
        return self.published_eda_bundle.pack_hash

    @property
    def manifest_hash(self) -> str:
        return self.published_eda_bundle.manifest_hash

    @property
    def bundle_hash(self) -> str:
        return self.published_eda_bundle.bundle_hash

    def evidence_refs(
        self,
        namespace: EvidenceNamespace = "eda_evidence",
        *,
        reference_kinds: set[EvidenceReferenceKind] | None = None,
    ) -> tuple[str, ...]:
        blocked = {
            conflict.ref for conflict in self.evidence_manifest.conflicts
            if conflict.severity == "error"
        }
        return tuple(sorted({
            entry.ref for entry in self.evidence_manifest.entries
            if entry.available
            and entry.ref not in blocked
            and entry.namespace == namespace
            and (reference_kinds is None or entry.reference_kind in reference_kinds)
        }))

    def lookup_evidence_ref(self, ref: str) -> EvidenceReferenceEntry | None:
        matches = [entry for entry in self.evidence_manifest.entries if entry.ref == ref]
        available = [entry for entry in matches if entry.available]
        return available[0] if len(available) == 1 else matches[0] if len(matches) == 1 else None

    @property
    def allowed_evidence_refs(self) -> list[EvidenceId]:
        return list(self.evidence_refs(
            "eda_evidence", reference_kinds={"direct_path", "semantic_ref"}
        ))

    @property
    def allowed_eda_result_refs(self) -> list[EvidenceId]:
        return list(self.evidence_refs(
            "eda_evidence", reference_kinds={"direct_path", "semantic_ref"}
        ))

    def reference_prompt_payload(self) -> dict[str, Any]:
        return {
            "approved_experiments": self.approved_experiments,
            "rejected_experiment_ids": self.rejected_experiment_ids,
            "unresolved_hypotheses": self.unresolved_hypotheses,
            "allowed_experiment_ids": self.allowed_experiment_ids,
            "allowed_hypothesis_ids": self.allowed_hypothesis_ids,
            "allowed_evidence_refs": self.allowed_evidence_refs,
            "allowed_eda_result_refs": self.allowed_eda_result_refs,
            "evidence_manifest_metadata": {
                "manifest_version": self.evidence_manifest.manifest_version,
                "pack_hash": self.pack_hash,
                "manifest_hash": self.manifest_hash,
                "bundle_hash": self.bundle_hash,
                "allowed_ref_count": len(self.allowed_eda_result_refs),
                "unavailable_ref_count": sum(
                    not entry.available for entry in self.evidence_manifest.entries
                ),
                "conflicting_ref_count": len({
                    conflict.ref for conflict in self.evidence_manifest.conflicts
                    if conflict.severity == "error"
                }),
            },
            "allowed_risk_ids": self.allowed_risk_ids,
            "allowed_validation_requirement_ids": self.allowed_validation_requirement_ids,
            "allowed_safety_constraint_ids": self.allowed_safety_constraint_ids,
            "safety_constraints": [
                item.model_dump(mode="json") for item in self.eda_evidence_pack.safety_constraints
            ],
            "validation_requirements": [
                item.model_dump(mode="json") for item in self.eda_evidence_pack.validation_requirements
            ],
            "risks": [item.model_dump(mode="json") for item in self.eda_evidence_pack.eda_risks],
            "optional_stage_failure_messages": self.optional_stage_failure_messages,
            "limitations": self.limitations,
        }


def build_final_synthesis_context(
    *,
    competition_desc: str,
    research: Any,
    published_eda_bundle: PublishedEdaEvidenceBundle,
    reasoning: Any,
    registries: ContractRegistries,
    eda_summary_text: str | None,
    optional_stage_failures: Iterable[Any] = (),
) -> FinalSynthesisContext:
    """Prepare the exact, validated Final Synthesizer input from stage results."""

    from kaggle_researcher.contracts.experiments import approved_experiment_summary
    from kaggle_researcher.contracts.evidence_manifest import validate_published_eda_bundle

    validate_published_eda_bundle(published_eda_bundle)
    evidence_pack = published_eda_bundle.evidence_pack
    failures = [getattr(value, "message", str(value)) for value in optional_stage_failures]
    return FinalSynthesisContext(
        competition_desc=competition_desc,
        plan_data=research.plan_data,
        retrieved_documents=list(research.retrieved_documents),
        domain_patterns=list(research.domain_patterns),
        research_hypotheses=research.hypotheses,
        published_eda_bundle=published_eda_bundle,
        eda_summary_text=eda_summary_text,
        metric=reasoning.metric,
        validation=reasoning.validation,
        leakage=reasoning.leakage,
        leaderboard=reasoning.leaderboard,
        experiment_plan=reasoning.experiments or ExperimentPlan(),
        review=reasoning.review,
        approved_experiments=approved_experiment_summary(registries.experiments),
        rejected_experiment_ids=sorted(registries.experiments.rejected_ids),
        unresolved_hypotheses=[
            item.model_dump(mode="json")
            for item in [
                *research.hypotheses.hypotheses,
                *evidence_pack.testable_hypotheses,
            ]
        ],
        allowed_experiment_ids=sorted(registries.experiments.approved_ids),
        allowed_hypothesis_ids=sorted(registries.hypotheses.by_id),
        allowed_risk_ids=sorted(registries.risks.by_id),
        allowed_validation_requirement_ids=sorted(registries.validation_requirements.by_id),
        allowed_safety_constraint_ids=sorted(registries.safety_constraints.by_id),
        optional_stage_failure_messages=failures,
        limitations=[
            *research.hypotheses.scout_limitations,
            *evidence_pack.limitations,
            *failures,
        ],
    )


def adapt_legacy_eda_evidence_pack(
    evidence_pack: EdaEvidencePack,
) -> PublishedEdaEvidenceBundle:
    """Compatibility adapter for pre-manifest artifacts; never emitted as context fields."""
    return publish_eda_evidence_bundle(
        evidence_pack,
        manifest_origin="legacy_migration",
        migration_warnings=(
            "Generated an in-memory evidence manifest from a frozen legacy EDA pack; "
            "origin=legacy_migration.",
        ),
    )


def migrate_legacy_final_synthesis_context_payload(
    payload: Mapping[str, Any],
):
    """Upgrade the pre-bundle context without mutating or persisting its snapshot."""
    from kaggle_researcher.contracts.errors import ContractMigrationError
    from kaggle_researcher.contracts.evidence_manifest import validate_published_eda_bundle
    from kaggle_researcher.contracts.migration import MigrationResult

    migrated = deepcopy(dict(payload))
    raw_bundle = migrated.get("published_eda_bundle")
    if raw_bundle is not None:
        try:
            bundle = PublishedEdaEvidenceBundle.model_validate(raw_bundle)
            validate_published_eda_bundle(bundle)
        except Exception as exc:
            raise ContractMigrationError(
                "Legacy FinalSynthesisContext contains an unprovable published EDA snapshot",
                contract="final_synthesis_context",
            ) from exc
    else:
        raw_pack = migrated.pop("eda_evidence_pack", None)
        if not isinstance(raw_pack, Mapping):
            raise ContractMigrationError(
                "Legacy FinalSynthesisContext has no frozen EDA pack from which to prove a snapshot",
                contract="final_synthesis_context",
            )
        try:
            pack = EdaEvidencePack.model_validate(raw_pack)
        except Exception as exc:
            raise ContractMigrationError(
                "Legacy FinalSynthesisContext EDA snapshot is invalid",
                contract="final_synthesis_context",
            ) from exc
        raw_manifest = migrated.pop("evidence_manifest", None)
        if raw_manifest is not None:
            # A separately supplied manifest must agree with the same frozen pack.
            try:
                declared = EvidenceReferenceManifest.model_validate(raw_manifest)
                canonical = publish_eda_evidence_bundle(pack)
            except Exception as exc:
                raise ContractMigrationError(
                    "Legacy FinalSynthesisContext evidence manifest cannot be verified",
                    contract="final_synthesis_context",
                ) from exc
            if (
                declared.pack_hash != canonical.pack_hash
                or declared.manifest_hash != canonical.manifest_hash
            ):
                raise ContractMigrationError(
                    "Legacy FinalSynthesisContext mixes an EDA pack and manifest from different snapshots",
                    contract="final_synthesis_context",
                )
        bundle = adapt_legacy_eda_evidence_pack(pack)
        migrated["published_eda_bundle"] = bundle.model_dump(mode="json")

    hypotheses = migrated.get("research_hypotheses")
    if isinstance(hypotheses, Mapping):
        hypothesis_competition = str(hypotheses.get("competition_id") or "")
        bundle_competition = bundle.evidence_pack.competition_id
        if hypothesis_competition and hypothesis_competition != bundle_competition:
            raise ContractMigrationError(
                "Legacy FinalSynthesisContext mixes artifacts from different competition runs",
                contract="final_synthesis_context",
            )

    migrated["contract_family"] = "final_synthesis_context"
    migrated["schema_version"] = "1.0"
    migrated.pop("allowed_evidence_refs", None)
    migrated.pop("allowed_eda_result_refs", None)
    limitations = list(migrated.get("limitations") or [])
    warning = (
        "Legacy FinalSynthesisContext migrated in memory; its evidence manifest has "
        "origin=legacy_migration."
    )
    if bundle.evidence_manifest.origin == "legacy_migration" and warning not in limitations:
        limitations.append(warning)
    migrated["limitations"] = limitations
    return MigrationResult(
        migrated,
        None,
        "1.0",
        True,
        ["embedded a verified immutable published EDA bundle"],
        [warning],
    )


__all__ = [
    "ExperimentPlanningContext", "FinalSynthesisContext",
    "SkepticalReviewContext", "ValidationArchitectContext",
    "build_final_synthesis_context",
    "adapt_legacy_eda_evidence_pack",
    "migrate_legacy_final_synthesis_context_payload",
]
