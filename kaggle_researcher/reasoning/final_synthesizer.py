from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.contracts.action_canonicalization import (
    ActionCanonicalizationDiagnostics,
    canonicalize_final_strategy_actions,
    canonicalize_semantic_strategy_actions,
    validate_semantic_action_postconditions,
)
from kaggle_researcher.contracts.action_evidence_resolution import (
    ActionEvidenceResolution,
    FinalStrategyActionEvidenceReport,
    classify_action,
    resolve_action_evidence_refs,
    resolve_final_strategy_action_evidence,
)
from kaggle_researcher.contracts.action_support import (
    FinalStrategyCompilationContext,
    FinalStrategyCompilationReport,
    UnsupportedFinalStrategyActionError,
    compile_final_strategy_action_support,
    enforce_action_evidence_support,
)
from kaggle_researcher.contracts.bundle_validation import validate_final_synthesis_bundle
from kaggle_researcher.contracts.composite_reference_resolution import (
    CompositeReferenceResolutionDiagnostics,
    resolve_composite_action_references,
    resolve_final_strategy_composite_references,
)
from kaggle_researcher.contracts.evidence import (
    EvidencePathResolutionError,
    resolve_evidence_ref,
)
from kaggle_researcher.contracts.errors import (
    EvidenceManifestBuildError,
    EvidenceManifestPackMismatchError,
)
from kaggle_researcher.contracts.evidence_manifest import (
    validate_published_eda_evidence_bundle,
)
from kaggle_researcher.contracts.experiments import (
    CrossNamespaceReferenceError,
    FORBIDDEN_CONTEXT_LABELS,
    ReferenceIssue,
    repair_final_experiment_references,
)
from kaggle_researcher.contracts.final_strategy import (
    ActionProvenance,
    Confidence,
    EvidenceOrigin,
    FALLBACK_LIMITATION,
    FinalStrategyAction,
    FinalStrategyResult,
    FinalStrategySection,
    HypothesisToEdaLink,
    FinalValidationMethod,
    Priority,
    REPAIR_LIMITATION,
    REQUIRED_SECTION_IDS,
    SynthesisStatus,
    SourceToHypothesisLink,
    TEMPORAL_VALIDATION_METHODS,
)
from kaggle_researcher.reasoning.deterministic_strategy import (
    CompiledAction,
    CompiledExperiment,
    StrategyContext,
    compile_competition_strategy,
)
from kaggle_researcher.reasoning.strategy_compaction import compact_final_strategy
from kaggle_researcher.reasoning.final_strategy_two_call import run_two_call_final_synthesis
from kaggle_researcher.reasoning.model_registry import supported_models
from kaggle_researcher.contracts.final_strategy_evidence import (
    build_action_evidence_bindings,
    validate_action_evidence_consistency,
)
from kaggle_researcher.contracts.final_strategy_compilation import (
    FinalStrategyCompilationDiagnostics,
    FinalStrategyCompilationError,
    FinalStrategyRepairError,
    FinalStrategySchemaValidationError,
)
from kaggle_researcher.contracts.final_strategy_draft import (
    FinalStrategyDraft,
    compile_final_strategy_draft,
)
from kaggle_researcher.contracts.final_synthesis_diagnostics import (
    FinalSynthesisDiagnostics,
    SynthesisAttemptDiagnostic,
    ValidationIssue,
    ValidationStage,
)
from kaggle_researcher.contracts.hypothesis_reference_migration import (
    HypothesisReferenceMigrationDiagnostics,
    migrate_final_strategy_hypothesis_references,
    migrate_hypothesis_references,
)
from kaggle_researcher.contracts.normalization import normalize_contract_payload
from kaggle_researcher.contracts.reference_catalog import (
    ReferenceCatalog,
    build_final_strategy_reference_catalog,
)
from kaggle_researcher.contracts.registries import ContractRegistries
from kaggle_researcher.contracts.synthesis_context import FinalSynthesisContext
from kaggle_researcher.eda.schemas import EdaEvidencePack, ResearchHypotheses
from kaggle_researcher.schemas import PlanData, RetrievedDocument


logger = logging.getLogger(__name__)


def _validate_synthesis_evidence_boundary(context: FinalSynthesisContext) -> None:
    validate_published_eda_evidence_bundle(context.published_eda_bundle)
    if context.evidence_manifest.schema_version != "1.0":
        raise EvidenceManifestBuildError(
            "Unsupported evidence manifest schema at Final Synthesizer boundary",
            stage="final_synthesis_pre_llm",
            contract="evidence_reference_manifest",
        )
    blocking_refs = {
        conflict.ref for conflict in context.evidence_manifest.conflicts
        if conflict.severity == "error"
    }
    exposed = set(context.allowed_eda_result_refs)
    if blocking_refs & exposed:
        raise EvidenceManifestBuildError(
            "Blocking evidence conflicts were exposed to Final Synthesizer",
            stage="final_synthesis_pre_llm",
            contract="evidence_reference_manifest",
        )


async def synthesize_final_strategy(
    *,
    context: FinalSynthesisContext,
    registries: ContractRegistries,
    client: DeepSeekClient,
    model: str,
    diagnostics_dir: Path | None = None,
) -> FinalStrategyResult:
    two_call = os.getenv("FINAL_SYNTHESIS_PROTOCOL", "two_call").strip().casefold() == "two_call"
    diagnostics = FinalSynthesisDiagnostics(
        competition_id=context.eda_evidence_pack.competition_id,
        protocol="two_call" if two_call else "monolithic_legacy",
        attempts=[] if two_call else [SynthesisAttemptDiagnostic(
            attempt="initial_llm",
            model=model,
        )],
        evidence_manifest_version=context.evidence_manifest.manifest_version,
        pack_hash=context.pack_hash,
        manifest_hash=context.manifest_hash,
        bundle_hash=context.bundle_hash,
        allowed_ref_count=len(context.allowed_eda_result_refs),
        conflicting_ref_count=len({
            conflict.ref for conflict in context.evidence_manifest.conflicts
            if conflict.severity == "error"
        }),
        unavailable_ref_count=sum(
            not entry.available for entry in context.evidence_manifest.entries
        ),
    )
    try:
        _validate_synthesis_evidence_boundary(context)
        diagnostics.prompt_manifest_hash = context.manifest_hash
        result = await _synthesize_final_strategy_impl(
            context=context,
            registries=registries,
            client=client,
            model=model,
            diagnostics_dir=diagnostics_dir,
            synthesis_diagnostics=diagnostics,
        )
        # Detect any accidental mutation performed by postprocessing or fallback
        # helpers before returning a strategy artifact.
        validate_published_eda_evidence_bundle(context.published_eda_bundle)
        if diagnostics.protocol == "two_call":
            return result
        _refresh_diagnostics_summary(diagnostics)
        return _assign_final_synthesis_status(
            result,
            diagnostics=diagnostics,
            diagnostics_dir=diagnostics_dir,
        )
    except Exception as exc:
        if isinstance(exc, EvidenceManifestBuildError):
            diagnostics.internal_contract_failure = exc.as_manifest_error()
            if isinstance(exc, EvidenceManifestPackMismatchError):
                diagnostics.actual_pack_hash = exc.actual_pack_hash
            _record_issue(
                diagnostics,
                "initial_llm",
                ValidationIssue(
                    stage="internal_contract_validation",
                    issue_type=type(exc).__name__,
                    message=_bounded_message(str(exc)),
                    expected_contract="PublishedEdaEvidenceBundle hash parity",
                ),
            )
        elif isinstance(exc, FinalStrategySchemaValidationError):
            attempt_name = (
                "deterministic_repair" if diagnostics.repair_attempted else "initial_llm"
            )
            stage: ValidationStage = (
                "repair_schema_validation"
                if diagnostics.repair_attempted
                else "llm_schema_validation"
            )
            target = _attempt_diagnostic(diagnostics, attempt_name)
            target.schema_validation_succeeded = False
            for issue in _collect_compilation_schema_issues(exc, stage=stage):
                if issue not in target.issues:
                    target.issues.append(issue)
        elif not any(attempt.issues for attempt in diagnostics.attempts):
            _record_issue(
                diagnostics,
                "initial_llm",
                ValidationIssue(
                    stage="llm_reference_validation",
                    issue_type=type(exc).__name__,
                    message=_bounded_message(str(exc)),
                    expected_contract="FinalStrategyResult",
                ),
            )
        raise
    finally:
        diagnostics.manifest_parity_succeeded = bool(
            diagnostics.prompt_manifest_hash
            and diagnostics.validator_manifest_hash
            and diagnostics.prompt_manifest_hash == diagnostics.validator_manifest_hash
        )
        _refresh_diagnostics_summary(diagnostics)
        write_final_synthesis_diagnostics(diagnostics_dir, diagnostics)


async def _synthesize_final_strategy_impl(
    *,
    context: FinalSynthesisContext,
    registries: ContractRegistries,
    client: DeepSeekClient,
    model: str,
    diagnostics_dir: Path | None,
    synthesis_diagnostics: FinalSynthesisDiagnostics,
) -> FinalStrategyResult:
    experiment_registry = registries.experiments
    competition_desc = context.competition_desc
    plan_data = context.plan_data
    retrieved_documents = context.retrieved_documents
    research_hypotheses = context.research_hypotheses
    eda_evidence_pack = context.eda_evidence_pack
    allowed_eda_refs = set(context.allowed_eda_result_refs)
    eda_summary_text = context.eda_summary_text
    try:
        reference_catalog = build_final_strategy_reference_catalog(
            eda_evidence_pack,
            evidence_manifest=context.evidence_manifest,
            research_hypotheses=research_hypotheses,
            source_claim_ids=[document.id for document in retrieved_documents],
            retrieved_documents=retrieved_documents,
        )
        synthesis_diagnostics.validator_manifest_hash = context.manifest_hash
    except FinalStrategyCompilationError:
        raise
    except Exception as exc:
        raise FinalStrategyRepairError(
            "Final strategy failed during reference_catalog_build.",
            phase="reference_catalog_build",
        ) from exc
    if os.getenv("FINAL_SYNTHESIS_PROTOCOL", "two_call").strip().casefold() == "two_call":
        def fallback_builder(reason: str) -> FinalStrategyResult:
            fallback = build_fallback_final_strategy(
                competition_id=eda_evidence_pack.competition_id,
                research_hypotheses=research_hypotheses.model_dump(mode="json").get("hypotheses", []),
                eda_evidence_pack=eda_evidence_pack.model_dump(mode="json"),
                eda_summary=eda_summary_text,
                task_type=plan_data.task_type,
                metric_name=plan_data.metric,
            )
            limitations = list(fallback.get("limitations") or [])
            explanation = f"Deterministic fallback selection was required: {_bounded_message(reason)}"
            if explanation not in limitations:
                limitations.append(explanation)
            fallback["limitations"] = limitations
            fallback["models_used"] = {
                **dict(fallback.get("models_used") or {}),
                "final_synthesizer": model,
            }
            grounded, _, _ = _ground_and_compile_strategy_payload(
                fallback,
                eda_evidence_pack=eda_evidence_pack,
                research_hypotheses=research_hypotheses,
                reference_catalog=reference_catalog,
                diagnostics_dir=diagnostics_dir,
                allowed_eda_refs=allowed_eda_refs,
            )
            result = FinalStrategyResult.model_validate(grounded)
            _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
            result = postprocess_final_strategy_result(
                result,
                eda_evidence_pack=eda_evidence_pack.model_dump(mode="json"),
                source_evidence=[
                    _retrieved_document_payload(document)
                    for document in retrieved_documents
                ],
            )
            return compact_final_strategy(
                result,
                evidence_pack=eda_evidence_pack.model_dump(mode="json"),
                source_ids=[str(document.id) for document in retrieved_documents],
            )

        return await run_two_call_final_synthesis(
            context=context,
            registries=registries,
            client=client,
            selection_model=os.getenv("FINAL_SYNTHESIS_SELECTION_MODEL", model),
            rendering_model=os.getenv("FINAL_SYNTHESIS_RENDERING_MODEL", model),
            diagnostics=synthesis_diagnostics,
            diagnostics_dir=diagnostics_dir,
            fallback_builder=fallback_builder,
        )
    try:
        raw = await client.chat_json(
            model=model,
            system_prompt=FINAL_SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=_build_final_synthesizer_prompt(
                context=context,
                registries=registries,
                reference_catalog=reference_catalog,
            ),
        )
    except Exception as exc:
        initial = _attempt_diagnostic(synthesis_diagnostics, "initial_llm")
        initial.output_received = _looks_like_parse_failure(exc)
        initial.json_parse_succeeded = False
        initial.issues.append(ValidationIssue(
            stage="llm_parse",
            issue_type=type(exc).__name__,
            message=_bounded_message(str(exc)),
            expected_contract="JSON object",
        ))
        raise
    initial = _attempt_diagnostic(synthesis_diagnostics, "initial_llm")
    initial.output_received = True
    initial.json_parse_succeeded = True
    initial.output_hash = _payload_hash(raw)
    _write_diagnostic_json(diagnostics_dir, "final_strategy_raw_payload.json", raw)
    _write_diagnostic_json(diagnostics_dir, "final_strategy_validation_errors.json", [])
    result = _result_from_payload(
        raw,
        plan_data=plan_data,
        eda_evidence_pack=eda_evidence_pack,
        research_hypotheses=research_hypotheses,
        eda_summary=eda_summary_text,
        model=model,
        reference_catalog=reference_catalog,
        diagnostics_dir=diagnostics_dir,
        synthesis_diagnostics=synthesis_diagnostics,
        allowed_eda_refs=allowed_eda_refs,
    )
    result = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=eda_evidence_pack.model_dump(mode="json"),
        source_evidence=[
            _retrieved_document_payload(document) for document in retrieved_documents
        ],
    )
    _enforce_primary_validation(result, eda_evidence_pack)
    known_hypothesis_ids, _ = _build_hypothesis_lookup(
        research_hypotheses.model_dump(mode="json").get("hypotheses", []),
        eda_evidence_pack.model_dump(mode="json"),
    )
    repaired_references = repair_final_experiment_references(result, experiment_registry)
    result = repaired_references.result
    if repaired_references.applied_repairs:
        experiment_repairs = [
            {
                "field_path": repair.field_path,
                "original_id": repair.original_id,
                "replacement_id": repair.replacement_id,
            }
            for repair in repaired_references.applied_repairs
        ]
        result.reference_repairs = [*experiment_repairs, *result.reference_repairs]
    _derive_eda_result_refs(result, allowed_eda_refs)
    issues = _final_reference_issues(result, context, registries)
    if issues:
        _record_reference_issues(
            synthesis_diagnostics,
            "initial_llm" if not synthesis_diagnostics.repair_attempted else "deterministic_repair",
            issues,
        )
        _ensure_repair_attempt(synthesis_diagnostics, model=model)
        try:
            result = await _repair_final_references_once(
                client=client,
                model=model,
                result=result,
                issues=issues,
                allowed_evidence_refs=context.allowed_evidence_refs,
                allowed_eda_result_refs=context.allowed_eda_result_refs,
                approved_experiment_ids=sorted(experiment_registry.approved_experiment_ids),
                allowed_risk_ids=context.allowed_risk_ids,
                allowed_validation_requirement_ids=context.allowed_validation_requirement_ids,
                allowed_safety_constraint_ids=context.allowed_safety_constraint_ids,
                reference_catalog=reference_catalog,
                diagnostics_dir=diagnostics_dir,
            )
            repair_attempt = _attempt_diagnostic(
                synthesis_diagnostics, "deterministic_repair"
            )
            repair_attempt.output_received = True
            repair_attempt.json_parse_succeeded = True
            repair_attempt.output_hash = _payload_hash(result.model_dump(mode="json"))
        except Exception as exc:
            repair_attempt = _attempt_diagnostic(
                synthesis_diagnostics, "deterministic_repair"
            )
            repair_attempt.reference_validation_succeeded = False
            if isinstance(exc, ValidationError):
                repair_attempt.output_received = True
                repair_attempt.json_parse_succeeded = True
                repair_attempt.issues.extend(collect_schema_issues(
                    exc, stage="repair_schema_validation"
                ))
            elif isinstance(exc, FinalStrategySchemaValidationError):
                repair_attempt.output_received = True
                repair_attempt.json_parse_succeeded = True
                repair_attempt.issues.extend(
                    _collect_compilation_schema_issues(
                        exc, stage="repair_schema_validation"
                    )
                )
            elif _looks_like_parse_failure(exc):
                repair_attempt.output_received = True
                repair_attempt.json_parse_succeeded = False
                repair_attempt.issues.append(ValidationIssue(
                    stage="repair_parse",
                    issue_type=type(exc).__name__,
                    message=_bounded_message(str(exc)),
                    expected_contract="JSON object",
                ))
            else:
                if isinstance(exc, FinalStrategyCompilationError):
                    repair_attempt.output_received = True
                    repair_attempt.json_parse_succeeded = True
                repair_attempt.issues.append(ValidationIssue(
                    stage="repair_reference_validation",
                    issue_type=type(exc).__name__,
                    message=_bounded_message(str(exc)),
                    expected_contract="FinalStrategyResult references",
                ))
            raise
        result = postprocess_final_strategy_result(
            result,
            eda_evidence_pack=eda_evidence_pack.model_dump(mode="json"),
            source_evidence=[
                _retrieved_document_payload(document) for document in retrieved_documents
            ],
        )
        _enforce_primary_validation(result, eda_evidence_pack)
        repaired_references = repair_final_experiment_references(result, experiment_registry)
        result = repaired_references.result
        _derive_eda_result_refs(result, allowed_eda_refs)
        remaining_issues = _final_reference_issues(result, context, registries)
        if remaining_issues:
            _record_reference_issues(
                synthesis_diagnostics, "deterministic_repair", remaining_issues
            )
            raise CrossNamespaceReferenceError(remaining_issues)
        repair_attempt = _attempt_diagnostic(
            synthesis_diagnostics, "deterministic_repair"
        )
        repair_attempt.output_received = True
        repair_attempt.json_parse_succeeded = True
        repair_attempt.schema_validation_succeeded = True
        repair_attempt.reference_validation_succeeded = True
        repair_attempt.output_hash = _payload_hash(result.model_dump(mode="json"))
    final_payload, _, _ = _ground_and_compile_strategy_payload(
        result.model_dump(mode="json"),
        eda_evidence_pack=eda_evidence_pack,
        research_hypotheses=research_hypotheses,
        reference_catalog=reference_catalog,
        diagnostics_dir=diagnostics_dir,
        write_support_artifact=False,
        allowed_eda_refs=allowed_eda_refs,
    )
    _synchronize_payload_eda_result_refs(final_payload, allowed_eda_refs)
    result = FinalStrategyResult.model_validate(final_payload)
    result = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=eda_evidence_pack.model_dump(mode="json"),
        source_evidence=[
            _retrieved_document_payload(document) for document in retrieved_documents
        ],
    )
    _derive_eda_result_refs(result, allowed_eda_refs)
    result = compact_final_strategy(
        result,
        evidence_pack=eda_evidence_pack.model_dump(mode="json"),
        source_ids=[document.id for document in retrieved_documents],
    )
    synthesis_diagnostics.quality_metrics = result.quality_metrics.model_dump(mode="json")
    synthesis_diagnostics.provenance_telemetry = dict(
        result.diagnostics_summary.get("provenance") or {}
    )
    validate_final_synthesis_bundle(
        eda_evidence_pack,
        context.experiment_plan,
        context.review,
        result,
        hypotheses=research_hypotheses,
        source_ids=[document.id for document in retrieved_documents],
        optional_stage_failures=context.optional_stage_failure_messages,
        evidence_manifest=context.evidence_manifest,
    )
    if not synthesis_diagnostics.fallback_required:
        effective = _attempt_diagnostic(
            synthesis_diagnostics,
            "deterministic_repair" if synthesis_diagnostics.repair_attempted else "initial_llm",
        )
        effective.schema_validation_succeeded = True
        effective.reference_validation_succeeded = True
    return result


async def _repair_final_references_once(
    *,
    client: DeepSeekClient,
    model: str,
    result: FinalStrategyResult,
    issues: list[Any],
    allowed_evidence_refs: list[str],
    allowed_eda_result_refs: list[str],
    approved_experiment_ids: list[str],
    allowed_risk_ids: list[str],
    allowed_validation_requirement_ids: list[str],
    allowed_safety_constraint_ids: list[str],
    reference_catalog: ReferenceCatalog,
    diagnostics_dir: Path | None = None,
) -> FinalStrategyResult:
    response = await client.chat_json(
        model=model,
        system_prompt=(
            "Correct only reference fields in the supplied FinalStrategyResult. "
            "Do not change action, reason, priority, section summary, or strategic intent. "
            "Use only allowed_evidence_refs in evidence_refs, only "
            "allowed_eda_result_refs in eda_result_refs, and only concrete "
            "approved_experiment_ids in experiment_ids. approved_experiments and other "
            "Use allowed_risk_ids only in risk_ids, allowed_validation_requirement_ids only "
            "in validation_requirement_ids, and allowed_safety_constraint_ids only in "
            "safety_constraint_ids. "
            "context collection labels are not evidence IDs. Return the complete corrected "
            "FinalStrategyResult JSON and nothing else."
        ),
        user_prompt=json.dumps({
            "invalid_references": [
                {
                    "path": issue.field_path,
                    "value": issue.invalid_value,
                    "reason": issue.reason,
                    "expected_namespace": issue.expected_namespace,
                    "actual_namespace": issue.actual_namespace,
                }
                for issue in issues
            ],
            "allowed_evidence_refs": allowed_evidence_refs,
            "allowed_eda_result_refs": allowed_eda_result_refs,
            "approved_experiment_ids": approved_experiment_ids,
            "allowed_risk_ids": allowed_risk_ids,
            "allowed_validation_requirement_ids": allowed_validation_requirement_ids,
            "allowed_safety_constraint_ids": allowed_safety_constraint_ids,
            "invalid_result": result.model_dump(mode="json"),
            "expected_schema": FinalStrategyResult.model_json_schema(),
        }, ensure_ascii=False, indent=2),
        timeout=120,
    )
    canonical, canonicalization_diagnostics = canonicalize_final_strategy_actions(response)
    _preserve_action_canonicalization_diagnostics(
        canonical,
        canonicalization_diagnostics,
    )
    original_response = deepcopy(canonical)
    migrated = canonical
    diagnostics = HypothesisReferenceMigrationDiagnostics()
    composite_diagnostics = CompositeReferenceResolutionDiagnostics()
    compilation_report = FinalStrategyCompilationReport()
    try:
        migrated, diagnostics = migrate_final_strategy_hypothesis_references(
            canonical,
            reference_catalog,
        )
        _preserve_hypothesis_migration_diagnostics(migrated, diagnostics)
        migrated, composite_diagnostics = resolve_final_strategy_composite_references(
            migrated,
            reference_catalog,
        )
        _preserve_composite_resolution_diagnostics(migrated, composite_diagnostics)
        migrated, compilation_report = compile_final_strategy_action_support(
            migrated,
            original_payload=original_response,
            context=FinalStrategyCompilationContext(reference_catalog=reference_catalog),
        )
        _preserve_action_support_report(migrated, compilation_report)
        _assign_synthesis_status_payload(
            migrated,
            status="repaired_success",
            diagnostics_dir=diagnostics_dir,
        )
        return FinalStrategyResult.model_validate(migrated)
    except ValidationError as exc:
        compilation_diagnostics = _compilation_diagnostics(
            phase="post_resolution_schema_validation",
            initial_issues=issues,
            payload=migrated,
            hypothesis_diagnostics=diagnostics,
            composite_diagnostics=composite_diagnostics,
            compilation_report=compilation_report,
        )
        raise FinalStrategySchemaValidationError(
            errors=exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            ),
            payload=migrated,
            diagnostics=compilation_diagnostics,
        ) from exc
    except FinalStrategyCompilationError:
        raise
    except Exception as exc:
        raise FinalStrategyRepairError(
            "Final strategy failed during reference_resolution.",
            phase="reference_resolution",
            diagnostics=_compilation_diagnostics(
                phase="reference_resolution",
                initial_issues=issues,
                payload=migrated,
                hypothesis_diagnostics=diagnostics,
                composite_diagnostics=composite_diagnostics,
                compilation_report=compilation_report,
            ),
        ) from exc


def _compilation_diagnostics(
    *,
    phase: str,
    initial_issues: list[Any],
    payload: dict[str, Any],
    hypothesis_diagnostics: HypothesisReferenceMigrationDiagnostics,
    composite_diagnostics: CompositeReferenceResolutionDiagnostics,
    compilation_report: FinalStrategyCompilationReport,
) -> FinalStrategyCompilationDiagnostics:
    diagnostic_unresolved = {
        *hypothesis_diagnostics.unknown_hypothesis_refs,
        *hypothesis_diagnostics.hypotheses_without_backing_evidence,
        *composite_diagnostics.composite_refs_without_evidence,
        *composite_diagnostics.unknown_composite_refs,
        *composite_diagnostics.policy_only_refs,
        *composite_diagnostics.broken_backing_evidence_refs,
    }
    remaining_initial = sum(
        _reference_issue_remains(payload, issue)
        for issue in initial_issues
    )
    return FinalStrategyCompilationDiagnostics(
        phase=phase,
        initial_reference_issues=len(initial_issues),
        resolved_references=max(0, len(initial_issues) - remaining_initial),
        unresolved_references=max(len(diagnostic_unresolved), remaining_initial),
        kept_actions=len(compilation_report.kept_actions),
        downgraded_actions=len(compilation_report.downgraded_actions),
        dropped_actions=len(compilation_report.dropped_actions),
    )


def _reference_issue_remains(payload: dict[str, Any], issue: Any) -> int:
    current: Any = payload
    for name, index in re.findall(r"([^.\[\]]+)(?:\[(\d+)\])?", issue.field_path):
        if not isinstance(current, dict) or name not in current:
            return 0
        current = current[name]
        if index:
            if not isinstance(current, list) or int(index) >= len(current):
                return 0
            current = current[int(index)]
    values = current if isinstance(current, (list, tuple, set)) else [current]
    return int(str(issue.invalid_value) in {str(value) for value in values})


def _final_reference_issues(
    result: FinalStrategyResult,
    context: FinalSynthesisContext,
    registries: ContractRegistries,
) -> list[ReferenceIssue]:
    issues: list[ReferenceIssue] = []
    source_ids = {document.id for document in context.retrieved_documents}

    def evidence_issue(
        field_path: str, value: str, *, allow_synthetic: bool = False
    ) -> ReferenceIssue | None:
        if allow_synthetic and value == "final_synthesizer.repaired":
            return None
        if value in FORBIDDEN_CONTEXT_LABELS:
            return ReferenceIssue(
                field_path, "eda_evidence", value, "context_label",
                "context_label_not_reference",
            )
        conflicts = [
            conflict for conflict in context.evidence_manifest.conflicts
            if conflict.ref == value and conflict.severity == "error"
        ]
        if conflicts:
            return ReferenceIssue(
                field_path, "eda_evidence", value,
                ",".join(conflicts[0].namespaces), "manifest_conflict",
            )
        matches = [
            entry for entry in context.evidence_manifest.entries if entry.ref == value
        ]
        if not matches:
            return ReferenceIssue(
                field_path, "eda_evidence", value, None, "unknown_reference"
            )
        available = [entry for entry in matches if entry.available]
        if not available:
            return ReferenceIssue(
                field_path, "eda_evidence", value,
                matches[0].namespace, "unavailable_reference",
            )
        namespace_matches = [
            entry for entry in available if entry.namespace == "eda_evidence"
        ]
        if not namespace_matches:
            return ReferenceIssue(
                field_path, "eda_evidence", value,
                available[0].namespace, "namespace_mismatch",
            )
        if not any(
            entry.reference_kind in {"direct_path", "semantic_ref"}
            for entry in namespace_matches
        ):
            return ReferenceIssue(
                field_path, "eda_evidence", value,
                namespace_matches[0].namespace, "reference_kind_mismatch",
            )
        return None

    def catalog_namespace(value: str) -> str | None:
        matches = []
        for namespace, identifiers in (
            ("source", source_ids),
            ("hypothesis", set(registries.hypotheses.by_id)),
            ("risk", set(registries.risks.by_id)),
            ("validation_requirement", set(registries.validation_requirements.by_id)),
            ("safety_constraint", set(registries.safety_constraints.by_id)),
            ("experiment", set(registries.experiments.by_id)),
        ):
            if value in identifiers:
                matches.append(namespace)
        return matches[0] if len(matches) == 1 else "multiple_catalogs" if matches else None

    for path, action in _iter_actions_with_paths(result):
        for field, values, allow_synthetic in (
            ("evidence_refs", action.evidence_refs, True),
            ("eda_result_refs", action.eda_result_refs, False),
        ):
            for value in values:
                issue = evidence_issue(
                    f"{path}.{field}", str(value), allow_synthetic=allow_synthetic
                )
                if issue is not None:
                    issues.append(issue)
        checks = (
            ("source_refs", action.source_refs, source_ids, "source"),
            ("hypothesis_ids", set(action.hypothesis_ids) | set(action.related_hypothesis_ids), set(registries.hypotheses.by_id), "hypothesis"),
            ("risk_ids", action.risk_ids, set(registries.risks.by_id), "risk"),
            ("validation_requirement_ids", action.validation_requirement_ids, set(registries.validation_requirements.by_id), "validation_requirement"),
            ("safety_constraint_ids", action.safety_constraint_ids, set(registries.safety_constraints.by_id), "safety_constraint"),
        )
        for field, values, allowed, expected in checks:
            for value in values:
                if value not in allowed:
                    issues.append(ReferenceIssue(
                        f"{path}.{field}", expected, str(value),
                        catalog_namespace(str(value)),
                        "namespace_mismatch" if catalog_namespace(str(value)) else "unknown_reference",
                    ))
        for value in action.experiment_ids:
            if value not in registries.experiments.approved_ids:
                namespace = "rejected_experiment" if value in registries.experiments.rejected_ids else catalog_namespace(str(value))
                issues.append(ReferenceIssue(
                    f"{path}.experiment_ids", "approved_experiment", str(value), namespace,
                    "rejected_experiment" if namespace == "rejected_experiment" else "unknown_or_unapproved_experiment",
                ))
    for index, section in enumerate(result.sections):
        for value in section.evidence_refs:
            issue = evidence_issue(
                f"sections[{index}].evidence_refs", str(value), allow_synthetic=True
            )
            if issue is not None:
                issues.append(issue)
    return issues


def _iter_actions_with_paths(result: FinalStrategyResult):
    for index, action in enumerate(result.actions):
        yield f"actions[{index}]", action


def _derive_eda_result_refs(
    result: FinalStrategyResult,
    allowed_eda_refs: set[str],
) -> None:
    for action in _all_actions(result):
        canonical = [
            reference for reference in action.evidence_refs
            if reference in allowed_eda_refs
        ]
        object.__setattr__(action, "eda_result_refs", canonical)


def _synchronize_payload_eda_result_refs(
    payload: dict[str, Any],
    allowed_eda_refs: set[str],
) -> None:
    for action in payload.get("actions") or []:
        if not isinstance(action, dict):
            continue
        original_eda_refs = _unique_strings(action.get("eda_result_refs") or [])
        invalid_eda_refs = [
            ref for ref in original_eda_refs if ref not in allowed_eda_refs
        ]
        # Preserve invalid transient refs until the namespace-repair stage can
        # diagnose them. They are never emitted by the finalized strategy.
        evidence_refs = _unique_strings([
            *(action.get("evidence_refs") or []),
            *invalid_eda_refs,
        ])
        action["evidence_refs"] = evidence_refs
        action["eda_result_refs"] = _unique_strings([
            *(ref for ref in evidence_refs if ref in allowed_eda_refs),
            *invalid_eda_refs,
        ])
        action["evidence_bindings"] = [
            binding for binding in action.get("evidence_bindings") or []
            if isinstance(binding, dict) and binding.get("ref") in evidence_refs
        ]
    action_map = {
        action.get("action_id"): action
        for action in payload.get("actions") or []
        if isinstance(action, dict) and action.get("action_id")
    }
    for section in payload.get("sections") or []:
        if not isinstance(section, dict) or not section.get("action_ids"):
            continue
        actions = [
            action_map[action_id]
            for action_id in section.get("action_ids") or []
            if action_id in action_map
        ]
        section["evidence_refs"] = _unique_strings([
            ref for action in actions for ref in action.get("evidence_refs") or []
        ])
        section["eda_result_refs"] = _unique_strings([
            ref for action in actions for ref in action.get("eda_result_refs") or []
        ])
        section["source_refs"] = _unique_strings([
            ref for action in actions for ref in action.get("source_refs") or []
        ])
        section["related_hypothesis_ids"] = _unique_strings([
            ref for action in actions for ref in action.get("hypothesis_ids") or []
        ])
    payload["action_provenance"] = []


FINAL_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are the Final Strategy Synthesizer for a Kaggle research pipeline. "
    "Return only JSON matching the FinalStrategyDraft schema. Do not include raw "
    "chain-of-thought. Use typed support_refs and never write evidence_refs directly. "
    "Do not make unsupported claims. Do not claim that notebooks were executed. "
    "Do not claim that a baseline is the final solution. Link every important "
    "recommendation through typed support_refs for EDA evidence and Scout hypotheses. "
    "Respect validation_evidence.primary_validation exactly; do not replace it with "
    "a different primary validation policy. If EDA selected StratifiedKFold, do not "
    "override it with temporal CV. If temporal validation is diagnostic only, state "
    "that clearly. If EDA evidence is missing for a claim, mark it as a hypothesis "
    "or limitation."
)


def _build_final_synthesizer_prompt(
    *,
    context: FinalSynthesisContext,
    registries: ContractRegistries,
    reference_catalog: ReferenceCatalog,
) -> str:
    experiment_registry = registries.experiments
    context_payload = context.reference_prompt_payload()
    payload = {
        "competition_desc": context.competition_desc,
        "plan_data": context.plan_data.model_dump(mode="json"),
        "retrieved_documents": [
            _retrieved_document_payload(document) for document in context.retrieved_documents[:20]
        ],
        "domain_patterns": context.domain_patterns,
        "eda_summary_markdown": context.eda_summary_text[:12000] if context.eda_summary_text else None,
        "must_follow_eda_evidence": _eda_must_follow_payload(context.eda_evidence_pack),
        "supporting_reasoning": {
            name: _bounded_reasoning_summary(value.model_dump(mode="json"))
            for name, value in (
                ("metric", context.metric),
                ("validation", context.validation),
                ("leakage", context.leakage),
                ("leaderboard", context.leaderboard),
            )
        },
        "final_strategy_context": context_payload,
        "required_rule": (
            "Every important recommendation must follow source -> hypothesis -> EDA -> strategy: "
            "source claim from retrieved_documents, linked Scout hypothesis, linked EDA "
            "evidence, then a concrete strategy action. Represent every supporting object as "
            "a typed support_refs item with namespace and ref_id."
        ),
        "guardrails": [
            "Do not include raw chain-of-thought.",
            "Do not make unsupported claims.",
            "Do not claim that notebooks were executed.",
            "Do not claim that baseline is final solution.",
            "Link every important recommendation through typed support_refs.",
            "Respect validation_evidence.primary_validation.",
            "If EDA selected StratifiedKFold, do not override it with temporal CV.",
            "If temporal validation is diagnostic only, state that clearly.",
            "If EDA evidence is missing for a claim, mark it as hypothesis or limitation.",
            "Do not recommend primary IDs as predictive features by default.",
            "If target encoding or WoE is marked unsafe, recommend only OOF/fold-fitted encoding.",
            "If drift severity is high or critical, include leaderboard-risk diagnostics.",
            "Every action must include non-empty support_refs.",
            "Every P0 action must cite concrete factual EDA paths that semantically support it.",
            "For primary validation cite validation_evidence.primary_validation and do not cite an unrelated evidence root.",
            "When an EDA recommended_next_action expresses the same intent, preserve its factual evidence links.",
            "Do not create leakage.block_critical_issue unless a semantically matching leakage check has status=failed and severity=high or critical.",
            "A baseline hypothesis and phrases such as leakage-safe, leak-free, verify leakage, or avoid leakage are not factual evidence of leakage.",
            "For warning-only leakage emit a diagnostic audit action; do not claim confirmed leakage.",
            "If leakage checks passed, do not emit critical leakage remediation.",
            "Every action intent, related_hypothesis_ids, and factual evidence root must use compatible semantic categories.",
            "Every section must include actions or evidence_summary_refs.",
            "Use only allowed_hypothesis_ids; do not invent hypothesis IDs.",
            "Every value in experiment_ids must exactly match an approved_experiment_id.",
            "Do not place hypothesis IDs into experiment_ids.",
            "Use namespace=hypothesis for EDA hypothesis IDs.",
            "Do not invent experiment IDs or restore reviewer-rejected experiments.",
            "Use only exact IDs from the corresponding allowed list.",
            "Do not write evidence_refs directly.",
            "Use support_refs with explicit namespace and ref_id.",
            "Hypothesis IDs use namespace=hypothesis.",
            "Risk IDs use namespace=risk.",
            "Validation requirements use namespace=validation_requirement.",
            "Safety constraints use namespace=safety_constraint.",
            "Direct EDA paths use namespace=evidence.",
            "Retrieved source claims use namespace=source_claim.",
            "Do not invent source-to-hypothesis or hypothesis-to-EDA links; those are assembled deterministically from Scout and EDA contracts.",
            "Recommend only supported_model_families and never compare two aliases of one canonical family.",
            "Baseline reproduction is the first modeling step when completed baseline evidence exists.",
            "Threshold selection is OOF-only downstream postprocessing, never the first modeling step.",
        ],
        "allowed_hypothesis_ids": context_payload["allowed_hypothesis_ids"],
        "allowed_experiment_ids": context_payload["allowed_experiment_ids"],
        "approved_experiment_ids": sorted(experiment_registry.approved_experiment_ids),
        "rejected_experiment_ids": context_payload["rejected_experiment_ids"],
        "approved_experiments": context_payload["approved_experiments"],
        "allowed_evidence_refs": context_payload["allowed_evidence_refs"],
        "allowed_eda_result_refs": context_payload["allowed_eda_result_refs"],
        "evidence_manifest_metadata": context_payload["evidence_manifest_metadata"],
        "allowed_risk_ids": context_payload["allowed_risk_ids"],
        "allowed_validation_requirement_ids": context_payload["allowed_validation_requirement_ids"],
        "allowed_safety_constraint_ids": context_payload["allowed_safety_constraint_ids"],
        "allowed_support_refs": [
            {
                "namespace": entry.namespace,
                "ref_id": entry.canonical_ref,
            }
            for entry in reference_catalog.entries
        ],
        "validated_source_catalog": [
            {
                "source_ref": entry.canonical_ref,
                "title": entry.title,
                "source_type": entry.source_type,
            }
            for entry in reference_catalog.entries
            if entry.namespace == "source_claim" and entry.source_type is not None
        ],
        "supported_model_families": [
            {
                "canonical_family_id": identity.canonical_family_id,
                "implementation_id": identity.implementation_id,
                "display_name": identity.display_name,
                "capabilities": dict(identity.capabilities),
            }
            for identity in supported_models(context.plan_data.task_type)
        ],
        "allowed_hypothesis_ids_instruction": (
            "Every action MUST reference at least one allowed hypothesis ID in "
            "related_hypothesis_ids. Do not invent IDs."
        ),
        "evidence_reference_instruction": (
            "Do not write evidence_refs directly. Use support_refs with explicit namespace "
            "and ref_id. Hypothesis IDs use namespace=hypothesis; risk IDs use namespace=risk; "
            "validation requirements use namespace=validation_requirement; safety constraints "
            "use namespace=safety_constraint; direct EDA paths use namespace=evidence; source "
            "claims use namespace=source_claim. approved_experiments is a context section, not "
            "a reference. Other context section names are not references either. Reference "
            "approved work through concrete experiment_ids."
        ),
        "required_sections": REQUIRED_SECTION_IDS,
        "expected_schema": FinalStrategyDraft.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _bounded_reasoning_summary(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    allowed = {
        "confidence", "recommended_cv", "validation_risk", "likely_split",
        "primary_validation", "secondary_validation", "risk_level", "possible_issues",
        "recommended_checks", "metric_explanation", "needs_calibration",
        "rank_averaging_useful", "threshold_search_needed", "shake_up_risk",
        "submission_selection_rule", "public_lb_trust", "warnings", "limitations",
    }
    return {key: value[key] for key in sorted(value) if key in allowed}


def _retrieved_document_payload(document: RetrievedDocument) -> dict[str, Any]:
    return {
        "id": document.id,
        "source": document.source,
        "title": document.title,
        "url": str(document.url) if document.url is not None else None,
        "content": document.content[:1500],
        "metadata": document.metadata,
    }


def _eda_must_follow_payload(eda_evidence_pack: EdaEvidencePack) -> dict[str, Any]:
    schema = _as_dict(eda_evidence_pack.inferred_schema)
    validation = _as_dict(eda_evidence_pack.validation_evidence)
    drift = _as_dict(eda_evidence_pack.drift_evidence)
    feature_probe = [_as_dict(item) for item in eda_evidence_pack.feature_probe_evidence]
    leakage = [_as_dict(item) for item in eda_evidence_pack.leakage_evidence]
    return {
        "schema_roles": {
            "train_base_table": schema.get("train_base_table"),
            "test_base_table": schema.get("test_base_table"),
            "target_column": schema.get("target_column"),
            "primary_id_column": schema.get("primary_id_column"),
            "sample_submission_table": schema.get("sample_submission_table"),
            "global_roles": schema.get("global_roles") or {},
        },
        "recommended_validation_candidate": validation.get("recommended_validation_candidate") or validation.get("primary_validation") or {},
        # Temporary input-adapter alias for older prompts; downstream code must
        # treat it as evidence, not as the final recommendation.
        "primary_validation": validation.get("recommended_validation_candidate") or validation.get("primary_validation") or {},
        "metric_evidence": eda_evidence_pack.metric_evidence,
        "critical_failed_leakage_checks": [
            item.get("check_id")
            for item in leakage
            if item.get("status") == "failed"
            and item.get("severity") in {"high", "critical"}
        ],
        "warning_leakage_checks": [
            item.get("check_id")
            for item in leakage
            if item.get("status") == "warning"
        ],
        "passed_leakage_checks": [
            item.get("check_id")
            for item in leakage
            if item.get("status") == "passed"
        ],
        "baseline_hypothesis_ids": [
            item.hypothesis_id
            for item in eda_evidence_pack.hypothesis_results
            if item.category == "baseline"
        ],
        "leakage_warnings": [
            item
            for item in leakage
            if item.get("status") in {"failed", "warning"}
        ],
        "unsafe_feature_probes": [
            item
            for item in feature_probe
            if item.get("status") == "unsafe" or item.get("leakage_risk") == "high"
        ],
        "drift_severity": drift.get("feature_drift_severity") or drift.get("severity"),
        "eda_implications": eda_evidence_pack.eda_implications,
        "eda_local_risks": [item.model_dump(mode="json") for item in eda_evidence_pack.eda_risks],
        "safety_constraints": [
            item.model_dump(mode="json") for item in eda_evidence_pack.safety_constraints
        ],
        "validation_requirements": [
            item.model_dump(mode="json") for item in eda_evidence_pack.validation_requirements
        ],
        "testable_hypotheses": [
            item.model_dump(mode="json") for item in eda_evidence_pack.testable_hypotheses
        ],
        "source_claim_validation": eda_evidence_pack.source_claim_validation,
        "recommended_next_actions": [
            item.model_dump(mode="json")
            for item in eda_evidence_pack.recommended_next_actions
        ],
        "evidence_origins": eda_evidence_pack.evidence_origins,
    }


def _result_from_payload(
    payload: dict[str, Any],
    *,
    plan_data: PlanData,
    eda_evidence_pack: EdaEvidencePack,
    research_hypotheses: ResearchHypotheses,
    eda_summary: str | None = None,
    model: str,
    reference_catalog: ReferenceCatalog,
    diagnostics_dir: Path | None = None,
    synthesis_diagnostics: FinalSynthesisDiagnostics | None = None,
    allowed_eda_refs: set[str] | None = None,
) -> FinalStrategyResult:
    normalized = (
        compile_final_strategy_draft(payload, reference_catalog)
        if _looks_like_final_strategy_draft(payload)
        else dict(payload)
    )
    normalized.setdefault("competition_id", eda_evidence_pack.competition_id)
    normalized.setdefault("task_type", plan_data.task_type)
    normalized.setdefault("metric", {"name": plan_data.metric})
    normalized.setdefault("recommended_validation", _primary_validation_method(eda_evidence_pack))
    normalized["sections"] = _normalize_sections(normalized.get("sections", []))
    normalized["actions"] = _normalize_actions(normalized.get("actions", []))
    normalized["models_used"] = {
        **dict(normalized.get("models_used") or {}),
        "final_synthesizer": model,
    }
    initial_action_count = len(normalized["actions"])
    _assign_synthesis_status_payload(
        normalized,
        status="llm_success",
        diagnostics_dir=diagnostics_dir,
    )
    hypotheses_payload = research_hypotheses.model_dump(mode="json").get("hypotheses", [])
    eda_payload = eda_evidence_pack.model_dump(mode="json")
    known_ids, _ = _build_hypothesis_lookup(hypotheses_payload, eda_payload)

    normalized, evidence_report, _ = _ground_and_compile_strategy_payload(
        normalized,
        eda_evidence_pack=eda_evidence_pack,
        research_hypotheses=research_hypotheses,
        reference_catalog=reference_catalog,
        diagnostics_dir=diagnostics_dir,
        write_normalized_payload=True,
        allowed_eda_refs=allowed_eda_refs,
    )
    evidence_resolution_issues = _collect_evidence_resolution_issues(evidence_report)
    deterministic_action_fallback = bool(evidence_report.fallback_action_ids)
    if synthesis_diagnostics is not None and evidence_resolution_issues:
        initial = _attempt_diagnostic(synthesis_diagnostics, "initial_llm")
        initial.reference_validation_succeeded = False
        initial.issues.extend(evidence_resolution_issues)
        repair_attempt = _ensure_repair_attempt(synthesis_diagnostics, model=model)
        repair_attempt.output_received = True
        repair_attempt.json_parse_succeeded = True
        repair_attempt.output_hash = _payload_hash(normalized)
        if evidence_report.fallback_action_ids:
            synthesis_diagnostics.fallback_required = True
            synthesis_diagnostics.fallback_reason = (
                "The initial output contained no usable actions; deterministic fallback "
                "actions were generated from registered evidence."
            )

    if deterministic_action_fallback and initial_action_count == 0:
        fallback = build_fallback_final_strategy(
            competition_id=eda_evidence_pack.competition_id,
            research_hypotheses=hypotheses_payload,
            eda_evidence_pack=eda_payload,
            eda_summary=eda_summary,
            task_type=plan_data.task_type,
            metric_name=plan_data.metric,
        )
        fallback["models_used"] = {
            **dict(fallback.get("models_used") or {}),
            "final_synthesizer": model,
        }
        fallback, _, _ = _ground_and_compile_strategy_payload(
            fallback,
            eda_evidence_pack=eda_evidence_pack,
            research_hypotheses=research_hypotheses,
            reference_catalog=reference_catalog,
            diagnostics_dir=diagnostics_dir,
            allowed_eda_refs=allowed_eda_refs,
        )
        result = FinalStrategyResult.model_validate(fallback)
        _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
        return result

    try:
        result = FinalStrategyResult.model_validate(normalized)
        unknown_hypothesis_issues = _unknown_hypothesis_reference_issues(
            result, set(known_ids)
        )
        if synthesis_diagnostics is not None:
            initial = _attempt_diagnostic(synthesis_diagnostics, "initial_llm")
            initial.schema_validation_succeeded = True
        if not unknown_hypothesis_issues:
            if synthesis_diagnostics is not None:
                if (
                    synthesis_diagnostics.repair_attempted
                    and not synthesis_diagnostics.fallback_required
                ):
                    repair_attempt = _attempt_diagnostic(
                        synthesis_diagnostics, "deterministic_repair"
                    )
                    repair_attempt.schema_validation_succeeded = True
                    repair_attempt.reference_validation_succeeded = True
                    synthesis_diagnostics.repair_succeeded = True
                else:
                    initial.reference_validation_succeeded = True
            _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
            return result
        if synthesis_diagnostics is not None:
            initial.reference_validation_succeeded = False
            initial.issues.extend(unknown_hypothesis_issues)
    except ValidationError as first_error:
        _write_validation_errors(diagnostics_dir, first_error, phase="initial_schema_validation")
        if synthesis_diagnostics is not None:
            initial = _attempt_diagnostic(synthesis_diagnostics, "initial_llm")
            initial.schema_validation_succeeded = False
            initial.reference_validation_succeeded = False
            initial.issues.extend(collect_schema_issues(
                first_error, stage="llm_schema_validation"
            ))
        logger.info(
            "Final strategy payload required deterministic repair (%d validation errors).",
            first_error.error_count(),
        )

    repaired = repair_final_strategy_payload(
        normalized,
        research_hypotheses=hypotheses_payload,
        eda_evidence_pack=eda_payload,
        eda_summary=eda_summary,
    )
    _assign_synthesis_status_payload(
        repaired,
        status="repaired_success",
        diagnostics_dir=diagnostics_dir,
    )
    if synthesis_diagnostics is not None:
        repair_attempt = _ensure_repair_attempt(synthesis_diagnostics, model=model)
        repair_attempt.output_received = True
        repair_attempt.json_parse_succeeded = True
        repair_attempt.output_hash = _payload_hash(repaired)
    try:
        repaired, _, _ = _ground_and_compile_strategy_payload(
            repaired,
            eda_evidence_pack=eda_evidence_pack,
            research_hypotheses=research_hypotheses,
            reference_catalog=reference_catalog,
            diagnostics_dir=diagnostics_dir,
            allowed_eda_refs=allowed_eda_refs,
        )
        result = FinalStrategyResult.model_validate(repaired)
        if synthesis_diagnostics is not None:
            repair_attempt.schema_validation_succeeded = True
            repair_attempt.reference_validation_succeeded = True
            synthesis_diagnostics.repair_succeeded = True
        _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
        return result
    except ValidationError as second_error:
        _write_validation_errors(diagnostics_dir, second_error, phase="repaired_schema_validation")
        if synthesis_diagnostics is not None:
            repair_attempt.schema_validation_succeeded = False
            repair_attempt.reference_validation_succeeded = False
            repair_attempt.issues.extend(collect_schema_issues(
                second_error, stage="repair_schema_validation"
            ))
            synthesis_diagnostics.repair_succeeded = False
            synthesis_diagnostics.fallback_required = True
            synthesis_diagnostics.fallback_reason = (
                "Deterministic repair failed FinalStrategyResult schema validation."
            )
        logger.warning(
            "Repaired final strategy remained invalid; using deterministic fallback "
            "(%d validation errors).",
            second_error.error_count(),
        )

    fallback = build_fallback_final_strategy(
        competition_id=eda_evidence_pack.competition_id,
        research_hypotheses=hypotheses_payload,
        eda_evidence_pack=eda_payload,
        eda_summary=eda_summary,
        task_type=plan_data.task_type,
        metric_name=plan_data.metric,
    )
    fallback.setdefault("task_type", plan_data.task_type)
    fallback.setdefault("metric", {"name": plan_data.metric})
    fallback.setdefault("recommended_validation", _primary_validation_method(eda_evidence_pack))
    fallback["models_used"] = {
        **dict(fallback.get("models_used") or {}),
        "final_synthesizer": model,
    }
    fallback, _, _ = _ground_and_compile_strategy_payload(
        fallback,
        eda_evidence_pack=eda_evidence_pack,
        research_hypotheses=research_hypotheses,
        reference_catalog=reference_catalog,
        diagnostics_dir=diagnostics_dir,
        allowed_eda_refs=allowed_eda_refs,
    )
    result = FinalStrategyResult.model_validate(fallback)
    _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
    return result


def _ground_and_compile_strategy_payload(
    payload: dict[str, Any],
    *,
    eda_evidence_pack: EdaEvidencePack,
    research_hypotheses: ResearchHypotheses,
    reference_catalog: ReferenceCatalog,
    diagnostics_dir: Path | None,
    write_normalized_payload: bool = False,
    write_support_artifact: bool = True,
    allowed_eda_refs: set[str] | None = None,
) -> tuple[
    dict[str, Any],
    FinalStrategyActionEvidenceReport,
    FinalStrategyCompilationReport,
]:
    canonical, canonicalization_diagnostics = canonicalize_final_strategy_actions(payload)
    _preserve_action_canonicalization_diagnostics(
        canonical, canonicalization_diagnostics
    )
    original_for_support_gate = deepcopy(canonical)
    migrated, migration_diagnostics = migrate_final_strategy_hypothesis_references(
        canonical, reference_catalog
    )
    _preserve_hypothesis_migration_diagnostics(migrated, migration_diagnostics)
    migrated, composite_diagnostics = resolve_final_strategy_composite_references(
        migrated, reference_catalog
    )
    _preserve_composite_resolution_diagnostics(migrated, composite_diagnostics)
    had_actions = bool(migrated.get("actions"))
    grounded, evidence_report = resolve_final_strategy_action_evidence(
        migrated,
        eda_evidence_pack=eda_evidence_pack,
        research_hypotheses=research_hypotheses,
        allowed_reference_index=reference_catalog,
    )
    if not had_actions and evidence_report.fallback_action_ids:
        limitations = _string_values(grounded.get("limitations"))
        if FALLBACK_LIMITATION not in limitations:
            limitations.append(FALLBACK_LIMITATION)
        grounded["limitations"] = limitations
    if write_normalized_payload:
        _write_diagnostic_json(
            diagnostics_dir, "final_strategy_normalized_payload.json", grounded
        )

    support_report_path = (
        str(diagnostics_dir / "final_strategy_action_support_report.json")
        if diagnostics_dir is not None
        else None
    )
    context = FinalStrategyCompilationContext(
        reference_catalog=reference_catalog,
        action_evidence_resolutions=evidence_report.actions,
        support_report_path=support_report_path,
    )
    try:
        compiled, compilation_report = compile_final_strategy_action_support(
            grounded,
            original_payload=original_for_support_gate,
            context=context,
        )
    except UnsupportedFinalStrategyActionError as exc:
        _write_action_support_report(
            diagnostics_dir,
            evidence_report=evidence_report,
            compilation_report=exc.compilation_report,
            status="failed",
        )
        _write_diagnostic_json(
            diagnostics_dir,
            "final_strategy_validation_errors.json",
            [{
                "phase": exc.phase,
                "error_type": type(exc).__name__,
                "action_id": exc.action_id,
                "priority": exc.priority,
                "message": str(exc),
            }],
        )
        raise
    _preserve_action_support_report(compiled, compilation_report)
    _populate_structured_provenance(
        compiled,
        research_hypotheses=research_hypotheses,
        eda_evidence_pack=eda_evidence_pack,
        reference_catalog=reference_catalog,
    )
    effective_allowed_refs = (
        allowed_eda_refs
        if allowed_eda_refs is not None
        else {
            entry.canonical_ref for entry in reference_catalog.entries
            if entry.namespace == "evidence" and entry.evidence_backed
        }
    )
    _synchronize_payload_eda_result_refs(compiled, effective_allowed_refs)
    if write_support_artifact:
        _write_action_support_report(
            diagnostics_dir,
            evidence_report=evidence_report,
            compilation_report=compilation_report,
            status="passed",
        )
    if write_normalized_payload:
        _write_diagnostic_json(
            diagnostics_dir, "final_strategy_normalized_payload.json", compiled
        )
    return compiled, evidence_report, compilation_report


def build_deterministic_provenance_links(
    *,
    research_hypotheses: ResearchHypotheses,
    eda_evidence_pack: EdaEvidencePack,
    reference_catalog: ReferenceCatalog,
) -> tuple[list[SourceToHypothesisLink], list[HypothesisToEdaLink], list[dict[str, str]]]:
    """Build producer-owned provenance links without asking the LLM to infer them."""

    source_links: list[SourceToHypothesisLink] = []
    eda_links: list[HypothesisToEdaLink] = []
    repairs: list[dict[str, str]] = []
    seen_source: set[tuple[str, str, str]] = set()
    seen_eda: set[tuple[str, str]] = set()

    for hypothesis in research_hypotheses.hypotheses:
        hypothesis_id = str(hypothesis.hypothesis_id)
        relationship = (
            "supports" if hypothesis.status == "supported_by_source" else "motivates"
        )
        for source_ref in hypothesis.source_refs:
            source_id = str(source_ref)
            resolution = reference_catalog.resolve(source_id, "source_claim")
            if not resolution.is_resolved:
                repairs.append({
                    "field_path": f"research_hypotheses.{hypothesis_id}.source_refs",
                    "original_id": source_id,
                    "replacement_id": "",
                })
                continue
            key = (source_id, hypothesis_id, relationship)
            if key in seen_source:
                continue
            seen_source.add(key)
            source_links.append(SourceToHypothesisLink(
                source_ref=source_id,
                hypothesis_id=hypothesis_id,
                relationship=relationship,
                claim_summary=_concise_provenance_summary(hypothesis.claim),
                confidence=hypothesis.confidence_before_eda,
            ))

    known_hypotheses = {
        entry.ref_id for entry in reference_catalog.entries
        if entry.namespace == "hypothesis"
    }
    for result in eda_evidence_pack.hypothesis_results:
        hypothesis_id = str(result.hypothesis_id)
        if hypothesis_id not in known_hypotheses:
            repairs.append({
                "field_path": f"hypothesis_results.{hypothesis_id}.hypothesis_id",
                "original_id": hypothesis_id,
                "replacement_id": "",
            })
            continue
        result_refs = list(dict.fromkeys(str(ref) for ref in result.evidence_refs))
        semantic_ref = f"hypothesis_results.{hypothesis_id}"
        if not result_refs and reference_catalog.resolve(
            semantic_ref, "evidence"
        ).is_resolved:
            result_refs = [semantic_ref]
        for eda_result_ref in result_refs:
            if not reference_catalog.resolve(eda_result_ref, "evidence").is_resolved:
                repairs.append({
                    "field_path": f"hypothesis_results.{hypothesis_id}.evidence_refs",
                    "original_id": eda_result_ref,
                    "replacement_id": "",
                })
                continue
            key = (hypothesis_id, eda_result_ref)
            if key in seen_eda:
                continue
            seen_eda.add(key)
            eda_links.append(HypothesisToEdaLink(
                hypothesis_id=hypothesis_id,
                eda_result_ref=eda_result_ref,
                result_status=result.status,
                finding_summary=_concise_provenance_summary(result.finding),
                confidence=result.confidence_after_eda,
            ))
    return source_links, eda_links, repairs


def _populate_structured_provenance(
    payload: dict[str, Any],
    *,
    research_hypotheses: ResearchHypotheses,
    eda_evidence_pack: EdaEvidencePack,
    reference_catalog: ReferenceCatalog,
) -> None:
    source_links, eda_links, repairs = build_deterministic_provenance_links(
        research_hypotheses=research_hypotheses,
        eda_evidence_pack=eda_evidence_pack,
        reference_catalog=reference_catalog,
    )
    payload["source_to_hypothesis_links"] = [
        link.model_dump(mode="json") for link in source_links
    ]
    payload["hypothesis_to_eda_links"] = [
        link.model_dump(mode="json") for link in eda_links
    ]
    source_hypothesis_pairs = {
        (str(link.source_ref), str(link.hypothesis_id)) for link in source_links
    }
    source_claims = {
        (str(link.source_ref), str(link.hypothesis_id)): link.claim_summary
        for link in source_links if link.claim_summary
    }
    action_provenance: list[dict[str, Any]] = []
    action_by_id: dict[str, dict[str, Any]] = {}
    for index, action in enumerate(payload.get("actions") or []):
        if not isinstance(action, dict):
            continue
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            continue
        hypothesis_ids = _unique_strings([
            *_string_values(action.get("related_hypothesis_ids")),
            *_string_values(action.get("hypothesis_ids")),
        ])
        valid_hypothesis_ids = list(hypothesis_ids)
        source_refs: list[str] = []
        for source_ref in _string_values(action.get("source_refs")):
            valid_source = reference_catalog.resolve(
                source_ref, "source_claim"
            ).is_resolved
            linked_to_action = any(
                (source_ref, hypothesis_id) in source_hypothesis_pairs
                for hypothesis_id in valid_hypothesis_ids
            )
            if valid_source and linked_to_action:
                if source_ref not in source_refs:
                    source_refs.append(source_ref)
            else:
                repairs.append({
                    "field_path": f"actions[{index}].source_refs",
                    "original_id": source_ref,
                    "replacement_id": "",
                })
        eda_result_refs = _unique_strings(
            _string_values(action.get("eda_result_refs"))
        )
        action["related_hypothesis_ids"] = valid_hypothesis_ids
        action["hypothesis_ids"] = list(valid_hypothesis_ids)
        action["source_refs"] = source_refs
        action["eda_result_refs"] = eda_result_refs
        if source_refs:
            summaries = [
                source_claims[(source_ref, hypothesis_id)]
                for source_ref in source_refs
                for hypothesis_id in valid_hypothesis_ids
                if (source_ref, hypothesis_id) in source_claims
            ]
            if summaries:
                action["source_claim"] = _safe_action_source_claim(summaries[0])
        action_provenance.append({
            "action_id": action_id,
            "source_refs": source_refs,
            "hypothesis_ids": valid_hypothesis_ids,
            "eda_result_refs": eda_result_refs,
        })
        action_by_id[action_id] = action
    payload["action_provenance"] = action_provenance

    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_actions = [
            action_by_id[action_id]
            for action_id in _string_values(section.get("action_ids"))
            if action_id in action_by_id
        ]
        if not section_actions:
            continue
        section["evidence_refs"] = _unique_strings(
            ref for action in section_actions for ref in action.get("evidence_refs", [])
        )
        section["related_hypothesis_ids"] = _unique_strings(
            ref for action in section_actions for ref in action.get("hypothesis_ids", [])
        )
        section["source_refs"] = _unique_strings(
            ref for action in section_actions for ref in action.get("source_refs", [])
        )
        section["eda_result_refs"] = _unique_strings(
            ref for action in section_actions for ref in action.get("eda_result_refs", [])
        )

    existing_repairs = list(payload.get("reference_repairs") or [])
    for repair in repairs:
        if repair not in existing_repairs:
            existing_repairs.append(repair)
    payload["reference_repairs"] = existing_repairs


def _concise_provenance_summary(value: Any, limit: int = 240) -> str:
    summary = " ".join(str(value or "Evidence relationship recorded.").split())
    if len(summary) <= limit:
        return summary
    return summary[: limit - 1].rstrip() + "…"


def _safe_action_source_claim(value: Any) -> str:
    summary = _concise_provenance_summary(value, limit=170).strip(" \"'“”")
    if summary:
        summary = summary[0].lower() + summary[1:]
    return _concise_provenance_summary(
        f"Retrieved research motivated testing whether {summary}",
        limit=220,
    )


def _preserve_hypothesis_migration_diagnostics(
    payload: dict[str, Any],
    diagnostics: HypothesisReferenceMigrationDiagnostics,
) -> None:
    if not diagnostics.changed:
        return
    repairs = list(payload.get("reference_repairs") or [])
    for field_name in (
        "moved_hypothesis_refs",
        "inherited_evidence_refs",
        "unknown_hypothesis_refs",
        "hypotheses_without_backing_evidence",
    ):
        for value in getattr(diagnostics, field_name):
            record = {
                "field_path": f"hypothesis_reference_migration.{field_name}",
                "original_id": value,
                "replacement_id": value if field_name != "unknown_hypothesis_refs" else "",
            }
            if record not in repairs:
                repairs.append(record)
    payload["reference_repairs"] = repairs


def _preserve_action_canonicalization_diagnostics(
    payload: dict[str, Any],
    diagnostics: ActionCanonicalizationDiagnostics,
) -> None:
    # Generated IDs and structural reuse under the same ID are not reference
    # changes. Only old_id -> new_id rewrites belong in reference_repairs.
    payload["reference_repairs"] = list(payload.get("reference_repairs") or [])


def _preserve_composite_resolution_diagnostics(
    payload: dict[str, Any],
    diagnostics: CompositeReferenceResolutionDiagnostics,
) -> None:
    if not diagnostics.changed:
        return
    repairs = list(payload.get("reference_repairs") or [])
    for field_name in (
        "resolved_composite_refs",
        "inherited_backing_evidence_refs",
        "composite_refs_without_evidence",
        "unknown_composite_refs",
        "policy_only_refs",
        "broken_backing_evidence_refs",
    ):
        for value in getattr(diagnostics, field_name):
            record = {
                "field_path": f"composite_reference_resolution.{field_name}",
                "original_id": value,
                "replacement_id": value if field_name == "inherited_backing_evidence_refs" else "",
            }
            if record not in repairs:
                repairs.append(record)
    payload["reference_repairs"] = repairs


def _preserve_action_support_report(
    payload: dict[str, Any],
    report: FinalStrategyCompilationReport,
) -> None:
    repairs = list(payload.get("reference_repairs") or [])
    for field_name in (
        "kept_actions",
        "downgraded_actions",
        "dropped_actions",
        "failed_actions",
    ):
        for decision in getattr(report, field_name):
            record = {
                "field_path": f"action_support_compilation.{field_name}",
                "original_id": decision.action_id,
                "replacement_id": decision.reason,
            }
            if record not in repairs:
                repairs.append(record)
    payload["reference_repairs"] = repairs


def _write_action_support_report(
    diagnostics_dir: Path | None,
    *,
    evidence_report: FinalStrategyActionEvidenceReport,
    compilation_report: FinalStrategyCompilationReport,
    status: str,
) -> None:
    _write_diagnostic_json(
        diagnostics_dir,
        "final_strategy_action_support_report.json",
        {
            "status": status,
            "evidence_resolution": evidence_report.model_dump(mode="json"),
            "strict_gate": compilation_report.model_dump(mode="json"),
        },
    )


def normalize_validation_error(
    error: dict[str, Any],
    *,
    stage: ValidationStage,
) -> ValidationIssue:
    location = error.get("loc") or ()
    field_path = ".".join(str(part) for part in location) or None
    invalid_input = error.get("input")
    return ValidationIssue(
        stage=stage,
        issue_type=str(error.get("type") or "validation_error"),
        field_path=field_path,
        message=_bounded_message(str(error.get("msg") or "Validation failed.")),
        invalid_value_type=(
            type(invalid_input).__name__ if "input" in error else None
        ),
        expected_contract="FinalStrategyResult",
    )


def _assign_synthesis_status_payload(
    payload: dict[str, Any],
    *,
    status: SynthesisStatus,
    diagnostics_dir: Path | None,
) -> None:
    state = {
        "llm_success": (True, False, False, False),
        "repaired_success": (False, True, True, False),
        "degraded_fallback": (False, False, False, True),
    }[status]
    payload.update({
        "synthesis_status": status,
        "llm_output_valid": state[0],
        "repair_attempted": state[1],
        "repair_succeeded": state[2],
        "fallback_used": state[3],
        "synthesis_diagnostics_path": (
            str(diagnostics_dir / "final_synthesis_diagnostics.json")
            if diagnostics_dir is not None
            else None
        ),
    })
    if status == "degraded_fallback":
        limitations = _string_values(payload.get("limitations"))
        if FALLBACK_LIMITATION not in limitations:
            limitations.append(FALLBACK_LIMITATION)
        payload["limitations"] = limitations


def _assign_final_synthesis_status(
    result: FinalStrategyResult,
    *,
    diagnostics: FinalSynthesisDiagnostics,
    diagnostics_dir: Path | None,
) -> FinalStrategyResult:
    if diagnostics.fallback_required:
        status: SynthesisStatus = "degraded_fallback"
    elif diagnostics.repair_attempted and diagnostics.repair_succeeded:
        status = "repaired_success"
    elif diagnostics.initial_output_valid and not diagnostics.repair_attempted:
        status = "llm_success"
    else:
        raise RuntimeError(
            "Final synthesis completed with an inconsistent diagnostics state."
        )
    payload = result.model_dump(mode="json")
    _assign_synthesis_status_payload(
        payload,
        status=status,
        diagnostics_dir=diagnostics_dir,
    )
    if status == "degraded_fallback":
        payload["repair_attempted"] = diagnostics.repair_attempted
        if not (payload.get("schema_version") == "2.0" and payload.get("evidence_catalog")):
            _complete_degraded_result_structure(payload)
    for section in payload.get("sections") or []:
        if section.get("section_id") != "executive_summary":
            continue
        summary = str(section.get("summary") or "")
        if summary.startswith("Synthesis status:"):
            section["summary"] = re.sub(
                r"^Synthesis status:\s*[^.]+",
                f"Synthesis status: {status}",
                summary,
            )
        break
    return FinalStrategyResult.model_validate(payload)


def _complete_degraded_result_structure(payload: dict[str, Any]) -> None:
    """Complete an already-grounded degraded result without inventing evidence."""

    canonical, _ = canonicalize_final_strategy_actions(payload)
    action_ids = [
        str(action.get("action_id"))
        for action in canonical.get("actions") or []
        if isinstance(action, dict) and action.get("action_id")
    ]
    if not action_ids:
        raise ValueError("A degraded final strategy requires at least one grounded action")
    experiment_ids = [
        str(action.get("experiment_id"))
        for action in canonical.get("actions") or []
        if isinstance(action, dict) and action.get("experiment_id")
    ]
    section_by_id = {
        str(section.get("section_id") or ""): dict(section)
        for section in canonical.get("sections") or []
        if isinstance(section, dict) and section.get("section_id")
    }
    top_action = next(
        (
            str(action.get("action"))
            for priority in ("P0", "P1", "P2", "P3")
            for action in canonical.get("actions") or []
            if isinstance(action, dict) and action.get("priority") == priority
        ),
        "No evidence-backed next step is available.",
    )
    metric = _as_dict(canonical.get("metric"))
    baseline_available = any(
        str(ref).startswith("baseline_evidence")
        for action in canonical.get("actions") or []
        if isinstance(action, dict)
        for ref in action.get("evidence_refs") or []
    )
    ordered: list[dict[str, Any]] = []
    for section_id in REQUIRED_SECTION_IDS:
        section = section_by_id.get(section_id, {
            "section_id": section_id,
            "title": _title_from_id(section_id),
            "summary": f"not_available: {section_id} was absent from the grounded degraded result.",
            "action_ids": [],
            "evidence_refs": ["final_synthesizer.repaired"],
            "availability": "not_available",
            "limitations": ["No validated evidence was available for this section."],
        })
        if section_id == "executive_summary":
            section["availability"] = "limited"
            section["summary"] = (
                "Degraded fallback status: the LLM synthesis did not yield a fully "
                "supported strategy. "
                f"Task type: {canonical.get('task_type') or 'unknown'}. "
                f"Metric: {metric.get('name') or metric.get('metric_name') or 'unknown'}. "
                f"Primary validation: {canonical.get('recommended_validation') or 'not_available'}. "
                "Highest-priority safety concern remains the highest-priority grounded "
                f"action. Baseline availability: {'available' if baseline_available else 'not_available'}. "
                f"Top evidence-backed next step: {top_action}"
            )
        if section_id == "first_48_hours" and not section.get("time_blocks"):
            section["time_blocks"] = [
                {
                    "time_window": window,
                    "summary": summary,
                    "action_ids": [action_ids[min(index, len(action_ids) - 1)]],
                    "experiment_ids": experiment_ids[index:index + 1],
                }
                for index, (window, summary) in enumerate((
                    ("0-4_hours", "Lock the surviving metric, validation, and safety contracts."),
                    ("4-12_hours", "Execute the first grounded action without broadening scope."),
                    ("12-24_hours", "Record validation evidence for the next grounded action."),
                    ("24-48_hours", "Review grounded results and retain only supported changes."),
                ))
            ]
        ordered.append(section)
    canonical["sections"] = ordered
    payload.clear()
    payload.update(canonical)


def collect_schema_issues(
    exc: ValidationError,
    *,
    stage: ValidationStage,
) -> list[ValidationIssue]:
    issues = [
        normalize_validation_error(error, stage=stage)
        for error in exc.errors(
            include_url=False,
            include_context=False,
            include_input=True,
        )
    ]
    return _sorted_validation_issues(issues)


def _collect_compilation_schema_issues(
    exc: FinalStrategySchemaValidationError,
    *,
    stage: ValidationStage,
) -> list[ValidationIssue]:
    if isinstance(exc.__cause__, ValidationError):
        return collect_schema_issues(exc.__cause__, stage=stage)
    return _sorted_validation_issues([
        ValidationIssue(
            stage=stage,
            issue_type=item.error_type,
            field_path=".".join(str(part) for part in item.location) or None,
            message=_bounded_message(item.message),
            expected_contract="FinalStrategyResult",
        )
        for item in exc.diagnostics.schema_validation_errors
    ])


def collect_reference_issues(
    errors: list[Any],
    *,
    stage: ValidationStage,
) -> list[ValidationIssue]:
    issues = [ValidationIssue(
        stage=stage,
        issue_type=str(getattr(error, "reason", None) or "invalid_reference"),
        field_path=str(getattr(error, "field_path", "")) or None,
        message=_bounded_message(
            "Invalid reference"
            + (
                f" {getattr(error, 'invalid_value')!r}"
                if getattr(error, "invalid_value", None) is not None
                else ""
            )
            + (
                f": {getattr(error, 'reason')}"
                if getattr(error, "reason", None)
                else ""
            )
        ),
        invalid_value_type=(
            type(getattr(error, "invalid_value")).__name__
            if getattr(error, "invalid_value", None) is not None
            else None
        ),
        invalid_reference=(
            _bounded_message(str(getattr(error, "invalid_value")), limit=200)
            if getattr(error, "invalid_value", None) is not None
            else None
        ),
        expected_contract=(
            str(getattr(error, "expected_namespace"))
            if getattr(error, "expected_namespace", None)
            else "known final synthesis reference"
        ),
    ) for error in errors]
    return _sorted_validation_issues(issues)


def _collect_evidence_resolution_issues(
    report: FinalStrategyActionEvidenceReport,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for action in report.actions:
        field_path = f"actions[{action.action_id}].evidence_refs"
        if not action.original_refs and action.added_refs:
            issues.append(ValidationIssue(
                stage="llm_reference_validation",
                issue_type="missing_evidence_reference",
                field_path=field_path,
                message="Action had no usable evidence reference; deterministic resolution added one.",
                invalid_value_type="list",
                expected_contract="at least one allowed evidence reference",
            ))
        for reference in sorted(set(action.unresolved_refs) | set(action.unsupported_refs)):
            issues.append(ValidationIssue(
                stage="llm_reference_validation",
                issue_type="invalid_evidence_reference",
                field_path=field_path,
                message="Action referenced evidence outside its allowed support contract.",
                invalid_value_type="str",
                invalid_reference=_bounded_message(reference, limit=200),
                expected_contract="allowed evidence reference for action intent",
            ))
    if report.fallback_action_ids:
        issues.append(ValidationIssue(
            stage="llm_schema_validation",
            issue_type="missing_usable_actions",
            field_path="actions",
            message="No usable LLM action remained; deterministic fallback actions were generated.",
            invalid_value_type="list",
            expected_contract="at least one valid FinalStrategyAction",
        ))
    return _sorted_validation_issues(issues)


def build_attempt_diagnostic(
    *,
    attempt: str,
    model: str | None,
    output: Any | None = None,
    issues: list[ValidationIssue] | None = None,
) -> SynthesisAttemptDiagnostic:
    received = output is not None
    return SynthesisAttemptDiagnostic(
        attempt=attempt,
        model=model,
        output_received=received,
        json_parse_succeeded=received,
        output_hash=_payload_hash(output) if received else None,
        issues=_sorted_validation_issues(issues or []),
    )


def write_final_synthesis_diagnostics(
    diagnostics_dir: Path | None,
    diagnostics: FinalSynthesisDiagnostics,
) -> None:
    payload = diagnostics.model_dump(mode="json")
    payload["fallback_reason"] = (
        _bounded_message(payload["fallback_reason"])
        if payload.get("fallback_reason")
        else None
    )
    for attempt in payload["attempts"]:
        for issue in attempt["issues"]:
            issue["message"] = _bounded_message(str(issue.get("message") or ""))
            if issue.get("invalid_reference") is not None:
                issue["invalid_reference"] = _bounded_message(
                    str(issue["invalid_reference"]), limit=200
                )
        attempt["issues"] = sorted(
            attempt["issues"],
            key=lambda issue: (
                issue.get("stage") or "",
                issue.get("field_path") or "",
                issue.get("issue_type") or "",
                issue.get("invalid_reference") or "",
                issue.get("message") or "",
            ),
        )
        attempt["warnings"] = sorted({
            _bounded_message(str(warning)) for warning in attempt["warnings"]
        })
    for group_name in ("selection_attempts", "rendering_attempts"):
        for attempt in payload.get(group_name) or []:
            for issue in attempt.get("issues") or []:
                issue["message"] = _bounded_message(
                    str(issue.get("message") or "")
                )
            attempt["warnings"] = sorted({
                _bounded_message(str(warning))
                for warning in attempt.get("warnings") or []
            })
    for failure in payload.get("provider_failures") or []:
        failure["message"] = _bounded_message(str(failure.get("message") or ""))
    _write_diagnostic_json(
        diagnostics_dir, "final_synthesis_diagnostics.json", payload
    )


def _attempt_diagnostic(
    diagnostics: FinalSynthesisDiagnostics,
    attempt: str,
) -> SynthesisAttemptDiagnostic:
    existing = next(
        (item for item in diagnostics.attempts if item.attempt == attempt), None
    )
    if existing is not None:
        return existing
    created = SynthesisAttemptDiagnostic(attempt=attempt)
    diagnostics.attempts.append(created)
    return created


def _ensure_repair_attempt(
    diagnostics: FinalSynthesisDiagnostics,
    *,
    model: str | None,
) -> SynthesisAttemptDiagnostic:
    diagnostics.repair_attempted = True
    attempt = _attempt_diagnostic(diagnostics, "deterministic_repair")
    if attempt.model is None:
        attempt.model = model
    return attempt


def _record_issue(
    diagnostics: FinalSynthesisDiagnostics,
    attempt: str,
    issue: ValidationIssue,
) -> None:
    _attempt_diagnostic(diagnostics, attempt).issues.append(issue)


def _record_reference_issues(
    diagnostics: FinalSynthesisDiagnostics,
    attempt: str,
    issues: list[Any],
) -> None:
    stage: ValidationStage = (
        "llm_reference_validation"
        if attempt == "initial_llm"
        else "repair_reference_validation"
    )
    target = _attempt_diagnostic(diagnostics, attempt)
    target.reference_validation_succeeded = False
    target.issues.extend(collect_reference_issues(issues, stage=stage))


def _refresh_diagnostics_summary(
    diagnostics: FinalSynthesisDiagnostics,
) -> None:
    if diagnostics.protocol == "two_call":
        diagnostics.initial_output_valid = diagnostics.selection_status == "llm_success"
        diagnostics.repair_attempted = diagnostics.selection_status == "repaired_success"
        diagnostics.repair_succeeded = diagnostics.selection_status == "repaired_success"
        diagnostics.fallback_required = diagnostics.selection_status == "degraded_fallback"
        return
    initial = _attempt_diagnostic(diagnostics, "initial_llm")
    diagnostics.initial_output_valid = bool(
        initial.output_received
        and initial.json_parse_succeeded
        and initial.schema_validation_succeeded
        and initial.reference_validation_succeeded
        and not initial.issues
    )
    diagnostics.repair_attempted = any(
        attempt.attempt == "deterministic_repair"
        for attempt in diagnostics.attempts
    )
    repair = next(
        (
            attempt
            for attempt in diagnostics.attempts
            if attempt.attempt == "deterministic_repair"
        ),
        None,
    )
    if repair is not None and not diagnostics.fallback_required:
        diagnostics.repair_succeeded = bool(
            repair.output_received
            and repair.json_parse_succeeded
            and repair.schema_validation_succeeded
            and repair.reference_validation_succeeded
        )


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded_message(value: str, *, limit: int = 500) -> str:
    sanitized = re.sub(
        r"(?i)authorization\s*[:=]\s*(?:bearer\s+)?\S+",
        "authorization=[REDACTED]",
        value,
    )
    sanitized = re.sub(
        r"(?i)(authorization|api[_-]?key|bearer|access[_-]?token|password)"
        r"\s*[:=]?\s*\S+",
        r"\1=[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(r"(?i)\bsk-[a-z0-9_-]{8,}\b", "[REDACTED]", sanitized)
    sanitized = " ".join(sanitized.split())
    return sanitized if len(sanitized) <= limit else sanitized[: limit - 1] + "…"


def _looks_like_parse_failure(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "json" in text or "parse" in text


def _sorted_validation_issues(
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    return sorted(
        issues,
        key=lambda issue: (
            issue.stage,
            issue.field_path or "",
            issue.issue_type,
            issue.invalid_reference or "",
            issue.message,
        ),
    )


def _write_validation_errors(
    diagnostics_dir: Path | None,
    error: ValidationError,
    *,
    phase: str,
) -> None:
    _write_diagnostic_json(
        diagnostics_dir,
        "final_strategy_validation_errors.json",
        [{
            "phase": phase,
            "error_type": type(error).__name__,
            "errors": error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            ),
        }],
    )


def _write_diagnostic_json(
    diagnostics_dir: Path | None,
    name: str,
    payload: Any,
) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / name
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def repair_final_strategy_payload(
    payload: dict[str, Any],
    *,
    research_hypotheses: list[dict[str, Any]],
    eda_evidence_pack: dict[str, Any] | None,
    eda_summary: str | None = None,
) -> dict[str, Any]:
    repaired = deepcopy(payload)
    eda = eda_evidence_pack or {}
    known_ids, category_ids = _build_hypothesis_lookup(research_hypotheses, eda)
    known_set = set(known_ids)
    repair_needed = False

    sections = _normalize_sections(repaired.get("sections", []))
    top_actions = _normalize_actions(repaired.get("actions", []))
    repaired_actions: list[dict[str, Any]] = []
    for action in top_actions:
        fixed, changed = _repair_action_payload(
            action,
            known_ids=known_ids,
            category_ids=category_ids,
            section_context="",
        )
        repaired_actions.append(fixed)
        repair_needed = repair_needed or changed

    repaired_sections: list[dict[str, Any]] = []
    for index, section in enumerate(sections, start=1):
        fixed_section = dict(section)
        section_id = str(fixed_section.get("section_id") or f"section_{index}").strip()
        title = str(fixed_section.get("title") or _title_from_id(section_id)).strip()
        summary = str(
            fixed_section.get("summary")
            or "This section was normalized from the final synthesis response."
        ).strip()
        repair_needed = repair_needed or not all(
            [fixed_section.get("section_id"), fixed_section.get("title"), fixed_section.get("summary")]
        )
        fixed_section.update(section_id=section_id, title=title, summary=summary)

        section_actions = []
        for action in _normalize_actions(fixed_section.get("actions", [])):
            fixed_action, changed = _repair_action_payload(
                action,
                known_ids=known_ids,
                category_ids=category_ids,
                section_context=f"{section_id} {title}",
            )
            section_actions.append(fixed_action)
            repair_needed = repair_needed or changed
        fixed_section["actions"] = section_actions
        fixed_section["related_hypothesis_ids"] = _known_only(
            fixed_section.get("related_hypothesis_ids"), known_set
        )
        fixed_section["evidence_refs"] = _string_values(fixed_section.get("evidence_refs"))

        if not section_actions and not fixed_section["evidence_refs"]:
            inferred_refs = _section_evidence_refs(section_id, title, eda)
            if inferred_refs:
                fixed_section["evidence_refs"] = inferred_refs
            else:
                placeholder = _placeholder_action(known_ids)
                if placeholder is not None:
                    fixed_section["actions"] = [placeholder]
                else:
                    fixed_section["evidence_refs"] = ["final_synthesizer.repaired"]
            repair_needed = True
        repaired_sections.append(fixed_section)

    repaired["sections"] = repaired_sections
    repaired["actions"] = repaired_actions
    if repair_needed:
        limitations = _string_values(repaired.get("limitations"))
        if REPAIR_LIMITATION not in limitations:
            limitations.append(REPAIR_LIMITATION)
        repaired["limitations"] = limitations

    # Unknown IDs are unsupported even though they satisfy the string schema.
    for action in _iter_action_payloads(repaired):
        action["related_hypothesis_ids"] = _known_only(
            [
                *_string_values(action.get("related_hypothesis_ids")),
                *_string_values(action.get("hypothesis_ids")),
            ],
            known_set,
        )
        if not action["related_hypothesis_ids"]:
            inferred = _infer_hypothesis_ids(
                action,
                known_ids=known_ids,
                category_ids=category_ids,
                section_context="",
            )
            action["related_hypothesis_ids"] = inferred
        action["hypothesis_ids"] = list(action["related_hypothesis_ids"])
    _assign_synthesis_status_payload(
        repaired,
        status="repaired_success",
        diagnostics_dir=None,
    )
    return repaired


def build_fallback_final_strategy(
    *,
    competition_id: str,
    research_hypotheses: list[dict[str, Any]],
    eda_evidence_pack: dict[str, Any] | None,
    eda_summary: str | None = None,
    task_type: str | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    eda = eda_evidence_pack or {}
    known_ids, category_ids = _build_hypothesis_lookup(research_hypotheses, eda)
    if not known_ids:
        raise ValueError(
            "Cannot build a linked final strategy without any known Scout or EDA hypothesis IDs."
        )
    source_links, eda_links = _input_provenance_payloads(
        research_hypotheses,
        eda,
    )

    metric = _as_dict(eda.get("metric_evidence"))
    task_type = str(
        task_type or metric.get("task_type") or eda.get("task_type") or "unknown"
    )
    metric_name = str(
        metric_name or metric.get("metric_name") or metric.get("name") or "unknown"
    )
    validation = _as_dict(eda.get("validation_evidence"))
    primary_validation = _as_dict(validation.get("primary_validation"))
    method = str(primary_validation.get("method") or "").strip()
    baseline = _as_dict(eda.get("baseline_evidence"))
    baseline_status = str(baseline.get("status") or "not_available").lower()
    baseline_completed = baseline_status in {
        "completed", "complete", "success", "succeeded", "successful",
    }
    strategy_context = StrategyContext.from_evidence(
        competition_id=competition_id,
        evidence_pack=eda,
        task_type=task_type,
        metric_name=metric_name,
        validation_strategy=method,
    )
    compiled_strategy = compile_competition_strategy(strategy_context)

    actions_by_section: dict[str, list[dict[str, Any]]] = {
        section_id: [] for section_id in REQUIRED_SECTION_IDS
    }
    summaries: dict[str, str] = {}
    availability: dict[str, str] = {
        section_id: "available" for section_id in REQUIRED_SECTION_IDS
    }
    section_limitations: dict[str, list[str]] = {
        section_id: [] for section_id in REQUIRED_SECTION_IDS
    }

    schema = _as_dict(eda.get("inferred_schema"))
    primary_id = schema.get("primary_id_column")
    if not primary_id and isinstance(schema.get("global_roles"), dict):
        primary_id = schema["global_roles"].get("primary_id_column")
    if schema:
        actions_by_section["dataset_facts_from_eda"].append(
            _fallback_action(
                "Inspect and preserve the EDA-inferred train, test, target, and identifier roles.",
                "The schema contract controls feature selection and submission alignment.",
                ["inferred_schema"],
                _ids_for_categories(category_ids, known_ids, "schema"),
            )
        )
        summaries["dataset_facts_from_eda"] = (
            "Dataset roles come from the validated inferred schema and must be "
            "preserved during feature and submission construction."
        )
    else:
        _mark_fallback_section_unavailable(
            "dataset_facts_from_eda", summaries, availability, section_limitations,
            "not_available: inferred schema evidence was unavailable.",
        )
    if primary_id:
        actions_by_section["what_not_to_do"].append(
            _fallback_action(
                f"Exclude `{primary_id}` from model features unless explicitly justified.",
                "EDA identified this column as the primary identifier.",
                ["inferred_schema.primary_id_column"],
                _ids_for_categories(category_ids, known_ids, "schema", "leakage"),
                priority="P0",
            )
        )

    if metric.get("requires_probabilities"):
        actions_by_section["metric_and_validation"].append(
            _fallback_action(
                "Output probabilities or scores rather than hard labels.",
                "The competition metric requires probabilistic predictions.",
                ["metric_evidence.requires_probabilities"],
                _ids_for_categories(category_ids, known_ids, "metric"),
                priority="P0",
            )
        )
    if metric.get("requires_threshold"):
        actions_by_section["metric_and_validation"].append(
            _fallback_action(
                "Tune the decision threshold only on validation folds.",
                "Threshold selection on training or test labels would bias evaluation.",
                ["metric_evidence.requires_threshold"],
                _ids_for_categories(category_ids, known_ids, "metric", "validation"),
                priority="P0",
            )
        )

    if method:
        actions_by_section["metric_and_validation"].append(
            _fallback_action(
                f"Use {method} as the primary validation method.",
                "EDA selected this method from the observed dataset structure.",
                ["validation_evidence.primary_validation"],
                _ids_for_categories(category_ids, known_ids, "validation"),
                priority="P0",
                validation_strategy=str(method),
            )
        )
    if metric or method:
        summaries["metric_and_validation"] = (
            f"Metric: {metric_name}. Primary validation: {method or 'not_available'}. "
            "Only contract-backed output and validation requirements are included."
        )
    else:
        _mark_fallback_section_unavailable(
            "metric_and_validation", summaries, availability, section_limitations,
            "not_available: metric and primary-validation evidence were unavailable.",
        )

    if _has_unsafe_target_encoding(eda):
        target_encoding = _fallback_action(
            "Avoid naive target encoding; use OOF or fold-fitted encoding only.",
            "EDA identified leakage risk in globally fitted target statistics.",
            ["feature_probe_evidence", "leakage_evidence"],
            _ids_for_categories(category_ids, known_ids, "leakage", "schema"),
            priority="P0",
        )
        actions_by_section["leakage_and_data_quality"].append(target_encoding)
    safety_rules = _fallback_safety_constraint_actions(
        eda, known_ids=known_ids, category_ids=category_ids
    )
    actions_by_section["what_not_to_do"].extend(safety_rules)
    if eda.get("leakage_evidence") or _has_unsafe_target_encoding(eda):
        summaries["leakage_and_data_quality"] = (
            "Leakage and data-quality guidance includes only observed checks and "
            "validated safety constraints."
        )
    else:
        _mark_fallback_section_unavailable(
            "leakage_and_data_quality", summaries, availability, section_limitations,
            "not_available: leakage diagnostics were unavailable or skipped.",
        )

    drift = _as_dict(eda.get("drift_evidence"))
    severity = drift.get("feature_drift_severity") or drift.get("severity")
    if severity in {"medium", "high", "critical"}:
        severity_ref = (
            "drift_evidence.feature_drift_severity"
            if drift.get("feature_drift_severity") is not None
            else "drift_evidence.severity"
        )
        actions_by_section["drift_and_leaderboard_risk"].append(
            _fallback_action(
                "Treat train/test drift as a leaderboard-risk diagnostic.",
                f"EDA reported {severity} drift severity.",
                [severity_ref],
                _ids_for_categories(category_ids, known_ids, "drift"),
                priority="P1",
            )
        )
        summaries["drift_and_leaderboard_risk"] = (
            f"Validated drift diagnostics report {severity} severity; treat this as "
            "diagnostic evidence rather than a guaranteed leaderboard effect."
        )
    elif drift:
        drift_status = str(drift.get("status") or "not_testable")
        availability["drift_and_leaderboard_risk"] = "limited"
        summaries["drift_and_leaderboard_risk"] = (
            f"not_available: drift severity was not established (status={drift_status})."
        )
        section_limitations["drift_and_leaderboard_risk"].append(
            "No medium/high/critical drift conclusion is supported."
        )
    else:
        _mark_fallback_section_unavailable(
            "drift_and_leaderboard_risk", summaries, availability, section_limitations,
            "not_available: drift diagnostics were unavailable or skipped.",
        )

    high_potential_families = _high_potential_feature_families(eda)
    for compiled_action in compiled_strategy.actions:
        if compiled_action.section_id not in actions_by_section:
            continue
        actions_by_section[compiled_action.section_id].append(
            _compiled_action_payload(
                compiled_action,
                known_ids=known_ids,
                category_ids=category_ids,
            )
        )
    compiled_feature_actions = [
        action for action in compiled_strategy.actions
        if action.section_id == "feature_priorities"
    ]
    if compiled_feature_actions:
        summaries["feature_priorities"] = (
            "Competition-specific feature experiments were compiled only from actual "
            "schema columns, feature diagnostics, probes, and baseline ablations."
        )
    elif eda.get("feature_probe_evidence"):
        availability["feature_priorities"] = "limited"
        summaries["feature_priorities"] = (
            "No feature family was validated as high-potential; retain probes as "
            "diagnostics rather than priorities."
        )
    else:
        _mark_fallback_section_unavailable(
            "feature_priorities", summaries, availability, section_limitations,
            "not_available: feature probes were unavailable.",
        )

    if baseline_completed:
        baseline_refs = ["baseline_evidence.status"]
        for field in ("model_type", "metric_name", "metric_value", "validation_policy"):
            if baseline.get(field) is not None:
                baseline_refs.append(f"baseline_evidence.{field}")
        actions_by_section["baseline_findings"].append(
            _fallback_action(
                "Use the completed EDA baseline as the comparison anchor for experiments.",
                "The baseline completed under the recorded model, metric, and validation configuration.",
                baseline_refs,
                _ids_for_categories(category_ids, known_ids, "metric", "validation", "baseline"),
            )
        )
        summaries["baseline_findings"] = (
            f"Baseline status: {baseline_status}; model={baseline.get('model_type') or 'not_recorded'}; "
            f"metric={baseline.get('metric_name') or metric_name}; "
            f"value={baseline.get('metric_value', 'not_recorded')}."
        )
    else:
        availability["baseline_findings"] = "not_available"
        reason = str(baseline.get("reason") or "baseline evidence was unavailable")
        summaries["baseline_findings"] = (
            f"not_available: baseline status={baseline_status}; {reason}."
        )
        section_limitations["baseline_findings"].append(
            "No completed EDA baseline exists, so the fallback does not recommend reproducing it."
        )

    model_action, model_summary, model_limitation = _fallback_modeling_plan(
        task_type=task_type,
        baseline=baseline,
        baseline_completed=baseline_completed,
        method=method,
        metric=metric,
        known_ids=known_ids,
        category_ids=category_ids,
    )
    if model_action:
        actions_by_section["modeling_plan"].append(model_action)
    else:
        availability["modeling_plan"] = "not_available"
    summaries["modeling_plan"] = model_summary
    if model_limitation:
        section_limitations["modeling_plan"].append(model_limitation)

    submission_action = _fallback_submission_action(
        schema=schema,
        known_ids=known_ids,
        category_ids=category_ids,
    )
    if submission_action is not None:
        actions_by_section["first_48_hours"].append(submission_action)

    rejected_actions = _fallback_rejected_hypothesis_actions(eda, known_ids)
    actions_by_section["what_not_to_do"].extend(rejected_actions)
    if actions_by_section["what_not_to_do"]:
        summaries["what_not_to_do"] = (
            "Deduplicated safety constraints and rejected-hypothesis implications "
            "that must not be promoted into modeling assumptions."
        )
    else:
        _mark_fallback_section_unavailable(
            "what_not_to_do", summaries, availability, section_limitations,
            "not_available: no validated safety constraint or rejected hypothesis implication was present.",
        )

    experiment_actions = _fallback_experiment_actions(
        eda=eda,
        task_type=task_type,
        metric_name=metric_name,
        method=method,
        baseline=baseline,
        baseline_completed=baseline_completed,
        feature_families=[],
        known_ids=known_ids,
        category_ids=category_ids,
    )
    compiled_experiment_actions = [
        _compiled_experiment_action_payload(
            experiment,
            known_ids=known_ids,
            category_ids=category_ids,
        )
        for experiment in compiled_strategy.experiments
    ]
    experiment_actions.extend(compiled_experiment_actions)
    experiment_actions = _ordered_fallback_experiment_actions(experiment_actions)
    actions_by_section["experiments_queue"].extend(experiment_actions)
    if experiment_actions:
        summaries["experiments_queue"] = (
            "Ordered executable experiments derived from validated validation, baseline, "
            "metric, and feature evidence."
        )
    else:
        _mark_fallback_section_unavailable(
            "experiments_queue", summaries, availability, section_limitations,
            "not_available: no executable experiment could be specified without inventing evidence.",
        )

    highest_safety = next(
        (
            action["action"]
            for section_id in ("leakage_and_data_quality", "what_not_to_do")
            for action in actions_by_section[section_id]
            if action.get("priority") == "P0"
        ),
        "No P0 leakage/safety concern was validated.",
    )
    top_step = next(
        (
            action["action"]
            for priority in ("P0", "P1", "P2", "P3")
            for section_id in REQUIRED_SECTION_IDS
            for action in actions_by_section[section_id]
            if action.get("priority") == priority
        ),
        "No evidence-backed modeling step is available.",
    )
    summaries["executive_summary"] = (
        "Degraded fallback status: the LLM synthesis was invalid and this report was "
        f"assembled deterministically. Task type: {task_type}. Metric: {metric_name}. "
        f"Primary validation: {method or 'not_available'}. Highest-priority safety concern: "
        f"{highest_safety} Baseline availability: {baseline_status}. "
        f"Top evidence-backed next step: {top_step}"
    )
    availability["executive_summary"] = "limited"
    section_limitations["executive_summary"].append(FALLBACK_LIMITATION)

    all_actions = [
        action
        for section_id in REQUIRED_SECTION_IDS
        if section_id != "first_48_hours"
        for action in actions_by_section[section_id]
    ]
    anchor_action = next((action for action in all_actions if action.get("action_id")), None)
    if anchor_action is None:
        anchor_action = _fallback_action(
            "Pause evidence-dependent modeling until the missing EDA contracts are available.",
            "Sparse evidence cannot safely support model or feature recommendations.",
            ["final_synthesizer.repaired"],
            [known_ids[0]],
            priority="P0",
            confidence="high",
            evidence_origin="Safety-warning",
            limitations=["Operational work is evidence-gated in degraded fallback mode."],
        )
        actions_by_section["first_48_hours"].append(anchor_action)
        all_actions.append(anchor_action)
    time_blocks = _fallback_first_48_hour_blocks(
        actions_by_section=actions_by_section,
        experiment_actions=experiment_actions,
        anchor_action_id=str(anchor_action["action_id"]),
    )
    summaries["first_48_hours"] = (
        "Ordered 0-48 hour execution sequence. Every block references an existing "
        "strategy action or structured experiment."
    )

    sections = []
    all_actions = []
    for section_id in REQUIRED_SECTION_IDS:
        section_actions = [
            action for action in actions_by_section[section_id] if action is not None
        ]
        all_actions.extend(deepcopy(section_actions))
        evidence_refs = _unique_strings(
            ref for action in section_actions for ref in action["evidence_refs"]
        )
        if not evidence_refs and availability[section_id] != "available":
            evidence_refs = ["final_synthesizer.repaired"]
        section_payload = {
            "section_id": section_id,
            "title": _title_from_id(section_id),
            "summary": summaries.get(
                section_id,
                "Limited deterministic guidance from validated Scout and EDA evidence.",
            ),
            "actions": section_actions,
            "evidence_refs": evidence_refs,
            "related_hypothesis_ids": _unique_strings(
                item
                for action in section_actions
                for item in action["related_hypothesis_ids"]
            ),
            "availability": availability[section_id],
            "limitations": section_limitations[section_id],
        }
        if section_id == "first_48_hours":
            section_payload["time_blocks"] = time_blocks
        sections.append(
            section_payload
        )

    fallback = {
        "schema_version": "1.0",
        "competition_id": competition_id,
        "task_type": task_type,
        "metric": {"name": metric_name},
        "recommended_validation": method or None,
        "sections": sections,
        "actions": all_actions,
        "experiments": _final_strategy_experiments(
            experiment_actions,
            compiled_strategy.experiments,
            task_type=task_type,
            metric_name=metric_name,
            method=method,
            baseline=baseline,
        ),
        "source_to_hypothesis_links": source_links,
        "hypothesis_to_eda_links": eda_links,
        "selected_validation_requirement_ids": [
            str(item["validation_requirement_id"])
            for item in eda.get("validation_requirements") or []
            if isinstance(item, dict)
            and item.get("validation_requirement_id")
            and item.get("mandatory", True)
        ],
        "enforced_safety_constraint_ids": [
            str(item["safety_constraint_id"])
            for item in eda.get("safety_constraints") or []
            if isinstance(item, dict)
            and item.get("safety_constraint_id")
            and item.get("blocking", False)
        ],
        "limitations": [FALLBACK_LIMITATION],
    }
    _assign_synthesis_status_payload(
        fallback,
        status="degraded_fallback",
        diagnostics_dir=None,
    )
    fallback, _ = canonicalize_semantic_strategy_actions(
        fallback,
        primary_id=_primary_id_from_eda(eda),
    )
    return fallback


def _mark_fallback_section_unavailable(
    section_id: str,
    summaries: dict[str, str],
    availability: dict[str, str],
    limitations: dict[str, list[str]],
    statement: str,
) -> None:
    availability[section_id] = "not_available"
    summaries[section_id] = statement
    limitations[section_id].append(statement)


def _high_potential_feature_families(eda: dict[str, Any]) -> list[str]:
    families: list[str] = []
    for raw_probe in eda.get("feature_probe_evidence") or []:
        probe = _to_dict(raw_probe)
        if not (
            probe.get("potential") == "high"
            or probe.get("status") == "high_potential"
            or probe.get("high_potential") is True
        ):
            continue
        family = str(probe.get("feature_family") or "").strip()
        if family and family not in families:
            families.append(family)
    return families


def _fallback_safety_constraint_actions(
    eda: dict[str, Any],
    *,
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen_rules: set[str] = set()
    for raw_constraint in eda.get("safety_constraints") or []:
        constraint = _to_dict(raw_constraint)
        rule = str(constraint.get("rule") or "").strip()
        constraint_id = str(constraint.get("safety_constraint_id") or "").strip()
        normalized = re.sub(r"\W+", " ", rule.lower()).strip()
        if not rule or not constraint_id or normalized in seen_rules:
            continue
        seen_rules.add(normalized)
        refs = _fallback_valid_evidence_refs(eda, constraint.get("evidence_refs") or [])
        action = _fallback_action(
            rule,
            str(constraint.get("reason") or "Validated EDA safety constraint."),
            refs or ["final_synthesizer.repaired"],
            _ids_for_categories(category_ids, known_ids, "leakage", "schema", "validation"),
            priority="P0" if constraint.get("blocking", True) else "P1",
            confidence="high",
            evidence_origin="Safety-warning",
            safety_constraint_ids=[constraint_id],
        )
        actions.append(action)
    return actions


def _fallback_rejected_hypothesis_actions(
    eda: dict[str, Any],
    known_ids: list[str],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_result in eda.get("hypothesis_results") or []:
        result = _to_dict(raw_result)
        if str(result.get("status") or "") != "rejected":
            continue
        hypothesis_id = str(result.get("hypothesis_id") or "").strip()
        if hypothesis_id not in known_ids:
            continue
        implication = str(
            result.get("impact_on_strategy")
            or "Do not promote this rejected hypothesis into the strategy."
        ).strip()
        text = f"Do not adopt rejected hypothesis `{hypothesis_id}`: {implication}"
        normalized = re.sub(r"\W+", " ", text.lower()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        refs = _fallback_valid_evidence_refs(eda, result.get("evidence_refs") or [])
        actions.append(_fallback_action(
            text,
            str(result.get("finding") or "EDA rejected the hypothesis."),
            refs or [f"hypothesis_results.{hypothesis_id}"],
            [hypothesis_id],
            priority="P0",
            confidence=str(result.get("confidence_after_eda") or "medium"),
            evidence_origin="Safety-warning",
        ))
    return actions


def _fallback_modeling_plan(
    *,
    task_type: str,
    baseline: dict[str, Any],
    baseline_completed: bool,
    method: str,
    metric: dict[str, Any],
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, str, str | None]:
    if baseline_completed and baseline.get("model_type"):
        model_type = str(baseline["model_type"])
        refs = ["baseline_evidence.status", "baseline_evidence.model_type"]
        return (
            _fallback_action(
                f"Keep `{model_type}` as the recorded baseline model family for controlled comparisons.",
                "The completed EDA baseline is the only model family with current-dataset execution evidence.",
                refs,
                _ids_for_categories(category_ids, known_ids, "baseline", "metric", "validation"),
                priority="P1",
                confidence="high",
            ),
            f"Use the completed baseline model family `{model_type}` as the comparison anchor; no broader model ranking is inferred.",
            None,
        )
    registry_family = _deterministic_registry_model_family(task_type)
    if registry_family and method:
        refs = ["validation_evidence.primary_validation"]
        if metric.get("task_type") is not None:
            refs.append("metric_evidence.task_type")
        else:
            refs.append("final_synthesizer.repaired")
        return (
            _fallback_action(
                f"Use the project deterministic `{registry_family}` family only to establish a diagnostic baseline.",
                "The project baseline runner registers this task-appropriate safe family, but no current-dataset model comparison exists.",
                refs,
                _ids_for_categories(category_ids, known_ids, "baseline", "metric", "validation"),
                priority="P1",
                confidence="low",
                evidence_origin="Fallback-generated",
                limitations=["Model-family selection is registry-backed, not supported by comparative EDA evidence."],
            ),
            f"A `{registry_family}` diagnostic baseline is permitted by the deterministic project registry; model selection is otherwise not evidence-backed.",
            "No completed baseline or comparative model evidence is available.",
        )
    return (
        None,
        "not_available: model selection is not evidence-backed for this task and no applicable deterministic baseline registry policy can be executed.",
        "No supported model family is selected in degraded fallback mode.",
    )


def _deterministic_registry_model_family(task_type: str) -> str | None:
    normalized = task_type.strip().lower()
    if normalized in {"binary_classification", "multiclass_classification", "classification"}:
        return "LogisticRegression/HistGradientBoostingClassifier"
    if normalized == "regression":
        return "LinearRegression/HistGradientBoostingRegressor"
    return None


def _fallback_submission_action(
    *,
    schema: dict[str, Any],
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> dict[str, Any] | None:
    primary_id = str(schema.get("primary_id_column") or "").strip()
    prediction_column = str(
        schema.get("prediction_column") or schema.get("target_column") or ""
    ).strip()
    if not primary_id or not prediction_column:
        return None
    prediction_ref = (
        "inferred_schema.prediction_column"
        if schema.get("prediction_column")
        else "inferred_schema.target_column"
    )
    return _fallback_action(
        f"After model and postprocessing decisions are frozen, assemble the submission with `{primary_id}` and `{prediction_column}` and verify row alignment before export.",
        "The inferred schema defines the identifier and prediction-output columns; this is an artifact-integrity check, not a performance claim.",
        ["inferred_schema.primary_id_column", prediction_ref],
        _ids_for_categories(category_ids, known_ids, "schema", "validation"),
        priority="P0",
        confidence="high",
        evidence_origin="Safety-warning",
    )


def _fallback_experiment_actions(
    *,
    eda: dict[str, Any],
    task_type: str,
    metric_name: str,
    method: str,
    baseline: dict[str, Any],
    baseline_completed: bool,
    feature_families: list[str],
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if not method:
        return []
    actions: list[dict[str, Any]] = []
    validation_ref = "validation_evidence.primary_validation"
    if baseline_completed:
        model = str(baseline.get("model_type") or "recorded baseline configuration")
        refs = ["baseline_evidence.status", validation_ref]
        for field in ("model_type", "metric_value"):
            if baseline.get(field) is not None:
                refs.append(f"baseline_evidence.{field}")
        actions.append(_fallback_action(
            "Run experiment `fallback_exp_reproduce_baseline`: reproduce the recorded EDA baseline.",
            "Reproduction verifies the comparison anchor before any feature or model change.",
            refs,
            _ids_for_categories(category_ids, known_ids, "baseline", "metric", "validation"),
            priority="P0",
            confidence="high",
            experiment_id="fallback_exp_reproduce_baseline",
            hypothesis="The recorded EDA baseline is reproducible under the same validation policy.",
            exact_change=f"Re-run the unchanged `{model}` baseline configuration and recorded preprocessing.",
            validation_policy=method,
            success_criterion=(
                f"All folds complete and the recorded {metric_name} result is reproduced without new leakage warnings."
            ),
            risk="Implementation or environment drift can prevent exact reproduction.",
        ))
    else:
        registry_family = _deterministic_registry_model_family(task_type)
        if registry_family:
            refs = [validation_ref]
            if eda.get("inferred_schema"):
                refs.append("inferred_schema")
            actions.append(_fallback_action(
                "Run experiment `fallback_exp_registry_baseline`: establish the deterministic safe baseline.",
                "No completed baseline exists; the project registry provides a diagnostic floor, not a preferred final model.",
                refs,
                _ids_for_categories(category_ids, known_ids, "baseline", "metric", "validation", "schema"),
                priority="P0",
                confidence="low",
                experiment_id="fallback_exp_registry_baseline",
                hypothesis="A project-registry baseline can establish a fold-comparable diagnostic floor.",
                exact_change=f"Fit `{registry_family}` on EDA-safe non-target, non-primary-ID feature roles.",
                validation_policy=method,
                success_criterion=f"All {method} folds complete and {metric_name} plus fold dispersion are recorded.",
                risk="The registry baseline is a sanity floor and does not establish model-family optimality.",
                limitations=["No completed EDA baseline was available."],
            ))

    metric = _as_dict(eda.get("metric_evidence"))
    if metric.get("requires_threshold") is True:
        actions.append(_fallback_action(
            "Run experiment `fallback_exp_threshold`: compare the default and OOF-fitted decision thresholds.",
            "The metric contract explicitly requires thresholded predictions.",
            ["metric_evidence.requires_threshold", validation_ref],
            _ids_for_categories(category_ids, known_ids, "metric", "validation"),
            priority="P1",
            confidence="high",
            experiment_id="fallback_exp_threshold",
            hypothesis="A threshold fitted only on out-of-fold predictions may improve the threshold-dependent metric.",
            exact_change="Compare the default threshold with a threshold selected exclusively from out-of-fold predictions.",
            validation_policy=method,
            success_criterion=f"OOF {metric_name} improves without using training-fit or test-label outcomes for threshold selection.",
            risk="Threshold overfitting can make fold gains unstable.",
        ))

    if metric.get("requires_calibration") is True:
        actions.append(_fallback_action(
            "Run experiment `fallback_exp_calibration`: compare unchanged OOF probabilities with fold-fitted calibration.",
            "The metric contract explicitly requires calibrated probabilities.",
            ["metric_evidence.requires_calibration", validation_ref],
            _ids_for_categories(category_ids, known_ids, "metric", "validation"),
            priority="P1",
            confidence="high",
            experiment_id="fallback_exp_calibration",
            hypothesis="Fold-fitted probability calibration may improve the calibration-sensitive metric.",
            exact_change=(
                "Fit the calibrator only from training-fold or nested-OOF predictions; "
                "compare it with the unchanged OOF probabilities."
            ),
            validation_policy=method,
            success_criterion=(
                f"OOF {metric_name} improves in the registered direction without "
                "degrading calibration diagnostics or crossing fold boundaries."
            ),
            risk="Calibration can overfit small folds when fitted outside a nested or OOF boundary.",
        ))

    target_diagnostics = _as_dict(eda.get("target_diagnostics"))
    target_distribution = _as_dict(target_diagnostics.get("distribution"))
    target_column = str(
        target_diagnostics.get("target_column")
        or _as_dict(eda.get("inferred_schema")).get("target_column")
        or ""
    ).strip()
    normalized_metric = re.sub(r"[^a-z0-9]+", "", metric_name.casefold())
    if (
        normalized_metric in {"rmsle", "rootmeansquaredlogarithmicerror"}
        and target_column
        and isinstance(target_distribution.get("min"), (int, float))
        and float(target_distribution["min"]) >= 0
    ):
        actions.append(_fallback_action(
            "Run experiment `fallback_exp_target_log1p`: compare direct-target regression with a fold-fitted log1p target arm.",
            "RMSLE semantics and the observed non-negative target support a reversible target-transform experiment.",
            [
                "metric_evidence.metric_name",
                "target_diagnostics.distribution.min",
                "inferred_schema.target_column",
                validation_ref,
            ],
            _ids_for_categories(
                category_ids, known_ids, "metric", "feature", "validation"
            ),
            priority="P1",
            confidence="high",
            experiment_id="fallback_exp_target_log1p",
            hypothesis="A reversible log1p target representation may better align training with RMSLE semantics.",
            exact_change=(
                f"Within each fold, train one arm on `{target_column}` and one on "
                f"log1p(`{target_column}`); invert with expm1 before scoring."
            ),
            validation_policy=method,
            success_criterion=(
                "Mean OOF RMSLE decreases, the worst fold does not materially degrade, "
                "and all inverted predictions remain valid."
            ),
            risk="Inverse transformation bias or invalid negative predictions can erase any training benefit.",
        ))

    if actions and feature_families:
        for family in feature_families[:3]:
            actions.append(_fallback_action(
                f"Run experiment `fallback_exp_feature_{_safe_id(family)}`: ablate `{family}` against the baseline.",
                "EDA marked this concrete feature family as high-potential.",
                [f"feature_probe_evidence.{family}", validation_ref],
                _ids_for_categories(category_ids, known_ids, "feature", "schema", "validation"),
                priority="P1",
                confidence="medium",
                experiment_id=f"fallback_exp_feature_{_safe_id(family)}",
                hypothesis=f"Adding `{family}` improves validation performance relative to the current baseline.",
                exact_change=f"Add only the `{family}` feature family; keep model, folds, preprocessing, and all other features fixed.",
                validation_policy=method,
                success_criterion=f"Mean {metric_name} improves and the direction of change is stable across folds.",
                risk="Feature-family gains may be fold-unstable or introduce leakage if preprocessing is not fold-fitted.",
            ))
    return actions


def _fallback_first_48_hour_blocks(
    *,
    actions_by_section: dict[str, list[dict[str, Any]]],
    experiment_actions: list[dict[str, Any]],
    anchor_action_id: str,
) -> list[dict[str, Any]]:
    def action_ids(*section_ids: str, priority: str | None = None) -> list[str]:
        return _unique_strings(
            action["action_id"]
            for section_id in section_ids
            for action in actions_by_section.get(section_id, [])
            if action.get("action_id") and (priority is None or action.get("priority") == priority)
        )

    experiment_ids = [
        str(action["experiment_id"])
        for action in experiment_actions
        if action.get("experiment_id")
    ]
    blocks = [
        {
            "time_window": "0-4_hours",
            "summary": "Lock metric, primary validation, schema roles, and blocking safety constraints.",
            "action_ids": action_ids(
                "metric_and_validation", "dataset_facts_from_eda",
                "leakage_and_data_quality", "what_not_to_do", priority="P0",
            )[:6],
            "experiment_ids": [],
        },
        {
            "time_window": "4-12_hours",
            "summary": "Establish or reproduce the diagnostic baseline under the locked validation policy.",
            "action_ids": action_ids("baseline_findings")[:4],
            "experiment_ids": experiment_ids[:1],
        },
        {
            "time_window": "12-24_hours",
            "summary": "Run the first controlled metric or feature ablations and retain fold-level results.",
            "action_ids": action_ids("feature_priorities", "drift_and_leaderboard_risk")[:4],
            "experiment_ids": experiment_ids[1:3],
        },
        {
            "time_window": "24-48_hours",
            "summary": "Compare supported model families, complete OOF-only postprocessing, then assemble and validate the submission artifact.",
            "action_ids": action_ids(
                "first_48_hours", "modeling_plan", "experiments_queue"
            )[:8],
            "experiment_ids": experiment_ids[3:] or experiment_ids[-1:],
        },
    ]
    for block in blocks:
        if not block["action_ids"] and not block["experiment_ids"]:
            block["action_ids"] = [anchor_action_id]
    return blocks


def _fallback_valid_evidence_refs(eda: dict[str, Any], refs: Any) -> list[str]:
    valid: list[str] = []
    for ref in _string_values(refs):
        try:
            resolve_evidence_ref(eda, ref)
        except EvidencePathResolutionError:
            continue
        valid.append(ref)
    return _unique_strings(valid)


def _input_provenance_payloads(
    research_hypotheses: list[dict[str, Any]],
    eda_evidence_pack: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_links: list[dict[str, Any]] = []
    eda_links: list[dict[str, Any]] = []
    seen_source: set[tuple[str, str, str]] = set()
    seen_eda: set[tuple[str, str]] = set()
    for raw_hypothesis in research_hypotheses:
        hypothesis = _to_dict(raw_hypothesis)
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            continue
        relationship = (
            "supports"
            if hypothesis.get("status") == "supported_by_source"
            else "motivates"
        )
        confidence = str(hypothesis.get("confidence_before_eda") or "low")
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        for source_ref in _unique_strings(
            _string_values(hypothesis.get("source_refs"))
        ):
            key = (source_ref, hypothesis_id, relationship)
            if key in seen_source:
                continue
            seen_source.add(key)
            source_links.append({
                "source_ref": source_ref,
                "hypothesis_id": hypothesis_id,
                "relationship": relationship,
                "claim_summary": _concise_provenance_summary(
                    hypothesis.get("claim")
                ),
                "confidence": confidence,
            })
    for raw_result in eda_evidence_pack.get("hypothesis_results") or []:
        result = _to_dict(raw_result)
        hypothesis_id = str(result.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            continue
        result_refs = _unique_strings(result.get("evidence_refs"))
        if not result_refs:
            result_refs = [f"hypothesis_results.{hypothesis_id}"]
        status = str(result.get("status") or "not_testable")
        if status not in {
            "confirmed", "partially_confirmed", "rejected", "not_testable", "skipped",
        }:
            status = "not_testable"
        confidence = str(result.get("confidence_after_eda") or "low")
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        for eda_result_ref in result_refs:
            key = (hypothesis_id, eda_result_ref)
            if key in seen_eda:
                continue
            seen_eda.add(key)
            eda_links.append({
                "hypothesis_id": hypothesis_id,
                "eda_result_ref": eda_result_ref,
                "result_status": status,
                "finding_summary": _concise_provenance_summary(
                    result.get("finding")
                ),
                "confidence": confidence,
            })
    return source_links, eda_links


def _build_hypothesis_lookup(
    research_hypotheses: list[dict[str, Any]],
    eda_evidence_pack: dict[str, Any] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    known_ids: list[str] = []
    category_ids: dict[str, list[str]] = {}
    candidates = list(research_hypotheses or [])
    candidates.extend((eda_evidence_pack or {}).get("hypothesis_results") or [])
    candidates.extend((eda_evidence_pack or {}).get("testable_hypotheses") or [])
    for candidate in candidates:
        item = _to_dict(candidate)
        hypothesis_id = str(item.get("hypothesis_id") or item.get("id") or "").strip()
        if not hypothesis_id:
            continue
        category = _canonical_hypothesis_category(
            str(item.get("category") or ""), hypothesis_id
        )
        if hypothesis_id not in known_ids:
            known_ids.append(hypothesis_id)
        if category:
            values = category_ids.setdefault(category, [])
            if hypothesis_id not in values:
                values.append(hypothesis_id)
    return known_ids, category_ids


def _canonical_hypothesis_category(category: str, hypothesis_id: str) -> str:
    normalized = category.strip().lower().replace("-", "_")
    aliases = {
        "val": "validation",
        "validation_strategy": "validation",
        "leak": "leakage",
        "data": "schema",
        "data_quality": "schema",
        "features": "feature",
        "leaderboard": "drift",
    }
    if normalized:
        return aliases.get(normalized, normalized)
    prefix = hypothesis_id.lower().split("_", 1)[0]
    return {
        "schema": "schema",
        "metric": "metric",
        "val": "validation",
        "validation": "validation",
        "leak": "leakage",
        "leakage": "leakage",
        "drift": "drift",
        "feature": "feature",
        "baseline": "baseline",
    }.get(prefix, "")


def _repair_action_payload(
    action: dict[str, Any],
    *,
    known_ids: list[str],
    category_ids: dict[str, list[str]],
    section_context: str,
) -> tuple[dict[str, Any], bool]:
    fixed = dict(action)
    original = deepcopy(fixed)
    priority = str(fixed.get("priority") or "P2").upper()
    fixed["priority"] = priority if priority in {"P0", "P1", "P2", "P3"} else "P2"
    fixed["action"] = str(
        fixed.get("action")
        or "Review this action manually because the LLM response was incomplete."
    ).strip()
    fixed["reason"] = str(
        fixed.get("reason")
        or fixed.get("why")
        or "The final synthesizer repaired an incomplete action conservatively."
    ).strip()
    confidence = str(fixed.get("confidence") or "medium").lower()
    fixed["confidence"] = confidence if confidence in {"low", "medium", "high"} else "low"
    fixed["evidence_refs"] = _string_values(fixed.get("evidence_refs"))
    known_set = set(known_ids)
    fixed["related_hypothesis_ids"] = _known_only(
        [
            *_string_values(fixed.get("related_hypothesis_ids")),
            *_string_values(fixed.get("hypothesis_ids")),
        ],
        known_set,
    )
    if not fixed["related_hypothesis_ids"]:
        fixed["related_hypothesis_ids"] = _infer_hypothesis_ids(
            fixed,
            known_ids=known_ids,
            category_ids=category_ids,
            section_context=section_context,
        )
    fixed["hypothesis_ids"] = list(fixed["related_hypothesis_ids"])
    if not fixed["evidence_refs"]:
        fixed["evidence_refs"] = _infer_evidence_refs(fixed, section_context)
        if fixed["evidence_refs"] == ["final_synthesizer.repaired"]:
            limitations = _string_values(fixed.get("limitations"))
            note = "Evidence linkage was unavailable and requires manual review."
            if note not in limitations:
                limitations.append(note)
            fixed["limitations"] = limitations
            fixed["confidence"] = "low"
    return fixed, fixed != original


def _infer_hypothesis_ids(
    action: dict[str, Any],
    *,
    known_ids: list[str],
    category_ids: dict[str, list[str]],
    section_context: str,
) -> list[str]:
    evidence_text = " ".join(_string_values(action.get("evidence_refs"))).lower()
    action_text = " ".join(
        [
            str(action.get("action") or ""),
            str(action.get("reason") or action.get("why") or ""),
            str(action.get("category") or ""),
            section_context,
        ]
    ).lower()
    categories: list[str] = []

    evidence_rules = (
        ("validation_evidence", ("validation",)),
        ("metric_evidence", ("metric",)),
        ("leakage_evidence", ("leakage",)),
        ("drift_evidence", ("drift",)),
        ("inferred_schema", ("schema",)),
        ("feature_probe_evidence", ("feature",)),
        ("baseline_evidence", ("metric", "validation")),
    )
    for marker, mapped in evidence_rules:
        if marker in evidence_text:
            categories.extend(mapped)

    text_rules = (
        (
            ("stratifiedkfold", "stratified_kfold", "kfold", "split", " cv ", "validation"),
            ("validation",),
        ),
        (
            ("metric", "accuracy", "auc", "rmse", "probability", "probabilities"),
            ("metric",),
        ),
        (("threshold",), ("metric", "validation")),
        (
            ("leakage", "target encoding", "target_in_test", "id_overlap", "unsafe"),
            ("leakage",),
        ),
        (
            ("drift", "adversarial validation", "psi", "leaderboard risk"),
            ("drift",),
        ),
        (
            ("schema", "target column", "primary id", "sample submission"),
            ("schema",),
        ),
        (
            ("feature engineering", "categorical", "numeric", "missingness"),
            ("feature", "schema"),
        ),
        (("baseline",), ("metric", "validation")),
    )
    padded_text = f" {action_text} "
    for markers, mapped in text_rules:
        if any(marker in padded_text for marker in markers):
            categories.extend(mapped)

    inferred = _ids_for_categories(category_ids, [], *categories)
    if inferred:
        return inferred
    if "schema_001" in known_ids:
        return ["schema_001"]
    return known_ids[:1]


def _infer_evidence_refs(
    action: dict[str, Any],
    section_context: str,
) -> list[str]:
    related_ids = _string_values(action.get("related_hypothesis_ids"))
    text = (
        f"{action.get('action', '')} {action.get('reason', '')} {section_context}"
    ).lower()
    refs: list[str] = []
    mapping = (
        ("val_", "validation_evidence.primary_validation"),
        ("metric_", "metric_evidence.metric_name"),
        ("schema_", "inferred_schema"),
        ("leak_", "leakage_evidence"),
        ("drift_", "drift_evidence"),
        ("feature_", "feature_probe_evidence"),
        ("baseline_", "baseline_evidence"),
    )
    for hypothesis_id in related_ids:
        for prefix, ref in mapping:
            if hypothesis_id.lower().startswith(prefix):
                refs.append(ref)
    if "threshold" in text:
        refs.extend(["metric_evidence.requires_threshold", "validation_evidence.primary_validation"])
    if "target encoding" in text or "leak" in text:
        refs.append("leakage_evidence")
    if "drift" in text:
        refs.append("drift_evidence")
    return _unique_strings(refs) or ["final_synthesizer.repaired"]


def _section_evidence_refs(
    section_id: str,
    title: str,
    eda: dict[str, Any],
) -> list[str]:
    text = f"{section_id} {title}".lower()
    candidates: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = [
        (
            ("baseline",),
            (
                ("baseline_evidence", "baseline_evidence"),
                ("metric_evidence", "metric_evidence"),
                ("validation_evidence", "validation_evidence"),
            ),
        ),
        (
            ("validation", "metric"),
            (("validation_evidence", "validation_evidence.primary_validation"),),
        ),
        (("leak", "safety"), (("leakage_evidence", "leakage_evidence"),)),
        (("drift", "leaderboard"), (("drift_evidence", "drift_evidence"),)),
        (
            ("schema", "dataset", "data"),
            (("inferred_schema", "inferred_schema"), ("table_profiles", "table_profiles")),
        ),
        (("feature",), (("feature_probe_evidence", "feature_probe_evidence"),)),
    ]
    refs: list[str] = []
    for markers, evidence_options in candidates:
        if any(marker in text for marker in markers):
            refs.extend(ref for key, ref in evidence_options if eda.get(key))
    return _unique_strings(refs)


def _placeholder_action(known_ids: list[str]) -> dict[str, Any] | None:
    if not known_ids:
        return None
    return _fallback_action(
        "Review this section manually because the LLM produced an incomplete section.",
        "The final synthesizer repaired a malformed section without enough direct evidence.",
        ["final_synthesizer.repaired"],
        [known_ids[0]],
        priority="P2",
        confidence="low",
    )


def _compiled_action_payload(
    action: CompiledAction,
    *,
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> dict[str, Any]:
    return _fallback_action(
        action.action,
        action.reason,
        list(action.evidence_refs),
        _ids_for_categories(category_ids, known_ids, *action.hypothesis_categories),
        priority=action.priority,
        confidence=action.confidence,
        feature_metadata=(
            dict(action.feature_metadata) if action.feature_metadata is not None else None
        ),
    )


def _compiled_experiment_action_payload(
    experiment: CompiledExperiment,
    *,
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> dict[str, Any]:
    return _fallback_action(
        f"Run experiment `{experiment.experiment_id}`: {experiment.name}.",
        "This executable experiment was compiled by a deterministic evidence rule.",
        list(experiment.evidence_refs),
        _ids_for_categories(
            category_ids, known_ids, *experiment.hypothesis_categories
        ),
        priority=experiment.priority,
        confidence=experiment.evidence_strength,
        experiment_id=experiment.experiment_id,
        hypothesis=experiment.hypothesis,
        exact_change=experiment.change,
        validation_policy=experiment.validation_strategy,
        success_criterion=experiment.acceptance_rule,
        risk=" ".join(experiment.risks),
    )


def _ordered_fallback_experiment_actions(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def stage_rank(action: dict[str, Any]) -> tuple[int, int, str]:
        experiment_id = str(action.get("experiment_id") or "")
        if "baseline" in experiment_id:
            stage = 0
        elif any(marker in experiment_id for marker in (
            "feature", "probe", "family_size", "is_alone", "title", "ticket",
            "cabin", "fare", "age", "categorical", "missing", "ablation",
        )):
            stage = 1
        elif "model" in experiment_id:
            stage = 2
        elif any(marker in experiment_id for marker in ("threshold", "calibrat")):
            stage = 3
        elif "submission" in experiment_id:
            stage = 4
        else:
            stage = 2
        priority = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(
            str(action.get("priority") or "P3"), 3
        )
        return stage, priority, experiment_id

    return sorted(actions, key=stage_rank)


def _final_strategy_experiments(
    experiment_actions: list[dict[str, Any]],
    compiled_experiments: tuple[CompiledExperiment, ...],
    *,
    task_type: str,
    metric_name: str,
    method: str,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    compiled_by_id = {
        experiment.experiment_id: experiment for experiment in compiled_experiments
    }
    model_family = str(
        baseline.get("model_type")
        or _deterministic_registry_model_family(task_type)
        or "recorded comparison anchor"
    )
    result: list[dict[str, Any]] = []
    for action in experiment_actions:
        experiment_id = str(action.get("experiment_id") or "")
        if not experiment_id:
            continue
        compiled = compiled_by_id.get(experiment_id)
        if compiled is not None:
            payload = {
                "experiment_id": compiled.experiment_id,
                "priority": compiled.priority,
                "name": compiled.name,
                "hypothesis": compiled.hypothesis,
                "change": compiled.change,
                "feature_inputs": list(compiled.feature_inputs),
                "model_family": compiled.model_family,
                "validation_strategy": compiled.validation_strategy,
                "success_metric": compiled.success_metric,
                "acceptance_rule": compiled.acceptance_rule,
                "evidence_refs": list(compiled.evidence_refs),
                "related_hypothesis_ids": list(
                    action.get("related_hypothesis_ids") or []
                ),
                "risks": list(compiled.risks),
                "fit_scope": compiled.fit_scope,
            }
        else:
            payload = {
                "experiment_id": experiment_id,
                "priority": action["priority"],
                "name": str(action["action"]).removeprefix(
                    f"Run experiment `{experiment_id}`: "
                ).rstrip("."),
                "hypothesis": action["hypothesis"],
                "change": action["exact_change"],
                "feature_inputs": [],
                "model_family": model_family,
                "validation_strategy": method,
                "success_metric": metric_name,
                "acceptance_rule": action["success_criterion"],
                "evidence_refs": list(action.get("evidence_refs") or []),
                "related_hypothesis_ids": list(
                    action.get("related_hypothesis_ids") or []
                ),
                "risks": [str(action["risk"])],
                "fit_scope": (
                    "oof_only"
                    if any(marker in experiment_id for marker in ("threshold", "calibrat"))
                    else "within_fold"
                ),
            }
        result.append(payload)
    return result


def _fallback_action(
    action: str,
    reason: str,
    evidence_refs: list[str],
    related_hypothesis_ids: list[str],
    *,
    priority: Priority = "P1",
    confidence: Confidence = "medium",
    validation_strategy: str | None = None,
    evidence_origin: EvidenceOrigin = "Fallback-generated",
    limitations: list[str] | None = None,
    safety_constraint_ids: list[str] | None = None,
    experiment_id: str | None = None,
    hypothesis: str | None = None,
    exact_change: str | None = None,
    validation_policy: str | None = None,
    success_criterion: str | None = None,
    risk: str | None = None,
    feature_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = re.sub(r"\W+", " ", action.lower()).strip()
    action_id = "fallback_action_" + hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:12]
    payload: dict[str, Any] = {
        "action_id": action_id,
        "priority": priority,
        "action": action,
        "reason": reason,
        "evidence_refs": evidence_refs,
        "related_hypothesis_ids": related_hypothesis_ids,
        "confidence": confidence,
        "evidence_origin": evidence_origin,
        "limitations": limitations or [],
        "safety_constraint_ids": safety_constraint_ids or [],
    }
    if feature_metadata is not None:
        payload["feature_metadata"] = feature_metadata
    if validation_strategy in _valid_validation_methods():
        payload["validation_strategy"] = validation_strategy
    structured_experiment = {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "exact_change": exact_change,
        "validation_policy": validation_policy,
        "success_criterion": success_criterion,
        "risk": risk,
    }
    if any(value is not None for value in structured_experiment.values()):
        payload.update(structured_experiment)
    return payload


def _ids_for_categories(
    category_ids: dict[str, list[str]],
    default_ids: list[str],
    *categories: str,
) -> list[str]:
    selected: list[str] = []
    for category in categories:
        selected.extend(category_ids.get(category, []))
    return _unique_strings(selected) or default_ids[:1]


def _known_only(value: Any, known_ids: set[str]) -> list[str]:
    return [item for item in _string_values(value) if item in known_ids]


def _string_values(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return _unique_strings(str(item).strip() for item in value if str(item).strip())


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _iter_action_payloads(payload: dict[str, Any]) -> Any:
    yield from payload.get("actions") or []
    for section in payload.get("sections") or []:
        yield from section.get("actions") or []


def _contains_unknown_hypothesis_ids(
    result: FinalStrategyResult,
    known_ids: set[str],
) -> bool:
    return bool(_unknown_hypothesis_ids(result, known_ids))


def _unknown_hypothesis_ids(
    result: FinalStrategyResult,
    known_ids: set[str],
) -> list[str]:
    return sorted({
        hypothesis_id
        for action in _all_actions(result)
        for hypothesis_id in action.related_hypothesis_ids
        if hypothesis_id not in known_ids
    })


def _unknown_hypothesis_reference_issues(
    result: FinalStrategyResult,
    known_ids: set[str],
) -> list[ValidationIssue]:
    issues = [
        ValidationIssue(
            stage="llm_reference_validation",
            issue_type="unknown_hypothesis_id",
            field_path=f"actions[{index}].related_hypothesis_ids",
            message=f"Unknown hypothesis reference: {hypothesis_id}",
            invalid_value_type="str",
            invalid_reference=_bounded_message(hypothesis_id, limit=200),
            expected_contract="known hypothesis ID",
        )
        for index, action in enumerate(_all_actions(result))
        for hypothesis_id in action.related_hypothesis_ids
        if hypothesis_id not in known_ids
    ]
    return _sorted_validation_issues(issues)


def _has_unsafe_target_encoding(eda: dict[str, Any]) -> bool:
    evidence = {
        "feature_probe_evidence": eda.get("feature_probe_evidence") or [],
        "leakage_evidence": eda.get("leakage_evidence") or [],
    }
    text = json.dumps(evidence, ensure_ascii=False).lower()
    return "target encoding" in text and any(
        marker in text for marker in ("unsafe", "high", "failed", "warning", "leak")
    )


def _has_high_potential_features(eda: dict[str, Any]) -> bool:
    for raw_probe in eda.get("feature_probe_evidence") or []:
        probe = _to_dict(raw_probe)
        if probe.get("potential") == "high" or probe.get("status") == "high_potential":
            return True
        if probe.get("high_potential") is True:
            return True
    return False


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return {}


def postprocess_final_strategy_result(
    result: FinalStrategyResult,
    *,
    eda_evidence_pack: dict[str, Any] | None,
    source_evidence: list[dict[str, Any]] | None = None,
) -> FinalStrategyResult:
    cleaned = result.model_copy(deep=True)
    eda = eda_evidence_pack or {}
    source_ids = {
        str(item.get("id") or item.get("source_ref") or "").strip()
        for item in source_evidence or []
        if item.get("id") or item.get("source_ref")
    }
    known_ids, category_ids = _build_hypothesis_lookup(
        [],
        eda,
    )
    for action in _all_actions(cleaned):
        for hypothesis_id in action.related_hypothesis_ids:
            if hypothesis_id not in known_ids:
                known_ids.append(hypothesis_id)

    _ensure_primary_id_safety_action(cleaned, eda, known_ids, category_ids)
    _ensure_conservative_baseline(cleaned, eda, known_ids, category_ids)
    _ensure_leakage_safety_action(cleaned, eda, known_ids, category_ids)

    primary_id = _primary_id_from_eda(eda)
    validation_method = _primary_validation_method_from_dict(eda)
    section_by_action_id = {
        action_id: section.section_id
        for section in cleaned.sections
        for action_id in section.action_ids
    }
    for action in cleaned.actions:
        _normalize_strategy_action(
            action,
            section_id=section_by_action_id.get(action.action_id or "", ""),
            eda=eda,
            source_ids=source_ids,
            primary_id=primary_id,
            validation_method=validation_method,
        )
        _repair_precise_action_evidence(
            action,
            eda=eda,
            source_ids=source_ids,
            reference_repairs=cleaned.reference_repairs,
        )

    _deduplicate_strategy_actions(cleaned, primary_id=primary_id)
    for action in cleaned.actions:
        _finalize_action_evidence_contract(
            action,
            eda=eda,
            source_ids=source_ids,
        )
    _finalize_structured_experiment_evidence(cleaned, eda)
    _refresh_action_and_section_provenance(cleaned)
    _append_evidence_availability_limitations(cleaned, eda)
    finalized = FinalStrategyResult.model_validate(cleaned.model_dump(mode="json"))
    validate_semantic_action_postconditions(
        finalized.model_dump(mode="json"),
        primary_id=primary_id,
    )
    return finalized


def _finalize_structured_experiment_evidence(
    result: FinalStrategyResult,
    eda: dict[str, Any],
) -> None:
    """Keep structured experiments aligned with their canonical action evidence."""

    action_by_experiment = {
        action.experiment_id: action
        for action in result.actions
        if action.experiment_id
    }
    for experiment in result.experiments:
        action = action_by_experiment.get(experiment.experiment_id)
        if action is not None:
            object.__setattr__(
                experiment, "evidence_refs", list(action.evidence_refs)
            )
            object.__setattr__(
                experiment,
                "related_hypothesis_ids",
                list(action.related_hypothesis_ids),
            )
        for ref in experiment.evidence_refs:
            try:
                resolve_evidence_ref(eda, ref)
            except EvidencePathResolutionError as exc:
                raise FinalStrategyRepairError(
                    f"Structured experiment {experiment.experiment_id!r} has an "
                    f"unresolvable evidence reference: {ref!r}",
                    phase="experiment_evidence_validation",
                ) from exc


def _repair_precise_action_evidence(
    action: FinalStrategyAction,
    *,
    eda: dict[str, Any],
    source_ids: set[str],
    reference_repairs: list[dict[str, str]],
) -> None:
    """Apply only unambiguous path repairs and safe factual downgrades."""

    non_eda_refs = _non_eda_action_refs(action, eda, source_ids)
    issues = validate_action_evidence_consistency(
        action, eda, allowed_non_eda_refs=non_eda_refs
    )
    if not issues:
        return
    refs = list(action.evidence_refs)
    codes = {issue.code for issue in issues}

    if "drift_value_contradiction" in codes:
        action.action = "Inspect the available drift diagnostics before modeling."
        action.reason = "The resolved drift severity does not support a high-drift claim."
        action.confidence = "low"
        action.limitations = _unique_strings([
            *action.limitations,
            "The original high-drift wording was downgraded to match the resolved severity.",
        ])
        refs = [ref for ref in refs if not ref.startswith("drift_evidence.")]
        refs.append("drift_evidence")
    if "threshold_value_contradiction" in codes:
        action.action = "Do not tune a decision threshold unless the metric contract changes."
        action.reason = "The resolved metric evidence does not require threshold tuning."
        action.confidence = "high"
    if "validation_value_contradiction" in codes:
        method = _primary_validation_method_from_dict(eda)
        if method:
            action.action = f"Use {method} as the primary validation policy selected by EDA."
            action.reason = "The action was aligned to validation_evidence.primary_validation."
            action.validation_strategy = method if method in _valid_validation_methods() else None
    if "baseline_value_contradiction" in codes:
        action.action = "Complete a fold-safe baseline before attempting to reproduce or extend it."
        action.reason = "The baseline evidence does not report a successfully completed baseline."
        action.confidence = "high"

    # Re-evaluate after wording downgrades. Precise refs are then safe to add.
    temporary_eda_refs = _resolvable_eda_refs(refs, eda)
    object.__setattr__(action, "evidence_refs", _unique_strings(refs))
    object.__setattr__(action, "eda_result_refs", temporary_eda_refs)
    remaining = validate_action_evidence_consistency(
        action, eda, allowed_non_eda_refs=non_eda_refs
    )
    for issue in remaining:
        if issue.code == "broken_evidence_ref" and issue.ref:
            if issue.ref.split(".", 1)[0] not in eda:
                # Cross-namespace/context labels remain visible to the existing
                # reference-repair orchestration instead of being mistaken for
                # malformed EDA paths.
                continue
            replacement = "final_synthesizer.repaired"
            refs = [ref for ref in refs if ref != issue.ref]
            if not refs:
                refs.append(replacement)
            _record_evidence_repair(reference_repairs, action, issue.ref, replacement)
            continue
        if issue.code.endswith("requires_precise_ref") and issue.ref:
            root = issue.ref.split(".", 1)[0]
            broad = next((ref for ref in refs if ref == root), None)
            refs = [ref for ref in refs if ref != broad]
            if issue.ref not in refs:
                refs.append(issue.ref)
            _record_evidence_repair(
                reference_repairs,
                action,
                broad or "<missing precise evidence ref>",
                issue.ref,
            )

    refs = _unique_strings(refs or ["final_synthesizer.repaired"])
    object.__setattr__(action, "evidence_refs", refs)
    object.__setattr__(action, "eda_result_refs", _resolvable_eda_refs(refs, eda))


def _finalize_action_evidence_contract(
    action: FinalStrategyAction,
    *,
    eda: dict[str, Any],
    source_ids: set[str],
) -> None:
    refs = _unique_strings(action.evidence_refs)
    object.__setattr__(action, "evidence_refs", refs)
    object.__setattr__(action, "eda_result_refs", _resolvable_eda_refs(refs, eda))
    issues = validate_action_evidence_consistency(
        action,
        eda,
        allowed_non_eda_refs=_non_eda_action_refs(action, eda, source_ids),
    )
    if issues:
        detail = "; ".join(issue.message for issue in issues[:6])
        raise FinalStrategyRepairError(
            f"Final strategy evidence consistency failed: {detail}",
            phase="evidence_consistency_validation",
        )
    object.__setattr__(
        action,
        "evidence_bindings",
        build_action_evidence_bindings(
            action,
            eda,
            allowed_non_eda_refs=_non_eda_action_refs(action, eda, source_ids),
        ),
    )


def _non_eda_action_refs(
    action: FinalStrategyAction,
    eda: dict[str, Any],
    source_ids: set[str],
) -> set[str]:
    return {
        *source_ids,
        *(
            ref for ref in action.evidence_refs
            if ref.split(".", 1)[0].split("[", 1)[0] not in eda
        ),
    }


def _resolvable_eda_refs(refs: list[str], eda: dict[str, Any]) -> list[str]:
    resolved: list[str] = []
    for ref in refs:
        try:
            resolve_evidence_ref(eda, ref)
        except EvidencePathResolutionError:
            continue
        resolved.append(ref)
    return _unique_strings(resolved)


def _record_evidence_repair(
    repairs: list[dict[str, str]],
    action: FinalStrategyAction,
    original: str,
    replacement: str,
) -> None:
    if original == replacement:
        return
    repair = {
        "field_path": f"actions[{action.action_id or 'unknown'}].evidence_specificity",
        "original_id": original,
        "replacement_id": replacement,
    }
    if repair not in repairs:
        repairs.append(repair)


def render_final_strategy(result: FinalStrategyResult) -> str:
    if result.schema_version == "2.0" and result.evidence_catalog:
        return _render_compacted_final_strategy(result)
    lines = ["# Final Strategy", ""]
    if result.synthesis_status == "repaired_success":
        lines.extend([
            "The model-generated strategy required deterministic contract repair.",
            "",
        ])
    elif result.synthesis_status == "degraded_fallback":
        lines.extend([
            "## Synthesis Status",
            "",
            "- Status: degraded fallback",
            "- The LLM-generated strategy did not satisfy the final strategy contract.",
            "- The report below was assembled deterministically from validated EDA and Scout evidence.",
            "",
        ])
    lines.extend([
        f"Competition: `{result.competition_id}`",
        f"Task type: `{result.task_type or 'unknown'}`",
        f"Metric: `{_metric_name(result.metric)}`",
        f"Validation: `{result.recommended_validation or 'unknown'}`",
        "",
    ])
    action_map = {
        action.action_id: action
        for action in result.actions
        if action.action_id
    }
    for section in result.sections:
        lines.extend([f"## {section.title}", "", section.summary, ""])
        if section.availability != "available":
            lines.append(f"- Availability: `{section.availability}`")
        for limitation in section.limitations:
            lines.append(f"- Limitation: {limitation}")
        if (section.availability != "available" or section.limitations) and (
            section.action_ids or section.time_blocks
        ):
            lines.append("")
        for action_id in dict.fromkeys(section.action_ids):
            action = action_map[action_id]
            evidence = ", ".join(action.evidence_refs)
            lines.append(
                f"- {action.priority} [{action.evidence_origin}]: "
                f"{action.action} Evidence: {evidence}"
            )
            if action.reason:
                lines.append(f"  Rationale: {action.reason}")
            if action.feature_metadata:
                metadata = action.feature_metadata
                lines.extend([
                    "  Feature inputs: "
                    + ", ".join(f"`{column}`" for column in metadata.input_columns),
                    f"  Deterministic transform: {metadata.deterministic_transform}",
                    f"  Fit scope: `{metadata.fit_scope}`",
                    f"  Leakage risk: `{metadata.leakage_risk}`",
                    f"  Expected diagnostic: {metadata.expected_diagnostic}",
                ])
            if action.risk_ids:
                lines.append(f"  Risks: {', '.join(action.risk_ids)}")
            if action.validation_requirement_ids:
                lines.append(
                    "  Validation requirements: "
                    + ", ".join(action.validation_requirement_ids)
                )
            if action.safety_constraint_ids:
                lines.append(
                    "  Safety constraints: " + ", ".join(action.safety_constraint_ids)
                )
            if action.experiment_id:
                lines.extend([
                    f"  Experiment ID: `{action.experiment_id}`",
                    f"  Hypothesis: {action.hypothesis}",
                    f"  Exact change: {action.exact_change}",
                    f"  Validation policy: `{action.validation_policy}`",
                    f"  Success criterion: {action.success_criterion}",
                    f"  Risk: {action.risk}",
                ])
        for block in section.time_blocks:
            label = block.time_window.replace("_hours", " hours").replace("-", "–")
            lines.extend([f"### {label}", "", block.summary, ""])
            if block.action_ids:
                lines.append("- Action references: " + ", ".join(
                    f"`{action_id}`" for action_id in block.action_ids
                ))
            if block.experiment_ids:
                lines.append("- Experiment references: " + ", ".join(
                    f"`{experiment_id}`" for experiment_id in block.experiment_ids
                ))
            lines.append("")
        if section.evidence_refs and not section.action_ids:
            lines.append(f"- Evidence: {', '.join(section.evidence_refs)}")
        lines.append("")
    if result.actions and not result.sections:
        lines.extend(["## Prioritized Actions", ""])
        for action in result.actions:
            lines.append(
                f"- {action.priority} [{action.evidence_origin}]: {action.action} "
                f"Evidence: {', '.join(action.evidence_refs)}"
            )
            if action.risk_ids:
                lines.append(f"  Risks: {', '.join(action.risk_ids)}")
            if action.validation_requirement_ids:
                lines.append("  Validation requirements: " + ", ".join(action.validation_requirement_ids))
            if action.safety_constraint_ids:
                lines.append("  Safety constraints: " + ", ".join(action.safety_constraint_ids))
            if action.experiment_id:
                lines.append(f"  Experiment ID: `{action.experiment_id}`")
        lines.append("")
    if any((
        result.acknowledged_risk_ids,
        result.selected_validation_requirement_ids,
        result.enforced_safety_constraint_ids,
    )):
        lines.extend(["## Enforced Contract References", ""])
        if result.acknowledged_risk_ids:
            lines.append("- Acknowledged risks: " + ", ".join(result.acknowledged_risk_ids))
        if result.selected_validation_requirement_ids:
            lines.append("- Selected validation requirements: " + ", ".join(result.selected_validation_requirement_ids))
        if result.enforced_safety_constraint_ids:
            lines.append("- Enforced safety constraints: " + ", ".join(result.enforced_safety_constraint_ids))
        lines.append("")
    if result.limitations:
        lines.extend(["## Evidence Availability & Uncertainty", ""])
        lines.extend(f"- {item}" for item in result.limitations)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_compacted_final_strategy(result: FinalStrategyResult) -> str:
    action_map = {
        action.action_id: action for action in result.actions if action.action_id
    }
    sections = {section.section_id: section for section in result.sections}
    lines = ["# Final Strategy", "", "## Synthesis Status", ""]
    if result.synthesis_status == "degraded_fallback":
        lines.extend([
            "**Degraded fallback:** the LLM draft did not satisfy the canonical contract; "
            "this compact strategy was compiled deterministically from validated evidence.",
            "",
        ])
    elif result.synthesis_status == "repaired_success":
        lines.extend(["The LLM draft passed after deterministic contract repair.", ""])
    else:
        lines.extend(["The LLM draft passed the canonical contract.", ""])

    executive = sections.get("executive_summary")
    lines.extend(["## Executive Summary", "", executive.summary if executive else "Not available.", ""])
    ordered = [
        ("Metric and Validation", "metric_and_validation"),
        ("Dataset Facts From EDA", "dataset_facts_from_eda"),
        ("Leakage and Data Quality", "leakage_and_data_quality"),
        ("Drift and Leaderboard Risk", "drift_and_leaderboard_risk"),
        ("Baseline Findings", "baseline_findings"),
        ("Feature Priorities", "feature_priorities"),
        ("Modeling Plan", "modeling_plan"),
    ]
    for title, section_id in ordered:
        section = sections.get(section_id)
        lines.extend([f"## {title}", ""])
        if section:
            lines.extend([section.summary, ""])
            for action_id in section.action_ids:
                action = action_map.get(action_id)
                if action:
                    _append_compact_action(lines, action)
        else:
            lines.extend(["Not available.", ""])

    lines.extend(["## Feature Experiment Families", ""])
    for family in result.feature_experiment_families:
        lines.extend([
            f"### {family.name}", "",
            f"- Inputs: {', '.join(f'`{column}`' for column in family.input_columns)}",
            f"- Baseline arm: {family.baseline_arm.name}",
            "- Candidate arms: " + "; ".join(
                f"{arm.name} — {arm.exact_change}" for arm in family.candidate_arms
            ),
            f"- Acceptance: {family.acceptance_rule}",
            f"- Risks: {'; '.join(family.risks)}",
            "",
        ])

    _append_compact_experiment_section(lines, "Core Experiments", result.core_experiments)
    _append_compact_experiment_section(lines, "Experiment Backlog", result.experiment_backlog)

    what_not = sections.get("what_not_to_do")
    lines.extend(["## What Not To Do", ""])
    if what_not:
        lines.extend([what_not.summary, ""])
        for action_id in what_not.action_ids:
            action = action_map.get(action_id)
            if action:
                _append_compact_action(lines, action)

    first_48 = sections.get("first_48_hours")
    lines.extend(["## First 48 Hours", ""])
    if first_48:
        for block in first_48.time_blocks:
            label = block.time_window.replace("_hours", " hours").replace("-", "–")
            refs = ", ".join(f"`{item}`" for item in block.experiment_ids) or "contract actions only"
            lines.extend([f"### {label}", "", block.summary, "", f"- Experiments: {refs}", ""])

    lines.extend(["## Enforced Contract References", ""])
    lines.append(
        "- Validation requirements: "
        + (", ".join(result.selected_validation_requirement_ids) or "none registered")
    )
    lines.append(
        "- Safety constraints: "
        + (", ".join(result.enforced_safety_constraint_ids) or "none registered")
    )
    lines.extend(["", "## Evidence Availability and Uncertainty", ""])
    lines.extend(f"- {item}" for item in result.limitations)
    lines.extend([
        f"- Evidence catalog: {len(result.evidence_catalog)} unique references; "
        f"approximately {result.quality_metrics.duplicate_preview_bytes_avoided} duplicate preview bytes avoided.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _append_compact_action(lines: list[str], action: FinalStrategyAction) -> None:
    primary = action.primary_evidence_refs[:1] or action.evidence_refs[:1]
    supporting = [ref for ref in action.evidence_refs if ref not in primary][:2]
    refs = [*primary, *supporting]
    lines.append(f"- {action.priority}: {action.action}")
    if refs:
        lines.append("  Evidence: " + ", ".join(f"`{ref}`" for ref in refs))
    additional = max(0, len(action.evidence_refs) - len(refs))
    if additional:
        lines.append(f"  Additional evidence: {additional} references.")


def _append_compact_experiment_section(
    lines: list[str], title: str, experiments: list[Any]
) -> None:
    lines.extend([f"## {title}", ""])
    if not experiments:
        lines.extend(["No experiments in this tier.", ""])
        return
    for index, experiment in enumerate(experiments, start=1):
        primary = experiment.primary_evidence_refs[:1] or experiment.evidence_refs[:1]
        lines.extend([
            f"{index}. **{experiment.name}** (`{experiment.experiment_id}`; {experiment.estimated_cost} cost)",
            f"   - Change: {experiment.change}",
            f"   - Acceptance: {experiment.acceptance_rule}",
            "   - Evidence: " + ", ".join(f"`{ref}`" for ref in primary),
        ])
        if experiment.dependencies:
            lines.append("   - Dependencies: " + ", ".join(experiment.dependencies))
    lines.append("")


def render_final_strategy_summary(result: FinalStrategyResult) -> str:
    actions = _unique_model_actions(_all_actions(result))
    p0_actions = [action for action in actions if action.priority == "P0"][:5]
    risks = [
        action
        for action in actions
        if action.evidence_origin == "Safety-warning"
        or _primary_evidence_category(action.evidence_refs) in {"drift", "leakage"}
    ][:3]
    experiments = [
        action
        for action in actions
        if action.priority in {"P1", "P2"}
        and action.evidence_origin in {
            "EDA-inferred",
            "Source-supported",
            "Hypothesis-to-test",
        }
    ][:3]
    do_not = [
        action
        for action in actions
        if action.action.lower().startswith(("do not", "avoid", "exclude", "never"))
    ][:3]
    lines = [
        "# Final Strategy Summary",
        "",
        f"- Competition: `{result.competition_id}`",
        f"- Task: `{result.task_type or 'unknown'}`",
        f"- Metric: `{_metric_name(result.metric)}`",
        f"- Validation: `{result.recommended_validation or 'unknown'}`",
        f"- EDA evidence: `{_evidence_availability_status(actions)}`",
    ]
    _append_summary_actions(lines, "Top P0 Actions", p0_actions)
    _append_summary_actions(lines, "Top Risks", risks)
    _append_summary_actions(lines, "First Experiments", experiments)
    _append_summary_actions(lines, "Do Not Do", do_not)
    if result.limitations:
        lines.extend(["", "## Key Uncertainty", f"- {result.limitations[0]}"])
    return "\n".join(lines).rstrip() + "\n"


def validate_rendered_strategy_quality(
    result: FinalStrategyResult,
    full_text: str,
    summary_text: str,
    *,
    eda_evidence_pack: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    eda = eda_evidence_pack or {}
    action_map = {
        action.action_id: action for action in result.actions if action.action_id
    }
    rendered_actions = [
        action_map[action_id]
        for section in result.sections
        for action_id in section.action_ids
    ]
    actions = _unique_model_actions(_all_actions(result))
    seen: list[FinalStrategyAction] = []
    for action in rendered_actions:
        if any(_actions_semantically_similar(action, prior) for prior in seen):
            warnings.append("Duplicate strategy actions remain after postprocessing.")
            break
        seen.append(action)

    primary_id = _primary_id_from_eda(eda)
    method = _primary_validation_method_from_dict(eda)
    for action in actions:
        for issue in validate_action_evidence_consistency(
            action,
            eda,
            allowed_non_eda_refs=action.source_refs,
        ):
            warnings.append(
                f"Action {action.action_id or '<unknown>'} evidence inconsistency "
                f"[{issue.code}]: {issue.message}"
            )
        text = f"{action.action} {action.reason}".lower()
        if _unsafe_primary_id_feature_text(text, primary_id):
            warnings.append("Primary ID is recommended as a predictive feature.")
            break
        if _unsafe_primary_id_split_text(text, primary_id) and method not in {
            "group_kfold",
            "stratified_group_kfold",
            "ranking_group_cv",
        }:
            warnings.append("Primary ID is recommended as a split key without group validation.")
            break
    if "perfectly separate" in full_text.lower():
        warnings.append("Drift language overstates diagnostic separability.")
    if full_text and len(summary_text) >= 0.5 * len(full_text):
        warnings.append("Final strategy summary is too close in length to the full strategy.")
    if any(
        action.source_refs
        and not _has_eda_evidence(action.evidence_refs)
        and action.evidence_origin not in {"Source-supported", "Hypothesis-to-test"}
        for action in actions
    ):
        warnings.append("A source-only recommendation lacks a source or hypothesis label.")
    if _primary_validation_method_from_dict(eda) and not any(
        action.priority == "P0"
        and "validation_evidence" in " ".join(action.evidence_refs)
        for action in actions
    ):
        warnings.append("No P0 action preserves the EDA-selected validation policy.")
    if eda.get("leakage_evidence") and not any(
        action.evidence_origin == "Safety-warning"
        and (
            "leak" in action.action.lower()
            or "encoding" in action.action.lower()
            or "identifier" in action.action.lower()
            or "primary id" in action.action.lower()
        )
        for action in actions
    ):
        warnings.append("Leakage evidence exists without a corresponding safety action.")
    return warnings


def _normalize_strategy_action(
    action: FinalStrategyAction,
    *,
    section_id: str,
    eda: dict[str, Any],
    source_ids: set[str],
    primary_id: str | None,
    validation_method: str | None,
) -> None:
    priority_match = re.match(r"^\s*(P[0-3])\s*[:\-]\s*", action.action, re.IGNORECASE)
    if priority_match:
        embedded_priority = priority_match.group(1).upper()
        if _priority_rank(embedded_priority) < _priority_rank(action.priority):
            action.priority = embedded_priority
        action.action = action.action[priority_match.end():].strip()
    action.evidence_refs = _unique_strings(action.evidence_refs)
    action.eda_result_refs = _unique_strings(action.eda_result_refs)
    action.source_refs = _unique_strings(action.source_refs)
    action.risk_ids = _unique_strings(action.risk_ids)
    action.validation_requirement_ids = _unique_strings(action.validation_requirement_ids)
    action.safety_constraint_ids = _unique_strings(action.safety_constraint_ids)
    action.related_hypothesis_ids = _unique_strings(action.related_hypothesis_ids)
    action.hypothesis_ids = _unique_strings(
        [*action.related_hypothesis_ids, *action.hypothesis_ids]
    )
    action.related_hypothesis_ids = list(action.hypothesis_ids)
    text = f"{action.action} {action.reason}".lower()

    if _unsafe_primary_id_feature_text(text, primary_id):
        action.action = (
            "Keep the primary ID excluded from model features and use it only for "
            "submission alignment or row tracking."
        )
        action.reason = "EDA assigned this column the primary identifier role."
        action.evidence_refs = _unique_strings(
            [*action.evidence_refs, "inferred_schema.primary_id_column"]
        )
        action.priority = "P0"
    elif _unsafe_primary_id_split_text(text, primary_id):
        if validation_method in {
            "group_kfold",
            "stratified_group_kfold",
            "ranking_group_cv",
        }:
            action.action = (
                "Use the EDA-selected group column for group-aware validation; keep "
                "the primary ID excluded from features and use it only for alignment."
            )
        else:
            action.action = (
                "Split according to the EDA-selected validation policy; keep the "
                "primary ID excluded from features and use it only for alignment or tracking."
            )
        action.reason = "A row identifier is not a valid split key unless EDA explicitly selects it as a group."
        action.evidence_refs = _unique_strings(
            [*action.evidence_refs, "inferred_schema.primary_id_column", "validation_evidence.primary_validation"]
        )
        action.priority = "P0"

    if "perfectly separate" in action.action.lower() or "perfectly separate" in action.reason.lower():
        action.action = re.sub(
            r"(?i)(?:the model |adversarial validation )?can perfectly separate(?:s)? train from test",
            "Adversarial validation indicates train/test are highly separable under the diagnostic model",
            action.action,
        )
        action.reason = re.sub(
            r"(?i)(?:the model |adversarial validation )?can perfectly separate(?:s)? train from test",
            "Adversarial validation indicates train/test are highly separable under the diagnostic model",
            action.reason,
        )
    if _drift_is_id_artifact(text, primary_id, eda):
        action.action = (
            "Treat primary-ID or index drift as a diagnostic artifact; assess feature "
            "drift separately and do not model from identifier drift."
        )
        action.reason = "Identifier/index distribution changes do not establish predictive feature drift."
        action.evidence_refs = _unique_strings(
            [*action.evidence_refs, "drift_evidence", "inferred_schema.primary_id_column"]
        )

    action.evidence_origin = _derive_evidence_origin(
        action,
        section_id=section_id,
        source_ids=source_ids,
    )
    if action.evidence_origin == "Source-supported" and not _is_test_wording(action.action):
        suggestion = re.sub(
            r"(?i)^(?:use|add|include|apply|build|train)\s+",
            "",
            action.action,
        )
        action.action = f"Test source-suggested {suggestion} and keep it only if validation improves."
        action.priority = "P1" if action.priority == "P0" else action.priority
        action.confidence = "medium" if action.confidence == "high" else action.confidence
    if (
        action.priority == "P0"
        and _looks_advanced_model(action.action)
        and "baseline_evidence" not in " ".join(action.evidence_refs)
        and "simple" not in action.action.lower()
    ):
        action.priority = "P1"


def _deduplicate_strategy_actions(
    result: FinalStrategyResult,
    *,
    primary_id: str | None,
) -> None:
    canonical, _ = canonicalize_semantic_strategy_actions(
        result.model_dump(mode="json"),
        primary_id=primary_id,
    )
    canonical_actions = [
        FinalStrategyAction.model_validate(action)
        for action in canonical["actions"]
    ]
    canonical_sections = [
        FinalStrategySection.model_validate(section)
        for section in canonical["sections"]
    ]
    object.__setattr__(result, "actions", canonical_actions)
    object.__setattr__(result, "sections", canonical_sections)
    object.__setattr__(
        result,
        "reference_repairs",
        list(canonical.get("reference_repairs") or []),
    )


def _refresh_action_and_section_provenance(result: FinalStrategyResult) -> None:
    action_map = {
        action.action_id: action for action in result.actions if action.action_id
    }
    object.__setattr__(result, "action_provenance", [
        ActionProvenance(
            action_id=action_id,
            source_refs=list(action.source_refs),
            hypothesis_ids=list(action.hypothesis_ids),
            motivating_hypothesis_ids=list(action.motivating_hypothesis_ids),
            safety_hypothesis_ids=list(action.safety_hypothesis_ids),
            validation_context_ids=list(action.validation_context_ids),
            eda_result_refs=list(action.eda_result_refs),
        )
        for action_id, action in action_map.items()
    ])
    for section in result.sections:
        section_actions = [
            action_map[action_id]
            for action_id in section.action_ids
            if action_id in action_map
        ]
        if not section_actions:
            continue
        section.evidence_refs = _unique_strings(
            ref for action in section_actions for ref in action.evidence_refs
        )
        section.related_hypothesis_ids = _unique_strings(
            ref for action in section_actions for ref in action.hypothesis_ids
        )
        section.source_refs = _unique_strings(
            ref for action in section_actions for ref in action.source_refs
        )
        section.eda_result_refs = _unique_strings(
            ref for action in section_actions for ref in action.eda_result_refs
        )


def _merge_action_candidate(
    groups: list[dict[str, Any]],
    action: FinalStrategyAction,
    section_index: int | None,
) -> None:
    for group in groups:
        existing = group["action"]
        if not _actions_semantically_similar(action, existing):
            continue
        existing.evidence_refs = _unique_strings([*existing.evidence_refs, *action.evidence_refs])
        existing.eda_result_refs = _unique_strings([*existing.eda_result_refs, *action.eda_result_refs])
        existing.source_refs = _unique_strings([*existing.source_refs, *action.source_refs])
        if not existing.source_claim and action.source_claim:
            existing.source_claim = action.source_claim
        existing.related_hypothesis_ids = _unique_strings(
            [*existing.related_hypothesis_ids, *action.related_hypothesis_ids]
        )
        existing.hypothesis_ids = _unique_strings(
            [*existing.hypothesis_ids, *action.hypothesis_ids]
        )
        existing.experiment_ids = _unique_strings(
            [*existing.experiment_ids, *action.experiment_ids]
        )
        existing.risk_ids = _unique_strings([*existing.risk_ids, *action.risk_ids])
        existing.validation_requirement_ids = _unique_strings([
            *existing.validation_requirement_ids,
            *action.validation_requirement_ids,
        ])
        existing.safety_constraint_ids = _unique_strings([
            *existing.safety_constraint_ids,
            *action.safety_constraint_ids,
        ])
        if _priority_rank(action.priority) < _priority_rank(existing.priority):
            existing.priority = action.priority
        if len(action.reason) > len(existing.reason):
            existing.reason = action.reason
        if len(action.action) > len(existing.action):
            existing.action = action.action
        existing.evidence_origin = _stronger_origin(
            existing.evidence_origin, action.evidence_origin
        )
        group["locations"].append(section_index)
        return
    groups.append({"action": action.model_copy(deep=True), "locations": [section_index]})


def _actions_semantically_similar(
    left: FinalStrategyAction,
    right: FinalStrategyAction,
) -> bool:
    if _primary_evidence_category(left.evidence_refs) != _primary_evidence_category(
        right.evidence_refs
    ):
        return False
    left_text = _semantic_action_text(left.action)
    right_text = _semantic_action_text(right.action)
    if left_text == right_text or left_text in right_text or right_text in left_text:
        return True
    left_tokens = set(left_text.split())
    right_tokens = set(right_text.split())
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.62


def _semantic_action_text(value: str) -> str:
    text = value.lower().replace("`", " ")
    replacements = {
        "cross validation": "cv",
        "cross-validation": "cv",
        "stratified k-fold": "stratified kfold",
        "stratifiedkfold": "stratified kfold",
        "mean encoding": "target encoding",
        "public leaderboard": "public lb",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b(?:p0|p1|p2|p3)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _primary_evidence_category(evidence_refs: list[str]) -> str:
    text = " ".join(evidence_refs).lower()
    for marker, category in (
        ("validation_evidence", "validation"),
        ("metric_evidence", "metric"),
        ("leakage_evidence", "leakage"),
        ("feature_probe_evidence", "feature"),
        ("drift_evidence", "drift"),
        ("inferred_schema", "schema"),
        ("table_profiles", "schema"),
        ("baseline_evidence", "baseline"),
        ("final_synthesizer.repaired", "fallback"),
    ):
        if marker in text:
            return category
    return "source"


def _derive_evidence_origin(
    action: FinalStrategyAction,
    *,
    section_id: str,
    source_ids: set[str],
) -> EvidenceOrigin:
    text = f"{action.action} {action.reason} {section_id}".lower()
    refs = " ".join(action.evidence_refs).lower()
    if "final_synthesizer.repaired" in refs:
        return "Fallback-generated"
    if (
        "leak" in text
        or section_id in {"leakage_and_data_quality", "what_not_to_do"}
        or action.action.lower().startswith(("do not", "avoid", "exclude", "never"))
    ):
        return "Safety-warning"
    if _has_eda_evidence(action.evidence_refs):
        if (
            "feature_probe_evidence" in refs
            or "eda_strategy_hints" in refs
            or _is_test_wording(action.action)
        ):
            return "EDA-inferred"
        return "EDA-confirmed"
    if action.source_refs or any(ref in source_ids for ref in action.evidence_refs):
        return "Source-supported"
    if any(
        marker in refs
        for marker in ("kaggle", "github", "arxiv", "paper", "huggingface", "pwc")
    ):
        return "Source-supported"
    return "Hypothesis-to-test"


def _has_eda_evidence(evidence_refs: list[str]) -> bool:
    return any(
        ref.startswith(
            (
                "validation_evidence",
                "metric_evidence",
                "leakage_evidence",
                "drift_evidence",
                "inferred_schema",
                "table_profiles",
                "feature_probe_evidence",
                "baseline_evidence",
            )
        )
        for ref in evidence_refs
    )


def _ensure_conservative_baseline(
    result: FinalStrategyResult,
    eda: dict[str, Any],
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> None:
    if any(
        "baseline" in action.action.lower()
        and (
            "simple" in action.action.lower()
            or "baseline_evidence" in " ".join(action.evidence_refs)
        )
        for action in _all_actions(result)
    ):
        return
    hypothesis_ids = _ids_for_categories(
        category_ids, known_ids, "baseline", "metric", "validation"
    )
    if not hypothesis_ids:
        return
    task_type = (result.task_type or "").lower()
    if "ranking" in task_type:
        baseline = "Establish a simple ranking-compatible baseline with group-aware validation."
    elif "regression" in task_type:
        baseline = "Establish a simple linear/ridge or tree baseline using safe features."
    elif "classification" in task_type:
        baseline = "Establish a simple linear/logistic or tree baseline using safe features."
    else:
        baseline = "Establish a simple task-appropriate baseline using safe features."
    refs = (
        ["baseline_evidence", "validation_evidence.primary_validation"]
        if eda.get("baseline_evidence")
        else ["validation_evidence.primary_validation"]
        if _primary_validation_method_from_dict(eda)
        else ["final_synthesizer.repaired"]
    )
    action = FinalStrategyAction(
        priority="P0",
        action=baseline,
        reason="A conservative baseline anchors later model and feature comparisons.",
        evidence_refs=refs,
        related_hypothesis_ids=hypothesis_ids,
        confidence="medium",
    )
    _add_grounding_action(result, "baseline_findings", action)


def _ensure_primary_id_safety_action(
    result: FinalStrategyResult,
    eda: dict[str, Any],
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> None:
    if not _primary_id_from_eda(eda) or any(
        "primary id" in action.action.lower()
        and any(marker in action.action.lower() for marker in ("exclude", "do not", "keep"))
        for action in _all_actions(result)
    ):
        return
    hypothesis_ids = _ids_for_categories(
        category_ids, known_ids, "schema", "leakage"
    )
    if not hypothesis_ids:
        return
    action = FinalStrategyAction(
        priority="P0",
        action=(
            "Keep the primary ID excluded from model features and use it only for "
            "submission alignment or row tracking."
        ),
        reason="EDA assigned this column the primary identifier role.",
        evidence_refs=["inferred_schema.primary_id_column"],
        related_hypothesis_ids=hypothesis_ids,
        confidence="high",
        evidence_origin="Safety-warning",
    )
    _add_grounding_action(result, "what_not_to_do", action)


def _ensure_leakage_safety_action(
    result: FinalStrategyResult,
    eda: dict[str, Any],
    known_ids: list[str],
    category_ids: dict[str, list[str]],
) -> None:
    actionable_checks = [
        item
        for item in eda.get("leakage_evidence") or []
        if isinstance(item, dict)
        and item.get("check_id")
        and (
            item.get("status") in {"warning", "not_testable"}
            or (
                item.get("status") == "failed"
                and item.get("severity") not in {"high", "critical"}
            )
        )
    ]
    if not actionable_checks or any(
        "leak" in action.action.lower() or "encoding" in action.action.lower()
        for action in _all_actions(result)
    ):
        return
    hypothesis_ids = _ids_for_categories(category_ids, known_ids, "leakage", "schema")
    if not hypothesis_ids:
        return
    action = FinalStrategyAction(
        priority="P1",
        action="Audit warning-level or incomplete leakage checks before training.",
        reason=(
            "Warning or not-testable leakage checks require diagnostics but do not prove "
            "that critical leakage exists."
        ),
        evidence_refs=[
            f"leakage_evidence.{item['check_id']}" for item in actionable_checks
        ],
        related_hypothesis_ids=hypothesis_ids,
        confidence="high",
        evidence_origin="Safety-warning",
    )
    _add_grounding_action(result, "leakage_and_data_quality", action)


def _append_evidence_availability_limitations(
    result: FinalStrategyResult,
    eda: dict[str, Any],
) -> None:
    notes = []
    if not eda.get("baseline_evidence"):
        notes.append("Baseline evidence unavailable or skipped; no baseline result is treated as confirmed.")
    if not eda.get("notebook_static_analysis"):
        notes.append("Notebook static analysis unavailable or skipped.")
    if not eda.get("drift_evidence"):
        notes.append("Drift evidence unavailable; absence of evidence is not evidence of no drift.")
    validation = _as_dict(eda.get("validation_evidence"))
    if "target_available" in validation and not validation.get("target_available"):
        notes.append("Target-based validation evidence was not testable because the target was unavailable.")
    if validation and not validation.get("time_columns"):
        notes.append("No supported time column was identified for temporal validation.")
    if validation and not validation.get("group_columns"):
        notes.append("No supported group column was identified for group-aware validation.")
    for note in notes:
        if note not in result.limitations:
            result.limitations.append(note)


def _primary_id_from_eda(eda: dict[str, Any]) -> str | None:
    schema = _as_dict(eda.get("inferred_schema"))
    primary_id = schema.get("primary_id_column")
    if not primary_id and isinstance(schema.get("global_roles"), dict):
        primary_id = schema["global_roles"].get("primary_id_column")
    return str(primary_id) if primary_id else None


def _primary_validation_method_from_dict(eda: dict[str, Any]) -> str | None:
    validation = _as_dict(eda.get("validation_evidence"))
    primary = _as_dict(validation.get("primary_validation"))
    method = primary.get("method")
    return str(method) if method else None


def _unsafe_primary_id_feature_text(text: str, primary_id: str | None) -> bool:
    id_terms = ["primary id", "row id", "identifier"]
    if primary_id:
        id_terms.append(primary_id.lower())
    return any(
        re.search(
            rf"\b(?:use|include|add|model|train on)\b.{{0,35}}\b{re.escape(term)}\b.{{0,25}}\b(?:feature|predictor|signal)\b",
            text,
        )
        for term in id_terms
    )


def _unsafe_primary_id_split_text(text: str, primary_id: str | None) -> bool:
    id_terms = ["primary id", "row id", "identifier", "index"]
    if primary_id:
        id_terms.append(primary_id.lower())
    return any(
        re.search(
            rf"\b(?:split|fold|group|tune)\b.{{0,30}}\b(?:by|using|on)\b.{{0,12}}\b{re.escape(term)}\b",
            text,
        )
        for term in id_terms
    )


def _drift_is_id_artifact(
    text: str,
    primary_id: str | None,
    eda: dict[str, Any],
) -> bool:
    if "drift" not in text and "psi" not in text and "separab" not in text:
        return False
    drift = _as_dict(eda.get("drift_evidence"))
    excluded = _string_values(drift.get("excluded_columns"))
    adversarial = _as_dict(drift.get("adversarial_validation"))
    excluded.extend(_string_values(adversarial.get("excluded_columns")))
    id_terms = {"primary id", "identifier", "index", *(item.lower() for item in excluded)}
    if primary_id:
        id_terms.add(primary_id.lower())
    return any(term and term in text for term in id_terms) or _id_artifact_drives_drift(
        drift,
        primary_id,
        excluded,
    )


def _id_artifact_drives_drift(
    drift: dict[str, Any],
    primary_id: str | None,
    excluded_columns: list[str],
) -> bool:
    artifact_names = {item.lower() for item in excluded_columns}
    artifact_names.update({"index", "row_index"})
    if primary_id:
        artifact_names.add(primary_id.lower())
    feature_severity = str(
        drift.get("feature_drift_severity")
        or drift.get("safe_feature_drift_severity")
        or ""
    ).lower()
    overall_severity = str(drift.get("severity") or "").lower()
    numeric_psi = _as_dict(drift.get("numeric_psi"))
    columns = numeric_psi.get("columns") or []
    high_columns = {
        str(item.get("column") or item.get("name") or "").lower()
        for item in columns
        if isinstance(item, dict)
        and str(item.get("severity") or "").lower() in {"high", "critical"}
    }
    return (
        overall_severity in {"high", "critical"}
        and feature_severity in {"", "low", "medium"}
        and bool(high_columns)
        and high_columns <= artifact_names
    )


def _section_relevance(section_id: str, category: str) -> int:
    preferred = {
        "validation": "metric_and_validation",
        "metric": "metric_and_validation",
        "leakage": "leakage_and_data_quality",
        "drift": "drift_and_leaderboard_risk",
        "schema": "dataset_facts_from_eda",
        "feature": "feature_priorities",
        "baseline": "baseline_findings",
        "fallback": "experiments_queue",
    }
    return 2 if preferred.get(category) == section_id else 1


def _priority_rank(priority: Priority) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[priority]


def _stronger_origin(left: EvidenceOrigin, right: EvidenceOrigin) -> EvidenceOrigin:
    rank = {
        "Safety-warning": 0,
        "EDA-confirmed": 1,
        "EDA-inferred": 2,
        "Source-supported": 3,
        "Hypothesis-to-test": 4,
        "Fallback-generated": 5,
    }
    return left if rank[left] <= rank[right] else right


def _is_test_wording(text: str) -> bool:
    return text.lower().startswith(("test ", "evaluate ", "compare ", "probe ", "try "))


def _looks_advanced_model(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "boosted ensemble",
            "gradient boosting",
            "random forest",
            "catboost",
            "lightgbm",
            "xgboost",
            "neural network",
            "transformer",
            "stacking",
        )
    )


def _unique_model_actions(
    actions: list[FinalStrategyAction],
) -> list[FinalStrategyAction]:
    unique: list[FinalStrategyAction] = []
    for action in actions:
        if not any(_actions_semantically_similar(action, existing) for existing in unique):
            unique.append(action)
    return unique


def _metric_name(metric: dict[str, Any]) -> str:
    return str(metric.get("name") or metric.get("metric_name") or "unknown")


def _evidence_availability_status(actions: list[FinalStrategyAction]) -> str:
    categories = {
        _primary_evidence_category(action.evidence_refs)
        for action in actions
        if _has_eda_evidence(action.evidence_refs)
    }
    return "available: " + ", ".join(sorted(categories)) if categories else "limited"


def _append_summary_actions(
    lines: list[str],
    heading: str,
    actions: list[FinalStrategyAction],
) -> None:
    lines.extend(["", f"## {heading}"])
    if not actions:
        lines.append("- None confirmed.")
        return
    for action in actions:
        lines.append(f"- {action.priority} [{action.evidence_origin}]: {action.action}")


def _apply_eda_grounding(
    result: FinalStrategyResult,
    eda_evidence_pack: EdaEvidencePack,
    research_hypotheses: ResearchHypotheses,
) -> None:
    _, category_ids = _build_hypothesis_lookup(
        research_hypotheses.model_dump(mode="json").get("hypotheses", []),
        eda_evidence_pack.model_dump(mode="json"),
    )
    validation_action = _validation_grounding_action(eda_evidence_pack, category_ids)
    if validation_action is not None:
        _add_grounding_action(result, "metric_and_validation", validation_action)

    id_action = _primary_id_grounding_action(eda_evidence_pack, category_ids)
    if id_action is not None:
        _add_grounding_action(result, "what_not_to_do", id_action)

    target_encoding_action = _target_encoding_grounding_action(eda_evidence_pack, category_ids)
    if target_encoding_action is not None:
        _add_grounding_action(result, "leakage_and_data_quality", target_encoding_action)

    drift_action = _drift_grounding_action(eda_evidence_pack, category_ids)
    if drift_action is not None:
        _add_grounding_action(result, "drift_and_leaderboard_risk", drift_action)


def _validation_grounding_action(
    eda_evidence_pack: EdaEvidencePack,
    category_ids: dict[str, list[str]],
) -> FinalStrategyAction | None:
    method = _primary_validation_method(eda_evidence_pack)
    if not method:
        return None
    strategy = method if method in _valid_validation_methods() else None
    return FinalStrategyAction(
        action_id=f"eda_validation_{_safe_id(method)}",
        priority="P0",
        action=f"Use {method} as the primary validation policy selected by EDA.",
        reason="EDA validation evidence selected this primary policy; do not replace it without a dataset-specific caveat.",
        evidence_refs=["validation_evidence.primary_validation"],
        related_hypothesis_ids=_related_ids(category_ids, "validation"),
        eda_result_refs=["validation_evidence.primary_validation"],
        validation_strategy=strategy,
        confidence="high",
    )


def _primary_id_grounding_action(
    eda_evidence_pack: EdaEvidencePack,
    category_ids: dict[str, list[str]],
) -> FinalStrategyAction | None:
    schema = _as_dict(eda_evidence_pack.inferred_schema)
    primary_id = schema.get("primary_id_column")
    if not primary_id and isinstance(schema.get("global_roles"), dict):
        primary_id = schema["global_roles"].get("primary_id_column")
    if not primary_id:
        return None
    return FinalStrategyAction(
        action_id=f"eda_do_not_use_id_{_safe_id(str(primary_id))}",
        priority="P0",
        action=f"Do not use `{primary_id}` as a predictive feature by default.",
        reason="EDA inferred this column as the primary identifier; it should be reserved for joins/submission alignment unless validated otherwise.",
        evidence_refs=["inferred_schema.primary_id_column"],
        related_hypothesis_ids=_related_ids(category_ids, "schema"),
        eda_result_refs=["inferred_schema.primary_id_column"],
        confidence="high",
    )


def _target_encoding_grounding_action(
    eda_evidence_pack: EdaEvidencePack,
    category_ids: dict[str, list[str]],
) -> FinalStrategyAction | None:
    probe = _target_encoding_probe(eda_evidence_pack)
    if probe is None:
        return None
    status = str(probe.get("status") or "unsafe")
    if status != "unsafe" and probe.get("leakage_risk") != "high":
        return None
    feature_family = str(
        probe.get("feature_family") or "naive_target_encoding_or_woe"
    )
    evidence_ref = f"feature_probe_evidence.{feature_family}"
    return FinalStrategyAction(
        action_id="eda_no_naive_target_encoding",
        priority="P0",
        action="Do not use naive target encoding or WoE; use OOF/fold-fitted encoding only.",
        reason="EDA marked target encoding as unsafe without a fold-fitted policy, so global target statistics would leak validation labels.",
        evidence_refs=[evidence_ref],
        related_hypothesis_ids=_related_ids(category_ids, "feature", "leakage"),
        eda_result_refs=[evidence_ref],
        confidence="high",
    )


def _drift_grounding_action(
    eda_evidence_pack: EdaEvidencePack,
    category_ids: dict[str, list[str]],
) -> FinalStrategyAction | None:
    drift = _as_dict(eda_evidence_pack.drift_evidence)
    severity = drift.get("feature_drift_severity") or drift.get("severity")
    if severity not in {"high", "critical"}:
        return None
    return FinalStrategyAction(
        action_id=f"eda_drift_{severity}",
        priority="P1",
        action="Treat high train/test drift as leaderboard-risk diagnostics before trusting public LB movement.",
        reason=f"EDA drift evidence reported {severity} feature drift.",
        evidence_refs=[
            "drift_evidence.feature_drift_severity"
            if drift.get("feature_drift_severity") is not None
            else "drift_evidence.severity"
        ],
        related_hypothesis_ids=_related_ids(category_ids, "drift", "leaderboard"),
        eda_result_refs=[
            "drift_evidence.feature_drift_severity"
            if drift.get("feature_drift_severity") is not None
            else "drift_evidence.severity"
        ],
        confidence="medium",
    )


def _add_grounding_action(
    result: FinalStrategyResult,
    section_id: str,
    action: FinalStrategyAction,
) -> None:
    if _has_similar_action(result, action):
        return
    if not action.action_id:
        canonical, _ = canonicalize_final_strategy_actions({"actions": [
            action.model_dump(mode="json")
        ]})
        action = FinalStrategyAction.model_validate(canonical["actions"][0])
    result.actions.append(action)
    section = _find_or_create_section(result, section_id, action)
    if action.action_id not in section.action_ids:
        section.action_ids.append(action.action_id)
    for ref in action.evidence_refs:
        if ref not in section.evidence_refs:
            section.evidence_refs.append(ref)
    for hypothesis_id in action.related_hypothesis_ids:
        if hypothesis_id not in section.related_hypothesis_ids:
            section.related_hypothesis_ids.append(hypothesis_id)


def _has_similar_action(result: FinalStrategyResult, action: FinalStrategyAction) -> bool:
    action_text = _normalize_text(action.action)
    for existing in _all_actions(result):
        existing_text = _normalize_text(existing.action)
        if existing.action_id and existing.action_id == action.action_id:
            return True
        if action_text and action_text in existing_text:
            return True
        if action.action_id == "eda_no_naive_target_encoding" and "target encoding" in existing_text:
            return True
    return False


def _find_or_create_section(
    result: FinalStrategyResult,
    section_id: str,
    action: FinalStrategyAction,
) -> FinalStrategySection:
    for section in result.sections:
        if section.section_id == section_id:
            return section
    section = FinalStrategySection(
        section_id=section_id,
        title=_title_from_id(section_id),
        summary=action.reason,
        evidence_refs=list(action.evidence_refs),
        related_hypothesis_ids=list(action.related_hypothesis_ids),
    )
    result.sections.append(section)
    return section


def _related_ids(category_ids: dict[str, list[str]], *categories: str) -> list[str]:
    ids: list[str] = []
    for category in categories:
        ids.extend(category_ids.get(category, []))
    if ids:
        return ids
    for values in category_ids.values():
        if values:
            return [values[0]]
    return []


def _target_encoding_probe(eda_evidence_pack: EdaEvidencePack) -> dict[str, Any] | None:
    for item in eda_evidence_pack.feature_probe_evidence:
        probe = _as_dict(item)
        if probe.get("feature_family") in {
            "naive_target_encoding_or_woe",
            "target_encoding_or_woe",
        }:
            return probe
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "value"


def _valid_validation_methods() -> set[str]:
    return set(FinalValidationMethod.__args__)


def _normalize_sections(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        sections = []
        for section_id, section in value.items():
            section_payload = dict(section) if isinstance(section, dict) else {"summary": str(section)}
            section_payload.setdefault("section_id", str(section_id))
            section_payload.setdefault("title", _title_from_id(str(section_id)))
            sections.append(section_payload)
        return sections
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _looks_like_final_strategy_draft(payload: dict[str, Any]) -> bool:
    if any(key in payload for key in ("executive_summary", "warnings")):
        return True
    for action in payload.get("actions") or []:
        if isinstance(action, dict) and "support_refs" in action:
            return True
    for section in payload.get("sections") or []:
        if not isinstance(section, dict):
            continue
        if "narrative" in section or "evidence_summary_refs" in section:
            return True
        if any(
            isinstance(action, dict) and "support_refs" in action
            for action in section.get("actions") or []
        ):
            return True
    return False


def _normalize_actions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _title_from_id(section_id: str) -> str:
    return section_id.replace("_", " ").title()


def _primary_validation_method(eda_evidence_pack: EdaEvidencePack) -> str | None:
    validation = eda_evidence_pack.validation_evidence or {}
    primary = validation.get("primary_validation") or {}
    method = primary.get("method")
    return str(method) if method else None


def _enforce_primary_validation(
    result: FinalStrategyResult,
    eda_evidence_pack: EdaEvidencePack,
) -> None:
    primary_method = _primary_validation_method(eda_evidence_pack)
    if primary_method != "stratified_kfold":
        return
    invalid_actions = [
        action.action_id or action.action
        for action in _all_actions(result)
        if action.validation_strategy in TEMPORAL_VALIDATION_METHODS
    ]
    if result.recommended_validation in TEMPORAL_VALIDATION_METHODS or invalid_actions:
        raise ValueError(
            "Final strategy attempted to override validation_evidence.primary_validation "
            "stratified_kfold with temporal validation."
        )


def _all_actions(result: FinalStrategyResult) -> list[FinalStrategyAction]:
    return list(result.actions)


__all__ = [
    "ActionProvenance",
    "ActionEvidenceResolution",
    "EvidenceOrigin",
    "FinalStrategyAction",
    "FinalStrategyResult",
    "FinalStrategySection",
    "HypothesisToEdaLink",
    "FinalValidationMethod",
    "REQUIRED_SECTION_IDS",
    "SourceToHypothesisLink",
    "build_fallback_final_strategy",
    "build_deterministic_provenance_links",
    "migrate_final_strategy_hypothesis_references", "migrate_hypothesis_references",
    "resolve_composite_action_references", "resolve_final_strategy_composite_references",
    "UnsupportedFinalStrategyActionError", "compile_final_strategy_action_support",
    "enforce_action_evidence_support",
    "ActionCanonicalizationDiagnostics", "canonicalize_final_strategy_actions",
    "canonicalize_semantic_strategy_actions", "validate_semantic_action_postconditions",
    "postprocess_final_strategy_result",
    "repair_final_strategy_payload",
    "render_final_strategy",
    "render_final_strategy_summary",
    "resolve_action_evidence_refs",
    "resolve_final_strategy_action_evidence",
    "classify_action",
    "synthesize_final_strategy",
    "validate_rendered_strategy_quality",
]
