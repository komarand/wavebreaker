from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.contracts.action_support import (
    FinalStrategyCompilationContext,
    FinalStrategyCompilationReport,
    UnsupportedFinalStrategyActionError,
    compile_final_strategy_action_support,
    enforce_action_evidence_support,
)
from kaggle_researcher.contracts.action_canonicalization import (
    ActionCanonicalizationDiagnostics,
    canonicalize_final_strategy_actions,
)
from kaggle_researcher.contracts.normalization import normalize_contract_payload
from kaggle_researcher.contracts.evidence import (
    build_evidence_registry,
    generate_allowed_evidence_refs,
)
from kaggle_researcher.contracts.experiments import (
    CrossNamespaceReferenceError,
    FORBIDDEN_CONTEXT_LABELS,
    ReferenceIssue,
    repair_final_experiment_references,
)
from kaggle_researcher.contracts.final_strategy import (
    Confidence,
    EvidenceOrigin,
    FALLBACK_LIMITATION,
    FinalStrategyAction,
    FinalStrategyResult,
    FinalStrategySection,
    FinalValidationMethod,
    Priority,
    REPAIR_LIMITATION,
    REQUIRED_SECTION_IDS,
    TEMPORAL_VALIDATION_METHODS,
)
from kaggle_researcher.contracts.final_strategy_draft import FinalStrategyDraft
from kaggle_researcher.contracts.final_strategy_compilation import (
    FinalStrategyCompilationDiagnostics,
    FinalStrategyCompilationError,
    FinalStrategyRepairError,
    FinalStrategySchemaValidationError,
)
from kaggle_researcher.contracts.bundle_validation import validate_final_synthesis_bundle
from kaggle_researcher.contracts.composite_reference_resolution import (
    CompositeReferenceResolutionDiagnostics,
    resolve_composite_action_references,
    resolve_final_strategy_composite_references,
)
from kaggle_researcher.contracts.hypothesis_reference_migration import (
    HypothesisReferenceMigrationDiagnostics,
    migrate_final_strategy_hypothesis_references,
    migrate_hypothesis_references,
)
from kaggle_researcher.contracts.reference_catalog import (
    ReferenceCatalog,
    build_final_strategy_reference_catalog,
)
from kaggle_researcher.contracts.registries import ContractRegistries
from kaggle_researcher.contracts.synthesis_context import FinalSynthesisContext
from kaggle_researcher.eda.schemas import EdaEvidencePack, ResearchHypotheses
from kaggle_researcher.schemas import PlanData, RetrievedDocument


logger = logging.getLogger(__name__)


async def synthesize_final_strategy(
    *,
    context: FinalSynthesisContext,
    registries: ContractRegistries,
    client: DeepSeekClient,
    model: str,
) -> FinalStrategyResult:
    experiment_registry = registries.experiments
    competition_desc = context.competition_desc
    plan_data = context.plan_data
    retrieved_documents = context.retrieved_documents
    research_hypotheses = context.research_hypotheses
    eda_evidence_pack = context.eda_evidence_pack
    eda_summary_text = context.eda_summary_text
    try:
        reference_catalog = build_final_strategy_reference_catalog(
            eda_evidence_pack,
            research_hypotheses=research_hypotheses,
            source_claim_ids=[document.id for document in retrieved_documents],
        )
    except FinalStrategyCompilationError:
        raise
    except Exception as exc:
        raise FinalStrategyRepairError(
            "Final strategy failed during reference_catalog_build.",
            phase="reference_catalog_build",
        ) from exc
    raw = await client.chat_json(
        model=model,
        system_prompt=FINAL_SYNTHESIZER_SYSTEM_PROMPT,
        user_prompt=_build_final_synthesizer_prompt(
            context=context,
            registries=registries,
            reference_catalog=reference_catalog,
        ),
    )
    result = _result_from_payload(
        raw,
        plan_data=plan_data,
        eda_evidence_pack=eda_evidence_pack,
        research_hypotheses=research_hypotheses,
        eda_summary=eda_summary_text,
        model=model,
        reference_catalog=reference_catalog,
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
    allowed_eda_refs = set(generate_allowed_evidence_refs(eda_evidence_pack))
    evidence_registry = registries.evidence
    _derive_eda_result_refs(result, allowed_eda_refs)
    issues = _final_reference_issues(result, registries)
    if issues:
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
        )
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
        remaining_issues = _final_reference_issues(result, registries)
        if remaining_issues:
            raise CrossNamespaceReferenceError(remaining_issues)
    validate_final_synthesis_bundle(
        eda_evidence_pack,
        context.experiment_plan,
        context.review,
        result,
        hypotheses=research_hypotheses,
        source_ids=[document.id for document in retrieved_documents],
        optional_stage_failures=context.optional_stage_failure_messages,
    )
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
    result: FinalStrategyResult, registries: ContractRegistries
) -> list[ReferenceIssue]:
    issues: list[ReferenceIssue] = []
    allowed_evidence = {
        *registries.evidence.ids("eda_evidence"),
        *registries.evidence.ids("source_claim"),
        *registries.evidence.ids("synthetic_inference"),
    }
    allowed_eda = set(registries.evidence.ids("eda_evidence"))
    source_ids = set(registries.evidence.ids("source_claim"))

    def actual(value: str) -> str | None:
        try:
            return registries.namespace_for(value)
        except Exception:
            return "ambiguous"

    for path, action in _iter_actions_with_paths(result):
        checks = (
            ("evidence_refs", action.evidence_refs, allowed_evidence, "evidence"),
            ("eda_result_refs", action.eda_result_refs, allowed_eda, "eda_evidence"),
            ("source_refs", action.source_refs, source_ids, "source"),
            ("hypothesis_ids", set(action.hypothesis_ids) | set(action.related_hypothesis_ids), set(registries.hypotheses.by_id), "hypothesis"),
            ("risk_ids", action.risk_ids, set(registries.risks.by_id), "risk"),
            ("validation_requirement_ids", action.validation_requirement_ids, set(registries.validation_requirements.by_id), "validation_requirement"),
            ("safety_constraint_ids", action.safety_constraint_ids, set(registries.safety_constraints.by_id), "safety_constraint"),
        )
        for field, values, allowed, expected in checks:
            for value in values:
                if value not in allowed:
                    is_context_label = str(value) in FORBIDDEN_CONTEXT_LABELS
                    issues.append(ReferenceIssue(
                        f"{path}.{field}", expected, str(value),
                        "context_label" if is_context_label else actual(str(value)),
                        "context_label_not_reference" if is_context_label
                        else "cross_namespace" if actual(str(value)) else "unknown",
                    ))
        for value in action.experiment_ids:
            if value not in registries.experiments.approved_ids:
                namespace = "rejected_experiment" if value in registries.experiments.rejected_ids else actual(str(value))
                issues.append(ReferenceIssue(
                    f"{path}.experiment_ids", "approved_experiment", str(value), namespace,
                    "rejected_experiment" if namespace == "rejected_experiment" else "unknown_or_unapproved_experiment",
                ))
    for index, section in enumerate(result.sections):
        for value in section.evidence_refs:
            if value not in allowed_evidence:
                is_context_label = str(value) in FORBIDDEN_CONTEXT_LABELS
                issues.append(ReferenceIssue(
                    f"sections[{index}].evidence_refs", "evidence", str(value),
                    "context_label" if is_context_label else actual(str(value)),
                    "context_label_not_reference" if is_context_label
                    else "cross_namespace" if actual(str(value)) else "unknown",
                ))
    return issues


def _iter_actions_with_paths(result: FinalStrategyResult):
    for index, action in enumerate(result.actions):
        yield f"actions[{index}]", action


def _derive_eda_result_refs(
    result: FinalStrategyResult,
    allowed_eda_refs: set[str],
) -> None:
    for action in _all_actions(result):
        if not action.eda_result_refs:
            action.eda_result_refs = [
                reference for reference in action.evidence_refs
                if reference in allowed_eda_refs
            ]


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
        ],
        "allowed_hypothesis_ids": context_payload["allowed_hypothesis_ids"],
        "allowed_experiment_ids": context_payload["allowed_experiment_ids"],
        "approved_experiment_ids": sorted(experiment_registry.approved_experiment_ids),
        "rejected_experiment_ids": context_payload["rejected_experiment_ids"],
        "approved_experiments": context_payload["approved_experiments"],
        "allowed_evidence_refs": context_payload["allowed_evidence_refs"],
        "allowed_eda_result_refs": context_payload["allowed_eda_result_refs"],
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
        "leakage_warnings": [
            item
            for item in eda_evidence_pack.leakage_evidence
            if _as_dict(item).get("status") in {"failed", "warning"}
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
) -> FinalStrategyResult:
    normalized = dict(payload)
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
    normalized, canonicalization_diagnostics = canonicalize_final_strategy_actions(
        normalized
    )
    _preserve_action_canonicalization_diagnostics(
        normalized,
        canonicalization_diagnostics,
    )
    original_for_support_gate = deepcopy(normalized)
    normalized, migration_diagnostics = migrate_final_strategy_hypothesis_references(
        normalized,
        reference_catalog,
    )
    _preserve_hypothesis_migration_diagnostics(normalized, migration_diagnostics)
    normalized, composite_diagnostics = resolve_final_strategy_composite_references(
        normalized,
        reference_catalog,
    )
    _preserve_composite_resolution_diagnostics(normalized, composite_diagnostics)
    normalized, compilation_report = compile_final_strategy_action_support(
        normalized,
        original_payload=original_for_support_gate,
        context=FinalStrategyCompilationContext(reference_catalog=reference_catalog),
    )
    _preserve_action_support_report(normalized, compilation_report)
    hypotheses_payload = research_hypotheses.model_dump(mode="json").get("hypotheses", [])
    eda_payload = eda_evidence_pack.model_dump(mode="json")
    known_ids, _ = _build_hypothesis_lookup(hypotheses_payload, eda_payload)

    try:
        result = FinalStrategyResult.model_validate(normalized)
        if not _contains_unknown_hypothesis_ids(result, set(known_ids)):
            _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
            return result
    except ValidationError as first_error:
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
    try:
        result = FinalStrategyResult.model_validate(repaired)
        _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
        return result
    except ValidationError as second_error:
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
    )
    fallback.setdefault("task_type", plan_data.task_type)
    fallback.setdefault("metric", {"name": plan_data.metric})
    fallback.setdefault("recommended_validation", _primary_validation_method(eda_evidence_pack))
    fallback["models_used"] = {
        **dict(fallback.get("models_used") or {}),
        "final_synthesizer": model,
    }
    result = FinalStrategyResult.model_validate(fallback)
    _apply_eda_grounding(result, eda_evidence_pack, research_hypotheses)
    return result


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
    repairs = list(payload.get("reference_repairs") or [])
    for field_name in ("generated_action_ids", "merged_duplicate_actions"):
        for action_id in getattr(diagnostics, field_name):
            record = {
                "field_path": f"action_canonicalization.{field_name}",
                "original_id": action_id,
                "replacement_id": action_id,
            }
            if record not in repairs:
                repairs.append(record)
    payload["reference_repairs"] = repairs


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
    return repaired


def build_fallback_final_strategy(
    *,
    competition_id: str,
    research_hypotheses: list[dict[str, Any]],
    eda_evidence_pack: dict[str, Any] | None,
    eda_summary: str | None = None,
) -> dict[str, Any]:
    eda = eda_evidence_pack or {}
    known_ids, category_ids = _build_hypothesis_lookup(research_hypotheses, eda)
    if not known_ids:
        raise ValueError(
            "Cannot build a linked final strategy without any known Scout or EDA hypothesis IDs."
        )

    actions_by_section: dict[str, list[dict[str, Any]]] = {
        section_id: [] for section_id in (
            "dataset_facts_from_eda",
            "metric_and_validation",
            "leakage_and_data_quality",
            "drift_and_leaderboard_risk",
            "feature_priorities",
            "baseline_findings",
            "what_not_to_do",
            "experiments_queue",
        )
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

    metric = _as_dict(eda.get("metric_evidence"))
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

    validation = _as_dict(eda.get("validation_evidence"))
    primary_validation = _as_dict(validation.get("primary_validation"))
    method = primary_validation.get("method")
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

    if _has_unsafe_target_encoding(eda):
        target_encoding = _fallback_action(
            "Avoid naive target encoding; use OOF or fold-fitted encoding only.",
            "EDA identified leakage risk in globally fitted target statistics.",
            ["feature_probe_evidence", "leakage_evidence"],
            _ids_for_categories(category_ids, known_ids, "leakage", "schema"),
            priority="P0",
        )
        actions_by_section["leakage_and_data_quality"].append(target_encoding)
        actions_by_section["what_not_to_do"].append(deepcopy(target_encoding))

    drift = _as_dict(eda.get("drift_evidence"))
    severity = drift.get("feature_drift_severity") or drift.get("severity")
    if severity in {"medium", "high", "critical"}:
        actions_by_section["drift_and_leaderboard_risk"].append(
            _fallback_action(
                "Treat train/test drift as a leaderboard-risk diagnostic.",
                f"EDA reported {severity} drift severity.",
                ["drift_evidence"],
                _ids_for_categories(category_ids, known_ids, "drift"),
                priority="P1",
            )
        )

    if _has_high_potential_features(eda):
        actions_by_section["feature_priorities"].append(
            _fallback_action(
                "Prioritize the high-potential feature families identified by EDA.",
                "Feature probes found dataset-supported candidates for early experiments.",
                ["feature_probe_evidence"],
                _ids_for_categories(category_ids, known_ids, "schema", "feature"),
                priority="P1",
            )
        )

    if eda.get("baseline_evidence"):
        actions_by_section["baseline_findings"].append(
            _fallback_action(
                "Reproduce the EDA baseline under the selected validation policy before expanding the model search.",
                "A stable baseline anchors subsequent experiment comparisons.",
                ["baseline_evidence", "validation_evidence.primary_validation"],
                _ids_for_categories(category_ids, known_ids, "metric", "validation", "baseline"),
            )
        )
    actions_by_section["experiments_queue"].append(
        _fallback_action(
            "Run the evidence-backed P0 actions first and record fold-level results.",
            "A small ordered queue makes strategy changes auditable against EDA evidence.",
            ["final_synthesizer.repaired"],
            [known_ids[0]],
            priority="P1",
            confidence="low",
        )
    )

    sections = []
    all_actions = []
    for section_id, section_actions in actions_by_section.items():
        if not section_actions:
            section_actions = [_placeholder_action(known_ids)]
        section_actions = [action for action in section_actions if action is not None]
        all_actions.extend(deepcopy(section_actions))
        sections.append(
            {
                "section_id": section_id,
                "title": _title_from_id(section_id),
                "summary": "Deterministic guidance grounded in available Scout and EDA evidence.",
                "actions": section_actions,
                "evidence_refs": _unique_strings(
                    ref for action in section_actions for ref in action["evidence_refs"]
                ),
                "related_hypothesis_ids": _unique_strings(
                    item
                    for action in section_actions
                    for item in action["related_hypothesis_ids"]
                ),
            }
        )

    return {
        "schema_version": "1.0",
        "competition_id": competition_id,
        "sections": sections,
        "actions": all_actions,
        "limitations": [FALLBACK_LIMITATION],
    }


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


def _fallback_action(
    action: str,
    reason: str,
    evidence_refs: list[str],
    related_hypothesis_ids: list[str],
    *,
    priority: Priority = "P1",
    confidence: Confidence = "medium",
    validation_strategy: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "priority": priority,
        "action": action,
        "reason": reason,
        "evidence_refs": evidence_refs,
        "related_hypothesis_ids": related_hypothesis_ids,
        "confidence": confidence,
    }
    if validation_strategy in _valid_validation_methods():
        payload["validation_strategy"] = validation_strategy
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
    return any(
        hypothesis_id not in known_ids
        for action in _all_actions(result)
        for hypothesis_id in action.related_hypothesis_ids
    )


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

    _deduplicate_strategy_actions(cleaned)
    _append_evidence_availability_limitations(cleaned, eda)
    return FinalStrategyResult.model_validate(cleaned.model_dump(mode="json"))


def render_final_strategy(result: FinalStrategyResult) -> str:
    lines = [
        "# Final Strategy",
        "",
        f"Competition: `{result.competition_id}`",
        f"Task type: `{result.task_type or 'unknown'}`",
        f"Metric: `{_metric_name(result.metric)}`",
        f"Validation: `{result.recommended_validation or 'unknown'}`",
        "",
    ]
    action_map = {
        action.action_id: action
        for action in result.actions
        if action.action_id
    }
    for section in result.sections:
        lines.extend([f"## {section.title}", "", section.summary, ""])
        for action_id in section.action_ids:
            action = action_map[action_id]
            evidence = ", ".join(action.evidence_refs)
            lines.append(
                f"- {action.priority} [{action.evidence_origin}]: "
                f"{action.action} Evidence: {evidence}"
            )
            if action.reason:
                lines.append(f"  Rationale: {action.reason}")
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


def _deduplicate_strategy_actions(result: FinalStrategyResult) -> None:
    canonical, _ = canonicalize_final_strategy_actions(result.model_dump(mode="json"))
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
    if not eda.get("leakage_evidence") or any(
        "leak" in action.action.lower() or "encoding" in action.action.lower()
        for action in _all_actions(result)
    ):
        return
    hypothesis_ids = _ids_for_categories(category_ids, known_ids, "leakage", "schema")
    if not hypothesis_ids:
        return
    action = FinalStrategyAction(
        priority="P0",
        action="Review failed or warning-level leakage checks before training.",
        reason="Skipped or warning-level leakage checks are not proof that the dataset is safe.",
        evidence_refs=["leakage_evidence"],
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
        evidence_refs=["drift_evidence"],
        related_hypothesis_ids=_related_ids(category_ids, "drift", "leaderboard"),
        eda_result_refs=["drift_evidence"],
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
    "EvidenceOrigin",
    "FinalStrategyAction",
    "FinalStrategyResult",
    "FinalStrategySection",
    "FinalValidationMethod",
    "REQUIRED_SECTION_IDS",
    "build_fallback_final_strategy",
    "migrate_final_strategy_hypothesis_references", "migrate_hypothesis_references",
    "resolve_composite_action_references", "resolve_final_strategy_composite_references",
    "UnsupportedFinalStrategyActionError", "compile_final_strategy_action_support",
    "enforce_action_evidence_support",
    "ActionCanonicalizationDiagnostics", "canonicalize_final_strategy_actions",
    "postprocess_final_strategy_result",
    "repair_final_strategy_payload",
    "render_final_strategy",
    "render_final_strategy_summary",
    "synthesize_final_strategy",
    "validate_rendered_strategy_quality",
]
