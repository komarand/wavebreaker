from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping

from pydantic import BaseModel, ConfigDict

from kaggle_researcher.contracts.reference_catalog import ReferenceCatalog


ActionSupportStatus = Literal[
    "supported", "partially_supported", "unsupported", "contradicted"
]
ActionResolutionMethod = Literal[
    "original",
    "hypothesis_link",
    "recommended_action_match",
    "category_fallback",
    "llm_repair",
    "deterministic_fallback",
    "none",
]


class ActionEvidenceResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    raw_action_category: str
    category: str
    original_intent: str
    intent: str
    intent_inference_signals: tuple[str, ...] = ()
    priority: str
    original_priority: str
    normalized_priority: str
    original_mandatory_status: bool = False
    normalized_mandatory_status: bool = False
    original_refs: tuple[str, ...] = ()
    resolved_refs: tuple[str, ...] = ()
    added_refs: tuple[str, ...] = ()
    unresolved_refs: tuple[str, ...] = ()
    unsupported_refs: tuple[str, ...] = ()
    contradictory_refs: tuple[str, ...] = ()
    candidate_refs: tuple[str, ...] = ()
    related_hypothesis_ids: tuple[str, ...] = ()
    related_hypothesis_categories: tuple[str, ...] = ()
    matching_recommended_action_ids: tuple[str, ...] = ()
    matching_leakage_check_refs: tuple[str, ...] = ()
    matching_failed_critical_leakage_refs: tuple[str, ...] = ()
    failed_critical_leakage_count: int = 0
    alignment_status: Literal["valid", "invalid", "not_applicable"] = "not_applicable"
    alignment_reason: str | None = None
    reclassification_reason: str | None = None
    policy_violation_codes: tuple[str, ...] = ()
    support_status: ActionSupportStatus
    resolution_method: ActionResolutionMethod
    resolution_attempts: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class FinalStrategyActionEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[ActionEvidenceResolution, ...] = ()
    fallback_action_ids: tuple[str, ...] = ()

    def for_action(self, action_id: str) -> ActionEvidenceResolution | None:
        return next((item for item in self.actions if item.action_id == action_id), None)


# Compatibility name for callers that treat the immutable catalog as an index.
EvidenceReferenceIndex = ReferenceCatalog


@dataclass(frozen=True)
class ActionIntentPolicy:
    intent: str
    allowed_hypothesis_categories: frozenset[str]
    required_evidence_roots: frozenset[str]
    requires_failed_check: bool = False
    allowed_severities: frozenset[str] = frozenset()
    default_priority: str | None = None
    mandatory_when_supported: bool = False
    safe_downgrade_intent: str | None = None


ACTION_INTENT_POLICIES: dict[str, ActionIntentPolicy] = {
    "leakage.block_critical_issue": ActionIntentPolicy(
        intent="leakage.block_critical_issue",
        allowed_hypothesis_categories=frozenset({"leakage", "data_quality"}),
        required_evidence_roots=frozenset({"leakage_evidence"}),
        requires_failed_check=True,
        allowed_severities=frozenset({"high", "critical"}),
        default_priority="P0",
        mandatory_when_supported=True,
        safe_downgrade_intent="leakage.audit_warning",
    ),
    "leakage.audit_warning": ActionIntentPolicy(
        intent="leakage.audit_warning",
        allowed_hypothesis_categories=frozenset({
            "leakage", "data_quality", "feature", "baseline",
        }),
        required_evidence_roots=frozenset({"leakage_evidence"}),
        allowed_severities=frozenset({"medium", "high", "critical"}),
    ),
    "leakage.run_required_checks": ActionIntentPolicy(
        intent="leakage.run_required_checks",
        allowed_hypothesis_categories=frozenset({
            "leakage", "data_quality", "feature", "baseline",
        }),
        required_evidence_roots=frozenset({"leakage_evidence"}),
    ),
    "baseline.run_sanity_baseline": ActionIntentPolicy(
        intent="baseline.run_sanity_baseline",
        allowed_hypothesis_categories=frozenset({
            "baseline", "metric", "validation",
        }),
        required_evidence_roots=frozenset({
            "baseline_evidence", "baseline_ablation_evidence",
            "validation_evidence", "metric_evidence", "recommended_next_actions",
        }),
    ),
    "baseline.exclude_known_unsafe_columns": ActionIntentPolicy(
        intent="baseline.exclude_known_unsafe_columns",
        allowed_hypothesis_categories=frozenset({"baseline", "leakage", "data_quality"}),
        required_evidence_roots=frozenset({"leakage_evidence", "inferred_schema"}),
    ),
    "validation.freeze_primary_splits": ActionIntentPolicy(
        intent="validation.freeze_primary_splits",
        allowed_hypothesis_categories=frozenset({"validation"}),
        required_evidence_roots=frozenset({"validation_evidence"}),
    ),
}


ACTION_ALLOWED_EVIDENCE_ROOTS: dict[str, frozenset[str]] = {
    "validation": frozenset({
        "validation_evidence", "metric_evidence", "hypothesis_results",
        "recommended_next_actions",
    }),
    "metric": frozenset({
        "metric_evidence", "validation_evidence", "hypothesis_results",
        "recommended_next_actions",
    }),
    "leakage": frozenset({
        "leakage_evidence", "validation_evidence", "hypothesis_results",
        "recommended_next_actions", "safety_constraints",
    }),
    "data_quality": frozenset({
        "file_inventory", "inferred_schema", "table_profiles", "leakage_evidence",
        "hypothesis_results",
    }),
    "relationship": frozenset({
        "relationship_evidence", "inferred_schema", "hypothesis_results",
        "recommended_next_actions",
    }),
    "drift": frozenset({
        "drift_evidence", "validation_evidence", "hypothesis_results",
        "recommended_next_actions",
    }),
    "baseline": frozenset({
        "baseline_evidence", "baseline_ablation_evidence", "validation_evidence", "metric_evidence",
        "recommended_next_actions",
    }),
    "feature": frozenset({
        "feature_probe_evidence", "feature_diagnostics", "baseline_ablation_evidence", "relationship_evidence",
        "leakage_evidence", "validation_evidence", "hypothesis_results",
    }),
    "modeling": frozenset({
        "baseline_evidence", "metric_evidence", "validation_evidence",
        "feature_probe_evidence", "hypothesis_results",
    }),
    "submission": frozenset({
        "inferred_schema", "file_inventory", "metric_evidence", "validation_evidence",
        "drift_evidence", "leakage_evidence", "hypothesis_results",
    }),
}

_METHOD_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stratified_group_kfold", ("stratified group kfold", "stratified_group_kfold")),
    ("ranking_group_cv", ("ranking group cv", "ranking_group_cv")),
    ("temporal_holdout", ("temporal holdout", "temporal_holdout", "time holdout")),
    ("temporal_cv", ("temporal cv", "temporal_cv", "time series split", "timeseriessplit")),
    ("stratified_kfold", (
        "stratified kfold", "stratified_kfold", "stratifiedkfold", "stratified cv",
    )),
    ("group_kfold", ("group kfold", "group_kfold", "grouped validation")),
    ("kfold", (" kfold", "kfold", "cross validation", "cross-validation")),
)
_TOKEN_ALIASES = {
    "cv": "validation", "crossvalidation": "validation", "cross-validation": "validation",
    "stratifiedkfold": "stratified_kfold", "probability": "probabilities",
    "scores": "probabilities", "score": "probabilities", "folds": "fold",
    "splits": "split", "grouped": "group",
}
_STOP_TOKENS = {
    "a", "an", "and", "as", "at", "be", "before", "by", "for", "from", "in",
    "into", "it", "of", "on", "or", "the", "this", "to", "use", "using", "with",
    "selected", "apply", "run", "make", "ensure",
}


def classify_action(action: Mapping[str, Any]) -> tuple[str, str]:
    """Return the conservative category and canonical intent for an action."""

    text = _action_text(action)
    primary_text = str(action.get("action") or "").lower().replace("_", " ")
    explicit_intent = str(action.get("intent") or action.get("action_intent") or "").strip()
    if explicit_intent in ACTION_INTENT_POLICIES:
        return explicit_intent.split(".", 1)[0], explicit_intent
    explicit = str(
        action.get("category") or action.get("action_type") or action.get("kind") or ""
    ).strip().lower().replace("-", "_")
    action_id = str(action.get("action_id") or "").lower()

    category = _canonical_category(explicit)
    # A baseline remains a baseline when its wording says "leak-free" or
    # "leakage-safe". Those adjectives describe execution hygiene, not a
    # confirmed leakage finding.
    if _has_any(primary_text, ("baseline", "sanity floor", "accuracy floor", "benchmark")):
        category = "baseline"
    if not category:
        category = _category_from_text(primary_text)
    if not category:
        category = _category_from_text(text)
    if not category:
        category = _category_from_prefix(action_id)

    if category == "validation":
        if _has_any(text, ("primary validation", "validation policy", "model comparison")):
            return category, "validation.use_primary_policy"
        if _has_any(text, ("freeze fold", "freeze split", "persist fold", "fold assignment")):
            return category, "validation.freeze_folds"
        return category, "validation.use_primary_policy"
    if category == "metric" and _has_any(
        text, ("probabil", "hard label", "rank", "prediction type", "output type")
    ):
        return category, "metric.use_required_prediction_type"
    if category == "leakage":
        if _has_any(text, ("warning", "audit", "suspicious")):
            return category, "leakage.audit_warning"
        if _has_any(text, ("block", "exclude", "remove", "drop")) and _has_any(
            text, ("leakage check", "target in test", "unsafe source", "leaking column")
        ):
            return category, "leakage.block_critical_issue"
        if _has_any(text, ("check", "verify", "investigate", "review", "before enabling")):
            return category, "leakage.run_required_checks"
        return category, "leakage.run_required_checks"
    if category == "submission":
        return category, "data.verify_submission_contract"
    if category == "relationship" and _has_any(text, ("aggregate", "join", "many to one")):
        return category, "relationship.aggregate_before_join"
    if category == "baseline":
        return category, "baseline.run_sanity_baseline"
    return category or "modeling", f"{category or 'modeling'}.action"


def resolve_action_evidence_refs(
    action: Any,
    *,
    eda_evidence_pack: Any,
    research_hypotheses: Any,
    allowed_reference_index: EvidenceReferenceIndex,
) -> ActionEvidenceResolution:
    """Resolve factual action support through the existing immutable reference catalog.

    Mutable mappings are updated in place with the resolved evidence and hypothesis links.
    The returned report keeps missing, broken, semantically unsupported, and contradictory
    references distinct.
    """

    action_payload = _as_mapping(action)
    eda = _model_payload(eda_evidence_pack)
    research = _model_payload(research_hypotheses)
    report, related = _resolve_action(
        action_payload,
        eda=eda,
        research=research,
        catalog=allowed_reference_index,
    )
    if isinstance(action, MutableMapping):
        action["evidence_refs"] = list(report.resolved_refs)
        action["related_hypothesis_ids"] = list(related)
        action["hypothesis_ids"] = list(related)
    return report


def resolve_final_strategy_action_evidence(
    payload: Mapping[str, Any],
    *,
    eda_evidence_pack: Any,
    research_hypotheses: Any,
    allowed_reference_index: EvidenceReferenceIndex,
) -> tuple[dict[str, Any], FinalStrategyActionEvidenceReport]:
    """Ground strategy actions and add only evidence-backed required canonical actions."""

    resolved_payload = deepcopy(dict(payload))
    eda = _model_payload(eda_evidence_pack)
    research = _model_payload(research_hypotheses)
    actions = [item for item in resolved_payload.get("actions") or [] if isinstance(item, dict)]
    reports: list[ActionEvidenceResolution] = []

    for action in actions:
        initial, related = _resolve_action(
            action, eda=eda, research=research, catalog=allowed_reference_index
        )
        if (
            initial.support_status != "supported"
            and str(action.get("priority") or "").upper() == "P0"
            and initial.intent in {
                "validation.use_primary_policy",
                "metric.use_required_prediction_type",
                "leakage.block_critical_issue",
            }
            and _apply_canonical_fallback(action, initial.intent, eda)
        ):
            repaired, related = _resolve_action(
                action, eda=eda, research=research, catalog=allowed_reference_index
            )
            initial = repaired.model_copy(update={
                "resolution_method": "deterministic_fallback",
                "resolution_attempts": _unique_tuple([
                    *initial.resolution_attempts, "deterministic_fallback",
                ]),
                "unresolved_refs": initial.unresolved_refs,
                "unsupported_refs": initial.unsupported_refs,
                "contradictory_refs": initial.contradictory_refs,
                "notes": _unique_tuple([
                    *initial.notes,
                    "The required canonical action was rewritten to match structured EDA evidence.",
                ]),
            })
        action["evidence_refs"] = list(initial.resolved_refs)
        action["related_hypothesis_ids"] = list(related)
        action["hypothesis_ids"] = list(related)
        reports.append(initial)

    fallback_ids: list[str] = []
    required = _required_fallback_actions(eda, research, actions)
    known_ids = {str(item.get("action_id") or "") for item in actions}
    for action, section_id in required:
        action_id = str(action["action_id"])
        if action_id in known_ids:
            continue
        report, related = _resolve_action(
            action, eda=eda, research=research, catalog=allowed_reference_index
        )
        if report.support_status != "supported" or not related:
            continue
        action["evidence_refs"] = list(report.resolved_refs)
        action["related_hypothesis_ids"] = list(related)
        action["hypothesis_ids"] = list(related)
        actions.append(action)
        known_ids.add(action_id)
        fallback_ids.append(action_id)
        reports.append(report.model_copy(update={
            "resolution_method": "deterministic_fallback",
            "resolution_attempts": _unique_tuple([
                *report.resolution_attempts, "deterministic_fallback",
            ]),
            "notes": _unique_tuple([
                *report.notes,
                "Added an evidence-backed mandatory canonical action missing from the LLM payload.",
            ]),
        }))
        _attach_action_to_section(resolved_payload, action, section_id)

    resolved_payload["actions"] = actions
    return resolved_payload, FinalStrategyActionEvidenceReport(
        actions=tuple(reports), fallback_action_ids=tuple(fallback_ids)
    )


def _resolve_action(
    action: Mapping[str, Any],
    *,
    eda: dict[str, Any],
    research: dict[str, Any],
    catalog: ReferenceCatalog,
) -> tuple[ActionEvidenceResolution, tuple[str, ...]]:
    raw_category, original_intent = classify_action(action)
    action_id = str(action.get("action_id") or "anonymous_action")
    priority = str(action.get("priority") or "P2").upper()
    original = _unique_strings(_string_values(action.get("evidence_refs")))
    related = _unique_strings([
        *_string_values(action.get("related_hypothesis_ids")),
        *_string_values(action.get("hypothesis_ids")),
    ])
    hypothesis_categories = _related_hypothesis_categories(related, research, eda)
    original_alignment, alignment_reason = _action_hypothesis_alignment(
        original_intent, hypothesis_categories
    )
    matching_failed_critical_refs = _matching_leakage_check_refs(
        action,
        eda,
        related,
        original,
        statuses={"failed"},
        severities={"high", "critical"},
    )
    matching_warning_refs = _matching_leakage_check_refs(
        action,
        eda,
        related,
        original,
        statuses={"warning"},
        severities={"medium", "high", "critical"},
    )
    matching_not_testable_refs = _matching_leakage_check_refs(
        action,
        eda,
        related,
        original,
        statuses={"not_testable"},
        severities={"low", "medium", "high", "critical"},
    )
    category, intent, reclassification_reason = _repair_action_intent(
        action,
        original_category=raw_category,
        original_intent=original_intent,
        hypothesis_categories=hypothesis_categories,
        matching_failed_critical_refs=matching_failed_critical_refs,
        matching_warning_refs=matching_warning_refs,
        matching_not_testable_refs=matching_not_testable_refs,
    )
    removed_hypothesis_ids = _remove_incompatible_hypothesis_links(
        related, intent, research, eda
    )
    if removed_hypothesis_ids and reclassification_reason is None:
        reclassification_reason = (
            "Removed related hypotheses whose categories are incompatible with "
            f"{intent}: {', '.join(removed_hypothesis_ids)}."
        )
    inference_signals = _intent_inference_signals(
        action,
        original_intent=original_intent,
        final_intent=intent,
        hypothesis_categories=hypothesis_categories,
        matching_failed_critical_refs=matching_failed_critical_refs,
    )
    policy_violations: list[str] = []
    if original_alignment == "invalid":
        policy_violations.append("intent_hypothesis_category_mismatch")
    if (
        original_intent == "leakage.block_critical_issue"
        and not matching_failed_critical_refs
    ):
        policy_violations.append("critical_leakage_action_without_failed_check")
    attempts: list[str] = ["original"]
    if hypothesis_categories:
        attempts.append("intent_hypothesis_alignment")
    if reclassification_reason:
        attempts.append("deterministic_intent_repair")
    unresolved: list[str] = []
    unsupported: list[str] = []
    contradictory: list[str] = []
    candidates: list[str] = []
    resolved: list[str] = []
    added: list[str] = []
    notes: list[str] = []
    matched_actions: list[str] = []

    contradiction_note = _contradiction_reason(action, intent, eda, research, related)
    if contradiction_note:
        contradictory.extend(original)
        notes.append(contradiction_note)

    for ref in original:
        if not catalog.is_valid_evidence_ref(ref):
            unresolved.append(ref)
        elif not _reference_supports_intent(ref, category, intent):
            unsupported.append(ref)
        elif (
            intent == "leakage.block_critical_issue"
            and ref not in matching_failed_critical_refs
        ):
            unsupported.append(ref)
        elif not contradiction_note:
            resolved.append(ref)

    results = _hypothesis_results_by_id(eda)
    needs_resolution = bool(not resolved or unresolved or unsupported or contradiction_note)
    if related and needs_resolution:
        attempts.append("hypothesis_link")
    for hypothesis_id in list(related) if needs_resolution else []:
        result = results.get(hypothesis_id)
        if result is None:
            continue
        status = str(result.get("status") or "")
        if status == "rejected":
            if _action_matches_rejected_hypothesis(action, hypothesis_id, research, intent, eda):
                semantic_ref = f"hypothesis_results.{hypothesis_id}"
                if catalog.is_valid_evidence_ref(semantic_ref):
                    _append_unique(contradictory, semantic_ref)
                notes.append(
                    f"Rejected hypothesis {hypothesis_id!r} cannot support the proposed action."
                )
            continue
        if status not in {"confirmed", "partially_confirmed"}:
            continue
        for ref in _string_values(result.get("evidence_refs")):
            _append_candidate(
                ref, candidates, resolved, added, catalog, category, intent,
                blocked=bool(contradiction_note),
            )

    recommended = (
        _matching_recommended_actions(action, category, intent, eda)
        if needs_resolution
        else []
    )
    if recommended:
        attempts.append("recommended_action_match")
    if len({item[2] for item in recommended}) > 1:
        notes.append("Multiple incompatible recommended actions matched; no recommendation was inherited.")
        recommended = []
    for index, item, _ in recommended:
        matched_actions.append(f"recommended_next_actions[{index}]")
        for ref in _string_values(item.get("evidence_refs")):
            _append_candidate(
                ref, candidates, resolved, added, catalog, category, intent,
                blocked=bool(contradiction_note),
            )
        for hypothesis_id in _hypotheses_for_recommendation(item, results):
            candidate_categories = _related_hypothesis_categories(
                [hypothesis_id], research, eda
            )
            policy = ACTION_INTENT_POLICIES.get(intent)
            if (
                policy is not None
                and candidate_categories
                and not set(candidate_categories).issubset(
                    policy.allowed_hypothesis_categories
                )
            ):
                continue
            _append_unique(related, hypothesis_id)

    structured = (
        _structured_candidate_refs(
            category,
            intent,
            action,
            eda,
            matching_failed_critical_refs=matching_failed_critical_refs,
            matching_warning_refs=matching_warning_refs,
        )
        if needs_resolution
        else []
    )
    if structured:
        attempts.append("category_fallback")
    for ref in structured:
        _append_candidate(
            ref, candidates, resolved, added, catalog, category, intent,
            blocked=bool(contradiction_note),
        )

    leakage_permitted_refs: set[str] | None = None
    if intent == "leakage.block_critical_issue":
        leakage_permitted_refs = set(matching_failed_critical_refs)
    elif intent == "leakage.audit_warning":
        leakage_permitted_refs = set(matching_warning_refs)
    elif intent == "leakage.run_required_checks":
        leakage_permitted_refs = set(matching_not_testable_refs)
    if leakage_permitted_refs is not None:
        for ref in list(resolved):
            if ref not in leakage_permitted_refs:
                _append_unique(unsupported, ref)
        resolved[:] = [ref for ref in resolved if ref in leakage_permitted_refs]
        added[:] = [ref for ref in added if ref in leakage_permitted_refs]
        candidates[:] = [ref for ref in candidates if ref in leakage_permitted_refs]

    if not related:
        related.extend(_category_hypothesis_ids(category, research, eda))

    if contradiction_note or contradictory:
        status: ActionSupportStatus = "contradicted"
    elif resolved and (unresolved or unsupported):
        status = "partially_supported"
    elif resolved:
        status = "supported"
    else:
        status = "unsupported"

    if reclassification_reason:
        method: ActionResolutionMethod = "deterministic_fallback"
    elif status == "supported" and not added:
        method = "original"
    elif matched_actions and added:
        method = "recommended_action_match"
    elif "hypothesis_link" in attempts and added:
        method = "hypothesis_link"
    elif added:
        method = "category_fallback"
    else:
        method = "none"
    final_hypothesis_categories = _related_hypothesis_categories(
        related, research, eda
    )
    report_alignment = original_alignment
    report_alignment_reason = alignment_reason
    if report_alignment == "not_applicable" and final_hypothesis_categories:
        report_alignment, report_alignment_reason = _action_hypothesis_alignment(
            intent, final_hypothesis_categories
        )
    return ActionEvidenceResolution(
        action_id=action_id,
        raw_action_category=raw_category,
        category=category,
        original_intent=original_intent,
        intent=intent,
        intent_inference_signals=tuple(inference_signals),
        priority=priority,
        original_priority=priority,
        normalized_priority=priority,
        original_mandatory_status=bool(
            action.get("mandatory")
            or (priority == "P0" and original_intent == "leakage.block_critical_issue")
        ),
        normalized_mandatory_status=bool(
            priority == "P0"
            and intent == "leakage.block_critical_issue"
            and matching_failed_critical_refs
            and resolved
        ),
        original_refs=tuple(original),
        resolved_refs=tuple(resolved),
        added_refs=tuple(added),
        unresolved_refs=tuple(unresolved),
        unsupported_refs=tuple(unsupported),
        contradictory_refs=tuple(contradictory),
        candidate_refs=tuple(candidates),
        related_hypothesis_ids=tuple(related),
        related_hypothesis_categories=tuple(final_hypothesis_categories),
        matching_recommended_action_ids=tuple(matched_actions),
        matching_leakage_check_refs=tuple(_unique_strings([
            *matching_failed_critical_refs,
            *matching_warning_refs,
            *matching_not_testable_refs,
        ])),
        matching_failed_critical_leakage_refs=tuple(matching_failed_critical_refs),
        failed_critical_leakage_count=len(matching_failed_critical_refs),
        alignment_status=report_alignment,
        alignment_reason=report_alignment_reason,
        reclassification_reason=reclassification_reason,
        policy_violation_codes=tuple(policy_violations),
        support_status=status,
        resolution_method=method,
        resolution_attempts=tuple(attempts),
        notes=tuple(_unique_strings(notes)),
    ), tuple(related)


def _append_candidate(
    ref: str,
    candidates: list[str],
    resolved: list[str],
    added: list[str],
    catalog: ReferenceCatalog,
    category: str,
    intent: str,
    *,
    blocked: bool,
) -> None:
    if not ref or not catalog.is_valid_evidence_ref(ref):
        return
    if not _reference_supports_intent(ref, category, intent):
        return
    _append_unique(candidates, ref)
    if blocked:
        return
    if ref not in resolved:
        resolved.append(ref)
        added.append(ref)


def _reference_supports_intent(ref: str, category: str, intent: str) -> bool:
    root = ref.split(".", 1)[0]
    if root not in ACTION_ALLOWED_EVIDENCE_ROOTS.get(category, frozenset()):
        return False
    if intent == "validation.use_primary_policy":
        return ref.startswith((
            "validation_evidence.primary_validation",
            "validation_evidence.reasoning_summary",
            "validation_evidence.recommended_validation",
            "hypothesis_results.",
            "recommended_next_actions.",
        ))
    if intent == "validation.freeze_folds":
        return ref.startswith((
            "validation_evidence.primary_validation",
            "validation_evidence.temporal_folds",
            "validation_evidence.group_columns",
            "validation_evidence.time_columns",
            "hypothesis_results.",
        ))
    if intent == "metric.use_required_prediction_type":
        return ref.startswith((
            "metric_evidence.requires_probabilities",
            "metric_evidence.rank_based",
            "metric_evidence.prediction_output_type",
            "metric_evidence.requires_threshold",
            "metric_evidence.threshold_search_needed",
            "hypothesis_results.",
        ))
    if intent == "leakage.block_critical_issue":
        # Critical remediation is supported only by one concrete leakage check.
        # A section root, narrative hypothesis, or safety policy cannot establish
        # that a high/critical check actually failed.
        return ref.startswith("leakage_evidence.")
    if intent in {"leakage.audit_warning", "leakage.run_required_checks"}:
        return ref.startswith("leakage_evidence.")
    if intent == "data.verify_submission_contract":
        return ref.startswith((
            "inferred_schema.sample_submission",
            "inferred_schema.submission",
            "file_inventory",
            "metric_evidence.required_columns",
            "hypothesis_results.",
        ))
    return True


def _structured_candidate_refs(
    category: str,
    intent: str,
    action: Mapping[str, Any],
    eda: dict[str, Any],
    *,
    matching_failed_critical_refs: list[str],
    matching_warning_refs: list[str],
) -> list[str]:
    refs: list[str] = []
    validation = _dict_value(eda.get("validation_evidence"))
    metric = _dict_value(eda.get("metric_evidence"))
    if intent in {"validation.use_primary_policy", "validation.freeze_folds"}:
        if _dict_value(validation.get("primary_validation")):
            refs.append("validation_evidence.primary_validation")
        if validation.get("reasoning_summary"):
            refs.append("validation_evidence.reasoning_summary")
        if intent == "validation.freeze_folds":
            for field in ("temporal_folds", "group_columns", "time_columns"):
                if validation.get(field):
                    refs.append(f"validation_evidence.{field}")
    elif intent == "metric.use_required_prediction_type":
        for field in (
            "requires_probabilities", "rank_based", "prediction_output_type",
            "requires_threshold", "threshold_search_needed",
        ):
            if metric.get(field) not in (None, False, ""):
                refs.append(f"metric_evidence.{field}")
    elif intent == "leakage.block_critical_issue":
        refs.extend(matching_failed_critical_refs)
    elif intent == "leakage.audit_warning":
        refs.extend(matching_warning_refs)
    elif intent == "leakage.run_required_checks":
        for item in _leakage_checks(eda):
            if str(item.get("status") or "").lower() == "not_testable":
                refs.append(f"leakage_evidence.{item['check_id']}")
    elif category == "drift" and eda.get("drift_evidence"):
        refs.append("drift_evidence")
    elif category == "baseline" and eda.get("baseline_evidence"):
        refs.append("baseline_evidence")
    elif category == "relationship" and eda.get("relationship_evidence"):
        refs.append("relationship_evidence")
    elif category == "feature" and eda.get("feature_probe_evidence"):
        refs.append("feature_probe_evidence")
    return refs


def _apply_canonical_fallback(
    action: MutableMapping[str, Any], intent: str, eda: dict[str, Any]
) -> bool:
    validation = _dict_value(eda.get("validation_evidence"))
    primary = _dict_value(validation.get("primary_validation"))
    metric = _dict_value(eda.get("metric_evidence"))
    if intent == "validation.use_primary_policy" and primary.get("method"):
        method = str(primary["method"])
        action.update({
            "action": (
                f"Use {method} as the primary validation policy and persist the fold "
                "assignments before training."
            ),
            "reason": "Structured EDA evidence selected this primary validation policy.",
            "evidence_refs": ["validation_evidence.primary_validation"],
            "validation_strategy": method,
            "confidence": "high",
            "evidence_origin": "EDA-confirmed",
        })
        return True
    if intent == "metric.use_required_prediction_type":
        if metric.get("requires_probabilities"):
            action.update({
                "action": "Generate probability predictions, not hard labels.",
                "reason": "The metric contract requires probabilistic predictions.",
                "evidence_refs": ["metric_evidence.requires_probabilities"],
                "confidence": "high",
                "evidence_origin": "EDA-confirmed",
            })
            return True
        if metric.get("rank_based"):
            action.update({
                "action": "Generate continuous scores suitable for rank-based evaluation.",
                "reason": "The metric contract is rank based.",
                "evidence_refs": ["metric_evidence.rank_based"],
                "confidence": "high",
                "evidence_origin": "EDA-confirmed",
            })
            return True
    if intent == "leakage.block_critical_issue":
        checks = _critical_leakage_checks(eda)
        if len(checks) == 1:
            check = checks[0]
            check_id = str(check["check_id"])
            action.update({
                "action": f"Block or remove the unsafe source identified by leakage check {check_id}.",
                "reason": str(check.get("finding") or "A critical leakage check failed."),
                "evidence_refs": [f"leakage_evidence.{check_id}"],
                "confidence": "high",
                "evidence_origin": "Safety-warning",
            })
            return True
    return False


def _required_fallback_actions(
    eda: dict[str, Any], research: dict[str, Any], existing: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], str]]:
    intents = {classify_action(action)[1] for action in existing}
    result: list[tuple[dict[str, Any], str]] = []
    validation = _dict_value(eda.get("validation_evidence"))
    primary = _dict_value(validation.get("primary_validation"))
    if primary.get("method") and "validation.use_primary_policy" not in intents:
        method = str(primary["method"])
        result.append(({
            "action_id": "eda_validation_primary",
            "priority": "P0",
            "action": (
                f"Use {method} as the primary validation policy and persist the fold "
                "assignments before training."
            ),
            "reason": "Structured EDA evidence selected this primary validation policy.",
            "evidence_refs": ["validation_evidence.primary_validation"],
            "related_hypothesis_ids": _category_hypothesis_ids("validation", research, eda),
            "validation_strategy": method,
            "confidence": "high",
            "evidence_origin": "EDA-confirmed",
        }, "metric_and_validation"))

    metric = _dict_value(eda.get("metric_evidence"))
    if (
        "metric.use_required_prediction_type" not in intents
        and (metric.get("requires_probabilities") or metric.get("rank_based"))
    ):
        probability = bool(metric.get("requires_probabilities"))
        ref = "metric_evidence.requires_probabilities" if probability else "metric_evidence.rank_based"
        result.append(({
            "action_id": "eda_metric_prediction_type",
            "priority": "P0",
            "action": (
                "Generate probability predictions, not hard labels."
                if probability
                else "Generate continuous scores suitable for rank-based evaluation."
            ),
            "reason": "Structured metric evidence defines the required prediction output.",
            "evidence_refs": [ref],
            "related_hypothesis_ids": _category_hypothesis_ids("metric", research, eda),
            "confidence": "high",
            "evidence_origin": "EDA-confirmed",
        }, "metric_and_validation"))

    if "leakage.block_critical_issue" not in intents:
        for check in _critical_leakage_checks(eda):
            check_id = str(check["check_id"])
            result.append(({
                "action_id": f"eda_leakage_block_{_safe_id(check_id)}",
                "priority": "P0",
                "action": f"Block or remove the unsafe source identified by leakage check {check_id}.",
                "reason": str(check.get("finding") or "A critical leakage check failed."),
                "evidence_refs": [f"leakage_evidence.{check_id}"],
                "related_hypothesis_ids": _category_hypothesis_ids("leakage", research, eda),
                "confidence": "high",
                "evidence_origin": "Safety-warning",
            }, "leakage_and_data_quality"))
    return result


def _attach_action_to_section(
    payload: MutableMapping[str, Any], action: Mapping[str, Any], section_id: str
) -> None:
    sections = payload.setdefault("sections", [])
    if not isinstance(sections, list):
        sections = []
        payload["sections"] = sections
    section = next(
        (item for item in sections if isinstance(item, dict) and item.get("section_id") == section_id),
        None,
    )
    if section is None:
        section = {
            "section_id": section_id,
            "title": section_id.replace("_", " ").title(),
            "summary": "Mandatory actions grounded in structured EDA evidence.",
            "action_ids": [],
            "evidence_refs": [],
            "related_hypothesis_ids": [],
        }
        sections.append(section)
    for field, values in (
        ("action_ids", [action["action_id"]]),
        ("evidence_refs", _string_values(action.get("evidence_refs"))),
        ("related_hypothesis_ids", _string_values(action.get("related_hypothesis_ids"))),
    ):
        current = _string_values(section.get(field))
        for value in values:
            _append_unique(current, value)
        section[field] = current


def _matching_recommended_actions(
    action: Mapping[str, Any], category: str, intent: str, eda: dict[str, Any]
) -> list[tuple[int, dict[str, Any], str]]:
    action_tokens = _intent_tokens(_action_text(action))
    matches: list[tuple[int, dict[str, Any], str]] = []
    values = eda.get("recommended_next_actions") or []
    if not isinstance(values, list):
        return matches
    for index, raw in enumerate(values):
        item = _dict_value(raw)
        if not item:
            continue
        item_category, item_intent = classify_action(item)
        if item_intent == intent:
            matches.append((index, item, item_intent))
            continue
        if item_category != category:
            continue
        item_tokens = _intent_tokens(_action_text(item))
        overlap = len(action_tokens & item_tokens)
        denominator = max(1, min(len(action_tokens), len(item_tokens)))
        if overlap / denominator >= 0.5:
            matches.append((index, item, item_intent))
    return matches


def _hypotheses_for_recommendation(
    action: Mapping[str, Any], results: Mapping[str, dict[str, Any]]
) -> list[str]:
    refs = set(_string_values(action.get("evidence_refs")))
    categories = set(_string_values(action.get("source_categories")))
    related: list[str] = []
    for hypothesis_id, result in results.items():
        if str(result.get("status") or "") not in {"confirmed", "partially_confirmed"}:
            continue
        result_refs = set(_string_values(result.get("evidence_refs")))
        if refs & result_refs or str(result.get("category") or "") in categories:
            related.append(hypothesis_id)
    return related


def _contradiction_reason(
    action: Mapping[str, Any],
    intent: str,
    eda: dict[str, Any],
    research: dict[str, Any],
    related: list[str],
) -> str | None:
    if intent.startswith("validation."):
        primary = _dict_value(
            _dict_value(eda.get("validation_evidence")).get("primary_validation")
        )
        primary_method = str(primary.get("method") or "").lower()
        proposed = _proposed_validation_method(action)
        if proposed and primary_method and proposed != primary_method:
            return (
                f"Action proposes {proposed!r}, but EDA selected primary validation "
                f"{primary_method!r}."
            )
    for hypothesis_id in related:
        result = _hypothesis_results_by_id(eda).get(hypothesis_id)
        if result and str(result.get("status") or "") == "rejected":
            if _action_matches_rejected_hypothesis(
                action, hypothesis_id, research, intent, eda
            ):
                return f"Action agrees with rejected hypothesis {hypothesis_id!r}."
    return None


def _action_matches_rejected_hypothesis(
    action: Mapping[str, Any],
    hypothesis_id: str,
    research: dict[str, Any],
    intent: str,
    eda: dict[str, Any],
) -> bool:
    if intent.startswith("validation."):
        proposed = _proposed_validation_method(action)
        primary = str(
            _dict_value(_dict_value(eda.get("validation_evidence")).get("primary_validation")).get("method")
            or ""
        ).lower()
        if proposed and primary and proposed != primary:
            return True
        if proposed and primary and proposed == primary:
            return False
    action_tokens = _intent_tokens(_action_text(action))
    for item in research.get("hypotheses") or []:
        hypothesis = _dict_value(item)
        if str(hypothesis.get("hypothesis_id") or "") != hypothesis_id:
            continue
        claim_tokens = _intent_tokens(
            " ".join(str(hypothesis.get(key) or "") for key in ("claim", "statement", "rationale"))
        )
        return len(action_tokens & claim_tokens) >= 2
    return False


def _related_hypothesis_categories(
    related: list[str], research: dict[str, Any], eda: dict[str, Any]
) -> list[str]:
    categories_by_id: dict[str, str] = {}
    for collection in (
        research.get("hypotheses") or [],
        eda.get("hypothesis_results") or [],
        eda.get("testable_hypotheses") or [],
    ):
        for raw in collection:
            item = _dict_value(raw)
            hypothesis_id = str(item.get("hypothesis_id") or item.get("id") or "")
            category = _canonical_category(str(item.get("category") or ""))
            if not category:
                category = _category_from_prefix(hypothesis_id.lower())
            if hypothesis_id and category:
                categories_by_id[hypothesis_id] = category
    return _unique_strings(
        categories_by_id[hypothesis_id]
        for hypothesis_id in related
        if hypothesis_id in categories_by_id
    )


def _action_hypothesis_alignment(
    intent: str, hypothesis_categories: list[str]
) -> tuple[Literal["valid", "invalid", "not_applicable"], str | None]:
    if not hypothesis_categories:
        return "not_applicable", None
    policy = ACTION_INTENT_POLICIES.get(intent)
    if policy is None:
        return "not_applicable", None
    incompatible = sorted(
        set(hypothesis_categories) - set(policy.allowed_hypothesis_categories)
    )
    if incompatible:
        return (
            "invalid",
            "intent_hypothesis_category_mismatch: "
            f"{intent} does not allow {', '.join(incompatible)} hypotheses",
        )
    return "valid", None


def _remove_incompatible_hypothesis_links(
    related: list[str],
    intent: str,
    research: dict[str, Any],
    eda: dict[str, Any],
) -> list[str]:
    policy = ACTION_INTENT_POLICIES.get(intent)
    if policy is None:
        return []
    kept: list[str] = []
    removed: list[str] = []
    for hypothesis_id in related:
        categories = _related_hypothesis_categories([hypothesis_id], research, eda)
        if categories and not set(categories).issubset(
            policy.allowed_hypothesis_categories
        ):
            removed.append(hypothesis_id)
        else:
            kept.append(hypothesis_id)
    related[:] = kept
    return removed


def _repair_action_intent(
    action: Mapping[str, Any],
    *,
    original_category: str,
    original_intent: str,
    hypothesis_categories: list[str],
    matching_failed_critical_refs: list[str],
    matching_warning_refs: list[str],
    matching_not_testable_refs: list[str],
) -> tuple[str, str, str | None]:
    """Repair only semantically recoverable intent mistakes before evidence lookup."""

    text = _action_text(action)
    baseline_semantics = _has_any(
        text, ("baseline", "sanity floor", "accuracy floor", "benchmark")
    )
    if original_intent == "leakage.block_critical_issue":
        if matching_failed_critical_refs and not (
            set(hypothesis_categories) - {"leakage", "data_quality"}
        ):
            return original_category, original_intent, None
        if baseline_semantics or (
            hypothesis_categories and set(hypothesis_categories) == {"baseline"}
        ):
            return (
                "baseline",
                "baseline.run_sanity_baseline",
                "Critical leakage intent was reclassified from baseline semantics and "
                "related baseline hypotheses; no matching failed high/critical check exists.",
            )
        if matching_warning_refs:
            return (
                "leakage",
                "leakage.audit_warning",
                "Warning-only leakage evidence permits a diagnostic audit, not confirmed "
                "critical remediation.",
            )
        return (
            "leakage",
            "leakage.run_required_checks",
            "Critical leakage intent lacks a matching failed high/critical check and was "
            "converted to a diagnostic prerequisite.",
        )

    if (
        original_intent == "leakage.audit_warning"
        and not matching_warning_refs
        and matching_not_testable_refs
    ):
        return (
            "leakage",
            "leakage.run_required_checks",
            "No warning check matched; concrete not-testable checks support only a "
            "diagnostic prerequisite.",
        )

    alignment, _ = _action_hypothesis_alignment(
        original_intent, hypothesis_categories
    )
    if alignment == "invalid" and set(hypothesis_categories) == {"baseline"}:
        return (
            "baseline",
            "baseline.run_sanity_baseline",
            "Intent was aligned to the sole related baseline hypothesis category.",
        )
    return original_category, original_intent, None


def _intent_inference_signals(
    action: Mapping[str, Any],
    *,
    original_intent: str,
    final_intent: str,
    hypothesis_categories: list[str],
    matching_failed_critical_refs: list[str],
) -> list[str]:
    signals: list[str] = []
    if action.get("intent") or action.get("action_intent"):
        signals.append("explicit_intent")
    else:
        signals.append("action_semantics")
    if hypothesis_categories:
        signals.append("related_hypothesis_categories=" + ",".join(hypothesis_categories))
    if matching_failed_critical_refs:
        signals.append("matching_failed_critical_leakage_check")
    if final_intent != original_intent:
        signals.append("deterministic_intent_repair")
    return signals


def _leakage_checks(eda: dict[str, Any]) -> list[dict[str, Any]]:
    values = eda.get("leakage_evidence") or []
    if not isinstance(values, list):
        return []
    return [
        item
        for raw in values
        if (item := _dict_value(raw)) and item.get("check_id")
    ]


def _matching_leakage_check_refs(
    action: Mapping[str, Any],
    eda: dict[str, Any],
    related: list[str],
    original_refs: list[str],
    *,
    statuses: set[str],
    severities: set[str],
) -> list[str]:
    """Return only concrete checks that are both eligible and action-linked."""

    action_tokens = _intent_tokens(_action_text(action))
    result_refs = {
        ref
        for hypothesis_id in related
        for ref in _string_values(
            _hypothesis_results_by_id(eda).get(hypothesis_id, {}).get("evidence_refs")
        )
    }
    related_narrative = " ".join(
        " ".join([
            str(result.get("finding") or ""),
            str(result.get("impact_on_strategy") or ""),
            " ".join(_string_values(result.get("limitations"))),
        ])
        for hypothesis_id in related
        if (result := _hypothesis_results_by_id(eda).get(hypothesis_id))
    ).lower().replace("_", " ")
    refs: list[str] = []
    for item in _leakage_checks(eda):
        status = str(item.get("status") or "").lower()
        severity = str(item.get("severity") or "").lower()
        if status not in statuses or severity not in severities:
            continue
        check_id = str(item["check_id"])
        ref = f"leakage_evidence.{check_id}"
        check_tokens = _intent_tokens(" ".join([
            check_id.replace("_", " "),
            str(item.get("finding") or ""),
            " ".join(_leaf_strings(item.get("evidence"))),
        ]))
        explicit_link = ref in original_refs or ref in result_refs
        semantic_overlap = action_tokens & check_tokens
        # Generic words such as target/check/leakage alone are too weak. Exact
        # check IDs or a concrete subject token establish the required link.
        exact_check = check_id.replace("_", " ") in _action_text(action)
        hypothesis_link = check_id.replace("_", " ") in related_narrative
        concrete_overlap = {
            token for token in semantic_overlap
            if token not in {"check", "leakage", "target", "feature", "source"}
        }
        if explicit_link or exact_check or hypothesis_link or concrete_overlap:
            refs.append(ref)
    return _unique_strings(refs)


def _leaf_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _leaf_strings(nested)]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _leaf_strings(nested)]
    if value in (None, ""):
        return []
    return [str(value)]


def _critical_leakage_checks(eda: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _leakage_checks(eda):
        if (
            item.get("check_id")
            and str(item.get("status") or "").lower() == "failed"
            and str(item.get("severity") or "").lower() in {"high", "critical"}
        ):
            result.append(item)
    return result


def _category_hypothesis_ids(
    category: str, research: dict[str, Any], eda: dict[str, Any]
) -> list[str]:
    result: list[str] = []
    for collection in (
        research.get("hypotheses") or [],
        eda.get("hypothesis_results") or [],
        eda.get("testable_hypotheses") or [],
    ):
        for raw in collection:
            item = _dict_value(raw)
            hypothesis_id = str(item.get("hypothesis_id") or item.get("id") or "")
            item_category = _canonical_category(str(item.get("category") or ""))
            if not item_category:
                item_category = _category_from_prefix(hypothesis_id.lower())
            if hypothesis_id and item_category == category:
                _append_unique(result, hypothesis_id)
    return result


def _hypothesis_results_by_id(eda: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in eda.get("hypothesis_results") or []:
        item = _dict_value(raw)
        hypothesis_id = str(item.get("hypothesis_id") or "")
        if hypothesis_id and hypothesis_id not in result:
            result[hypothesis_id] = item
    return result


def _proposed_validation_method(action: Mapping[str, Any]) -> str | None:
    explicit = str(action.get("validation_strategy") or "").strip().lower()
    if explicit:
        return explicit
    text = _action_text(action)
    for method, patterns in _METHOD_PATTERNS:
        if _has_any(text, patterns):
            return method
    return None


def _category_from_prefix(action_id: str) -> str:
    normalized = re.sub(r"^action[_-]?", "", action_id)
    prefix = normalized.split("_", 1)[0].split("-", 1)[0]
    return {
        "val": "validation", "validation": "validation", "metric": "metric",
        "leak": "leakage", "leakage": "leakage", "schema": "data_quality",
        "data": "data_quality", "quality": "data_quality", "rel": "relationship",
        "relationship": "relationship", "drift": "drift", "baseline": "baseline",
        "feature": "feature", "model": "modeling", "submission": "submission",
    }.get(prefix, "")


def _category_from_text(text: str) -> str:
    rules = (
        ("submission", ("sample submission", "submission contract", "submission column")),
        ("leakage", ("leak", "unsafe source", "target in test")),
        ("metric", ("metric", "probabil", "hard label", "rank based", "threshold")),
        ("baseline", ("baseline", "sanity floor", "ablation")),
        ("feature", ("feature", "encoding", "missingness")),
        ("relationship", ("join", "relationship", "aggregate before")),
        ("drift", ("drift", "leaderboard", "adversarial validation")),
        ("validation", ("validation", "fold", "split", "kfold", "cross validation")),
        ("data_quality", ("schema", "data quality", "primary id", "identifier")),
    )
    return next((category for category, markers in rules if _has_any(text, markers)), "")


def _canonical_category(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "val": "validation", "validation_strategy": "validation",
        "threshold_tuning": "metric", "target": "metric",
        "do_not_do": "leakage", "primary_id_safety": "data_quality",
        "role_exclusion": "data_quality", "schema": "data_quality",
        "data": "data_quality", "drift_and_leaderboard": "drift",
        "feature_engineering": "feature", "feature_probe": "feature",
        "first_experiments": "modeling",
    }.get(normalized, normalized if normalized in ACTION_ALLOWED_EVIDENCE_ROOTS else "")


def _action_text(action: Mapping[str, Any]) -> str:
    return " ".join(
        str(action.get(key) or "")
        for key in ("action", "reason", "why", "category", "applies_to", "source_categories")
    ).lower().replace("_", " ")


def _intent_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)?", text.lower()))
    normalized = {_TOKEN_ALIASES.get(token, token) for token in tokens}
    return {token for token in normalized if token not in _STOP_TOKENS and len(token) > 1}


def _model_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return deepcopy(dict(dumped)) if isinstance(dumped, Mapping) else {}
    return {}


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError("action must be a mapping or Pydantic model")


def _dict_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return {}


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if isinstance(item, str) and item.strip()]
    return []


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            _append_unique(result, value)
    return result


def _unique_tuple(values: Any) -> tuple[str, ...]:
    return tuple(_unique_strings(values))


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "issue"


__all__ = [
    "ACTION_ALLOWED_EVIDENCE_ROOTS",
    "ActionEvidenceResolution",
    "ActionResolutionMethod",
    "ActionSupportStatus",
    "EvidenceReferenceIndex",
    "FinalStrategyActionEvidenceReport",
    "classify_action",
    "resolve_action_evidence_refs",
    "resolve_final_strategy_action_evidence",
]
