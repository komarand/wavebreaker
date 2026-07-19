from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field, StringConstraints, ValidationError

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.final_strategy import (
    Confidence,
    EvidenceOrigin,
    FinalValidationMethod,
    Priority,
)
from kaggle_researcher.contracts.ids import ExperimentId, HypothesisId
from kaggle_researcher.contracts.reference_catalog import (
    ReferenceCatalog,
    ReferenceNamespace,
)


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_PROMPT_CONTEXT_EVIDENCE_ALIASES = {
    # The 2026-07-19 prompt exposed this bounded context label while the canonical
    # EDA registry owns the concrete semantic item below.
    "feature_probe_evidence.unsafe_feature_probes": (
        "feature_probe_evidence.naive_target_encoding_or_woe"
    ),
}


class FinalStrategySupportRef(ContractModel):
    namespace: ReferenceNamespace
    ref_id: NonEmptyString


class FinalStrategyActionDraft(ContractModel):
    action_id: NonEmptyString | None = None
    priority: Priority
    action: NonEmptyString
    reason: NonEmptyString
    support_refs: list[FinalStrategySupportRef] = Field(min_length=1)
    related_hypothesis_ids: list[HypothesisId] = Field(default_factory=list)
    experiment_ids: list[ExperimentId] = Field(default_factory=list)
    source_claim: NonEmptyString | None = None
    validation_strategy: FinalValidationMethod | None = None
    confidence: Confidence = "medium"
    evidence_origin: EvidenceOrigin = "Hypothesis-to-test"
    limitations: list[NonEmptyString] = Field(default_factory=list)


class FinalStrategySectionDraft(ContractModel):
    section_id: NonEmptyString
    title: NonEmptyString
    narrative: NonEmptyString
    actions: list[FinalStrategyActionDraft] = Field(default_factory=list)
    evidence_summary_refs: list[FinalStrategySupportRef] = Field(default_factory=list)
    limitations: list[NonEmptyString] = Field(default_factory=list)


class FinalStrategyDraft(ContractModel):
    schema_version: NonEmptyString = "1.0"
    competition_id: NonEmptyString
    executive_summary: NonEmptyString | None = None
    sections: list[FinalStrategySectionDraft]
    actions: list[FinalStrategyActionDraft] = Field(default_factory=list)
    warnings: list[NonEmptyString] = Field(default_factory=list)
    limitations: list[NonEmptyString] = Field(default_factory=list)


class FinalStrategyDraftReferenceIssue(ContractModel):
    field_path: NonEmptyString
    ref_id: NonEmptyString
    expected_namespace: ReferenceNamespace | None = None
    resolution_status: NonEmptyString
    message: NonEmptyString


class FinalStrategyDraftReferenceError(ValueError):
    def __init__(self, issues: list[FinalStrategyDraftReferenceIssue]) -> None:
        self.stage = "final_strategy_draft_normalization"
        self.contract = "final_strategy_draft_references"
        self.issues = tuple(issues)
        self.field_paths = tuple(issue.field_path for issue in issues)
        self.invalid_ids = tuple(issue.ref_id for issue in issues)
        details = "; ".join(
            f"{issue.field_path} contains {issue.ref_id!r}: {issue.message}"
            for issue in issues[:8]
        )
        super().__init__(f"Final Strategy draft reference normalization failed: {details}")


def normalize_legacy_final_strategy_to_draft(
    raw_response: Mapping[str, Any] | FinalStrategyDraft,
    catalog: ReferenceCatalog,
) -> FinalStrategyDraft:
    """Normalize legacy reference fields into catalog-resolved typed support refs."""

    if isinstance(raw_response, FinalStrategyDraft):
        raw: Mapping[str, Any] = raw_response.model_dump(mode="json")
    else:
        raw = raw_response

    sections = [
        _normalize_section(value, catalog, f"sections[{index}]")
        for index, value in enumerate(_mapping_list(raw.get("sections")))
    ]
    actions = [
        _normalize_action(value, catalog, f"actions[{index}]")
        for index, value in enumerate(_mapping_list(raw.get("actions")))
    ]
    payload = {
        "schema_version": raw.get("schema_version") or "1.0",
        "competition_id": raw.get("competition_id"),
        "executive_summary": raw.get("executive_summary"),
        "sections": sections,
        "actions": actions,
        "warnings": _unique_strings(raw.get("warnings")),
        "limitations": _unique_strings(raw.get("limitations")),
    }
    return FinalStrategyDraft.model_validate(payload)


def compile_final_strategy_draft(
    raw_response: Mapping[str, Any] | FinalStrategyDraft,
    catalog: ReferenceCatalog,
) -> dict[str, Any]:
    """Compile the LLM-facing draft into the strict canonical strategy shape.

    The LLM contract intentionally uses narrative/actions/support_refs. Those names
    are never passed through to FinalStrategyResult, whose strict extra-field policy
    remains unchanged.
    """

    draft = normalize_legacy_final_strategy_to_draft(raw_response, catalog)
    sections = []
    for section in draft.sections:
        compiled_actions = [_compile_action(action) for action in section.actions]
        section_refs = _split_support_refs(section.evidence_summary_refs)
        sections.append({
            "section_id": section.section_id,
            "title": section.title,
            "summary": section.narrative,
            "actions": compiled_actions,
            "evidence_refs": section_refs["evidence_refs"],
            "related_hypothesis_ids": section_refs["related_hypothesis_ids"],
            "source_refs": section_refs["source_refs"],
            "limitations": list(section.limitations),
        })
    if draft.executive_summary:
        executive = next(
            (item for item in sections if item["section_id"] == "executive_summary"),
            None,
        )
        if executive is not None:
            executive["summary"] = draft.executive_summary
    return {
        "schema_version": "2.0",
        "competition_id": draft.competition_id,
        "sections": sections,
        "actions": [_compile_action(action) for action in draft.actions],
        "limitations": list(dict.fromkeys([
            *draft.limitations,
            *(f"LLM warning: {warning}" for warning in draft.warnings),
        ])),
    }


def _normalize_section(
    raw: Mapping[str, Any],
    catalog: ReferenceCatalog,
    path: str,
) -> dict[str, Any]:
    support_refs: list[FinalStrategySupportRef] = []
    seen: set[tuple[str, str]] = set()
    _extend_typed_support_refs(
        support_refs,
        seen,
        raw.get("evidence_summary_refs"),
        catalog,
        f"{path}.evidence_summary_refs",
    )
    _extend_legacy_refs(
        support_refs,
        seen,
        raw.get("evidence_refs"),
        catalog,
        f"{path}.evidence_refs",
    )
    return {
        "section_id": raw.get("section_id"),
        "title": raw.get("title"),
        "narrative": raw.get("narrative") or raw.get("summary"),
        "actions": [
            _normalize_action(action, catalog, f"{path}.actions[{index}]")
            for index, action in enumerate(_mapping_list(raw.get("actions")))
        ],
        "evidence_summary_refs": [item.model_dump(mode="json") for item in support_refs],
        "limitations": _unique_strings(raw.get("limitations")),
    }


def _normalize_action(
    raw: Mapping[str, Any],
    catalog: ReferenceCatalog,
    path: str,
) -> dict[str, Any]:
    support_refs: list[FinalStrategySupportRef] = []
    seen: set[tuple[str, str]] = set()
    _extend_typed_support_refs(
        support_refs,
        seen,
        raw.get("support_refs"),
        catalog,
        f"{path}.support_refs",
    )
    for field, namespace in (
        ("evidence_refs", None),
        ("eda_result_refs", "evidence"),
        ("source_refs", "source_claim"),
        ("risk_ids", "risk"),
        ("validation_requirement_ids", "validation_requirement"),
        ("safety_constraint_ids", "safety_constraint"),
        ("related_hypothesis_ids", "hypothesis"),
        ("hypothesis_ids", "hypothesis"),
    ):
        _extend_legacy_refs(
            support_refs,
            seen,
            raw.get(field),
            catalog,
            f"{path}.{field}",
            expected_namespace=namespace,
        )

    related_hypothesis_ids = _unique_strings([
        *_string_list(raw.get("related_hypothesis_ids")),
        *_string_list(raw.get("hypothesis_ids")),
    ])
    return {
        "action_id": raw.get("action_id"),
        "priority": raw.get("priority"),
        "action": raw.get("action"),
        "reason": raw.get("reason"),
        "support_refs": [item.model_dump(mode="json") for item in support_refs],
        "related_hypothesis_ids": related_hypothesis_ids,
        "experiment_ids": _unique_strings(raw.get("experiment_ids")),
        "source_claim": raw.get("source_claim"),
        "validation_strategy": raw.get("validation_strategy"),
        "confidence": raw.get("confidence") or "medium",
        "evidence_origin": _canonical_evidence_origin(raw.get("evidence_origin")),
        "limitations": _unique_strings(raw.get("limitations")),
    }


def _compile_action(action: FinalStrategyActionDraft) -> dict[str, Any]:
    refs = _split_support_refs(action.support_refs)
    hypotheses = list(dict.fromkeys([
        *action.related_hypothesis_ids,
        *refs["related_hypothesis_ids"],
    ]))
    return {
        "action_id": action.action_id,
        "priority": action.priority,
        "action": action.action,
        "reason": action.reason,
        "evidence_refs": refs["evidence_refs"],
        "related_hypothesis_ids": hypotheses,
        "hypothesis_ids": list(hypotheses),
        "experiment_ids": list(action.experiment_ids),
        "source_claim": action.source_claim,
        "source_refs": refs["source_refs"],
        "risk_ids": refs["risk_ids"],
        "validation_requirement_ids": refs["validation_requirement_ids"],
        "safety_constraint_ids": refs["safety_constraint_ids"],
        "validation_strategy": action.validation_strategy,
        "confidence": action.confidence,
        "evidence_origin": action.evidence_origin,
        "limitations": list(action.limitations),
    }


def _split_support_refs(
    values: list[FinalStrategySupportRef],
) -> dict[str, list[str]]:
    fields = {
        "evidence": "evidence_refs",
        "hypothesis": "related_hypothesis_ids",
        "source_claim": "source_refs",
        "risk": "risk_ids",
        "validation_requirement": "validation_requirement_ids",
        "safety_constraint": "safety_constraint_ids",
    }
    result = {field: [] for field in fields.values()}
    for item in values:
        field = fields[item.namespace]
        if item.ref_id not in result[field]:
            result[field].append(item.ref_id)
    return result


def _canonical_evidence_origin(value: Any) -> str:
    normalized = str(value or "Hypothesis-to-test").strip()
    for dash in ("‑", "–", "—", "−", "‐", "‒"):
        normalized = normalized.replace(dash, "-")
    aliases = {
        "eda-confirmed": "EDA-confirmed",
        "eda-inferred": "EDA-inferred",
        "source-supported": "Source-supported",
        "hypothesis-to-test": "Hypothesis-to-test",
        "safety-warning": "Safety-warning",
        "fallback-generated": "Fallback-generated",
    }
    return aliases.get(normalized.casefold(), normalized)


def _extend_typed_support_refs(
    output: list[FinalStrategySupportRef],
    seen: set[tuple[str, str]],
    values: Any,
    catalog: ReferenceCatalog,
    field_path: str,
) -> None:
    for index, value in enumerate(values if isinstance(values, list) else []):
        try:
            support_ref = FinalStrategySupportRef.model_validate(value)
        except ValidationError:
            raise
        _append_resolved_ref(
            output,
            seen,
            ref_id=support_ref.ref_id,
            expected_namespace=support_ref.namespace,
            catalog=catalog,
            field_path=f"{field_path}[{index}]",
        )


def _extend_legacy_refs(
    output: list[FinalStrategySupportRef],
    seen: set[tuple[str, str]],
    values: Any,
    catalog: ReferenceCatalog,
    field_path: str,
    *,
    expected_namespace: ReferenceNamespace | None = None,
) -> None:
    for index, ref_id in enumerate(_string_list(values)):
        _append_resolved_ref(
            output,
            seen,
            ref_id=ref_id,
            expected_namespace=expected_namespace,
            catalog=catalog,
            field_path=f"{field_path}[{index}]",
        )


def _append_resolved_ref(
    output: list[FinalStrategySupportRef],
    seen: set[tuple[str, str]],
    *,
    ref_id: str,
    expected_namespace: ReferenceNamespace | None,
    catalog: ReferenceCatalog,
    field_path: str,
) -> None:
    if expected_namespace in {None, "evidence"}:
        ref_id = _PROMPT_CONTEXT_EVIDENCE_ALIASES.get(ref_id, ref_id)
    resolution = catalog.resolve(ref_id, expected_namespace)
    if not resolution.is_resolved or resolution.entry is None:
        diagnostics = "; ".join(item.message for item in resolution.diagnostics)
        issue = FinalStrategyDraftReferenceIssue(
            field_path=field_path,
            ref_id=ref_id,
            expected_namespace=expected_namespace,
            resolution_status=resolution.status,
            message=diagnostics or "Reference could not be resolved through the catalog.",
        )
        raise FinalStrategyDraftReferenceError([issue])
    key = (resolution.entry.namespace, resolution.entry.canonical_ref)
    if key in seen:
        return
    seen.add(key)
    output.append(FinalStrategySupportRef(
        namespace=resolution.entry.namespace,
        ref_id=resolution.entry.canonical_ref,
    ))


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique_strings(value: Any) -> list[str]:
    return list(dict.fromkeys(_string_list(value)))


__all__ = [
    "compile_final_strategy_draft",
    "FinalStrategyActionDraft",
    "FinalStrategyDraft",
    "FinalStrategyDraftReferenceError",
    "FinalStrategyDraftReferenceIssue",
    "FinalStrategySectionDraft",
    "FinalStrategySupportRef",
    "ReferenceNamespace",
    "normalize_legacy_final_strategy_to_draft",
]
