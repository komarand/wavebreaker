from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import (
    call_reasoning_json,
    format_retrieved_documents,
    validate_evidence_ids,
)
from kaggle_researcher.schemas import (
    PlanData,
    RetrievedDocument,
    ValidationPolicy,
    ValidationResult,
)


class ValidationNormalizationError(ValueError):
    """The Validation Architect payload cannot be normalized without guessing."""


def normalize_validation_result_payload(payload: Any) -> dict[str, Any]:
    """Project Validation Architect output onto the strict canonical schema.

    Known policy aliases are folded into ``reason`` or top-level ``policy_notes``.
    Other nested fields are preserved as deterministic policy notes and removed
    before Pydantic sees the canonical payload.
    """

    if not isinstance(payload, Mapping):
        raise ValidationNormalizationError("ValidationResult output must be a JSON object")

    normalized = dict(payload)
    unknown_top_level = sorted(set(normalized) - set(ValidationResult.model_fields))
    if unknown_top_level:
        raise ValidationNormalizationError(
            "Unknown top-level ValidationResult fields: " + ", ".join(unknown_top_level)
        )

    raw_notes = normalized.get("policy_notes")
    if raw_notes is None:
        notes: list[str] = []
    elif isinstance(raw_notes, list) and all(isinstance(note, str) for note in raw_notes):
        notes = list(raw_notes)
    else:
        raise ValidationNormalizationError("policy_notes must be a list of strings or null")

    for field_name in ("primary_validation", "secondary_validation"):
        raw_policy = normalized.get(field_name)
        if raw_policy is None:
            continue
        if not isinstance(raw_policy, Mapping):
            raise ValidationNormalizationError(f"{field_name} must be a JSON object or null")

        policy = dict(raw_policy)
        reason_parts: list[str] = []
        for alias in ("reason", "description", "why"):
            value = policy.pop(alias, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValidationNormalizationError(f"{field_name}.{alias} must be a string")
            stripped = value.strip()
            if stripped and stripped not in reason_parts:
                reason_parts.append(stripped)
        if reason_parts:
            policy["reason"] = " ".join(reason_parts)

        chronology = policy.pop("must_preserve_chronology", None)
        if chronology is not None and not isinstance(chronology, bool):
            raise ValidationNormalizationError(
                f"{field_name}.must_preserve_chronology must be boolean"
            )
        if chronology:
            notes.append(f"{field_name} constraint: must preserve chronology.")

        allowed_only_if = policy.pop("allowed_only_if", None)
        if allowed_only_if is not None:
            if not isinstance(allowed_only_if, str):
                raise ValidationNormalizationError(
                    f"{field_name}.allowed_only_if must be a string"
                )
            if allowed_only_if.strip():
                notes.append(
                    f"{field_name} constraint: allowed only if {allowed_only_if.strip()}"
                )

        unknown_nested = sorted(set(policy) - set(ValidationPolicy.model_fields))
        for key in unknown_nested:
            value = policy.pop(key)
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
            notes.append(f"{field_name} unmodeled constraint {key}={serialized}")
        normalized[field_name] = policy

    normalized["policy_notes"] = list(dict.fromkeys(notes))
    return normalized


async def design_validation(
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> ValidationResult:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    result = await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Validation Architect. Recommend CV/split strategy only. "
            "Do not propose models or feature engineering. Keep confidence below high "
            "unless sources explicitly describe split, time, or group structure. "
            "Separate facts, hypotheses, and recommendations in the returned fields. "
            "Every source-backed claim must cite evidence_ids from retrieved_documents. "
            "If temporal indicators exist (date, time, week, month, WEEK_NUM, period, "
            "timestamp, stability, out-of-time, OOT), primary validation must be strict "
            "temporal: out-of-time holdout on latest periods plus rolling/expanding CV. "
            "StratifiedGroupKFold may be secondary/diagnostic only if it never trains on "
            "future periods to validate on past periods. Important claims must include "
            "provenance labels such as kaggle, arxiv, heuristic, not_verified_on_data. "
            "secondary_validation may be null when no well-supported distinct secondary "
            "validation design exists. Do not duplicate primary_validation merely to avoid "
            "null. When non-null, secondary_validation must include the same required method "
            "field as primary_validation. Example without a secondary: secondary_validation: null. "
            "Return only fields present in the supplied ValidationResult JSON schema. In each "
            "primary_validation or secondary_validation object, the only allowed keys are method, "
            "reason, n_splits, shuffle, group_column, and split_column. Put chronology requirements, "
            "conditional-use rules, warnings, and other constraints in top-level policy_notes or "
            "failure_modes. Never return description, why, must_preserve_chronology, or "
            "allowed_only_if inside a validation object."
        ),
        user_payload={
            "competition_desc": competition_desc,
            "plan_data": plan_data.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": ValidationResult.model_json_schema(),
        },
        result_model=ValidationResult,
        stage="validation_architect",
        response_normalizer=normalize_validation_result_payload,
    )
    unknown_ids = validate_evidence_ids(result, docs)
    if unknown_ids:
        raise ValueError(f"ValidationResult contains unknown evidence_ids: {unknown_ids}")

    enforced = enforce_temporal_validation_policy(
        result.model_dump(),
        competition_desc=competition_desc,
        plan_data=plan_data.model_dump(),
        retrieved_documents=[doc.model_dump(mode="json") for doc in docs],
    )
    enforced = enforce_validation_confidence_policy(
        enforced,
        competition_desc=competition_desc,
        plan_data=plan_data.model_dump(),
        retrieved_documents=[doc.model_dump(mode="json") for doc in docs],
    )
    return ValidationResult.model_validate(normalize_validation_result_payload(enforced))


def enforce_temporal_validation_policy(
    validation_result: dict[str, Any],
    competition_desc: str,
    plan_data: dict[str, Any],
    retrieved_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if not _has_temporal_signal(competition_desc, plan_data, retrieved_documents):
        validation_result.setdefault("policy_enforced", False)
        validation_result.setdefault("policy_notes", [])
        return validation_result

    updated = dict(validation_result)
    notes = list(updated.get("policy_notes") or [])
    original_text = " ".join(
        str(updated.get(key, ""))
        for key in ("recommended_cv", "likely_split", "reasoning", "primary_validation")
    ).lower()
    risky_primary = any(
        token in original_text
        for token in ("stratifiedgroupkfold", "stratified group kfold", "randomkfold", "stratifiedkfold")
    )
    updated["primary_validation"] = {
        "method": "out_of_time_holdout_and_rolling_cv",
        "reason": (
            "Primary: out-of-time holdout on latest periods plus rolling/expanding temporal CV. "
            "Temporal/stability signals require preserving chronology and future-period generalization."
        ),
    }
    updated["secondary_validation"] = {
        "method": "StratifiedGroupKFold",
        "reason": (
            "Secondary diagnostic only; not a temporal guarantee. Allowed only if "
            "groups are ordered or used only as a robustness diagnostic; never train on future "
            "periods to validate on past periods"
        ),
    }
    updated["do_not_use"] = [
        "RandomKFold",
        "plain StratifiedKFold",
        "StratifiedGroupKFold that mixes future periods into training",
    ]
    updated["recommended_cv"] = (
        "Out-of-time holdout + rolling/expanding temporal CV; "
        "StratifiedGroupKFold only as secondary diagnostic"
    )
    updated["validation_risk"] = "high"
    failure_modes = list(updated.get("failure_modes") or [])
    for failure_mode in (
        "Chronology violation: training on future periods and validating on past periods.",
        "Over-trusting StratifiedGroupKFold as a temporal guarantee.",
    ):
        if failure_mode not in failure_modes:
            failure_modes.append(failure_mode)
    updated["failure_modes"] = failure_modes
    updated["policy_enforced"] = True
    if risky_primary:
        notes.append("Risky validation recommendation detected and demoted from primary.")
    notes.append("Temporal/stability signals detected; strict temporal validation promoted to primary.")
    notes.append("primary_validation constraint: must preserve chronology.")
    notes.append(
        "secondary_validation constraint: allowed only if groups are ordered or used only as a "
        "robustness diagnostic; never train on future periods to validate on past periods"
    )
    updated["policy_notes"] = list(dict.fromkeys(notes))
    return updated


def enforce_validation_confidence_policy(
    validation_result: dict[str, Any],
    competition_desc: str,
    plan_data: dict[str, Any],
    retrieved_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if validation_result.get("confidence") != "high":
        validation_result.setdefault("policy_notes", [])
        return validation_result
    if _has_split_structure_signal(competition_desc, plan_data, retrieved_documents):
        validation_result.setdefault("policy_notes", [])
        return validation_result

    updated = dict(validation_result)
    notes = list(updated.get("policy_notes") or [])
    updated["confidence"] = "medium"
    notes.append(
        "Confidence reduced from high because retrieved_documents do not describe split, time, or group structure."
    )
    updated["policy_notes"] = list(dict.fromkeys(notes))
    return updated


def _has_temporal_signal(
    competition_desc: str,
    plan_data: dict[str, Any],
    retrieved_documents: list[dict[str, Any]],
) -> bool:
    keywords = (
        "date",
        "time",
        "week",
        "month",
        "week_num",
        "period",
        "timestamp",
        "stability",
        "out-of-time",
        "oot",
    )
    haystack = " ".join(
        [
            competition_desc,
            str(plan_data.get("metric", "")),
            str(plan_data.get("domain", "")),
            *[f"{doc.get('title', '')} {doc.get('content', '')}" for doc in retrieved_documents],
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)


def _has_split_structure_signal(
    competition_desc: str,
    plan_data: dict[str, Any],
    retrieved_documents: list[dict[str, Any]],
) -> bool:
    keywords = (
        "split",
        "fold",
        "group",
        "stratified",
        "cv",
        "cross-validation",
        "cross validation",
        "date",
        "time",
        "week",
        "month",
        "week_num",
        "period",
        "timestamp",
        "out-of-time",
        "oot",
    )
    haystack = " ".join(
        [
            competition_desc,
            str(plan_data.get("metric", "")),
            str(plan_data.get("domain", "")),
            *[f"{doc.get('title', '')} {doc.get('content', '')}" for doc in retrieved_documents],
        ]
    ).lower()
    return any(keyword in haystack for keyword in keywords)
