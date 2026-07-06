from __future__ import annotations

from typing import Any

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import (
    call_reasoning_json,
    format_retrieved_documents,
    validate_evidence_ids,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument, ValidationResult


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
            "provenance labels such as kaggle, arxiv, heuristic, not_verified_on_data."
        ),
        user_payload={
            "competition_desc": competition_desc,
            "plan_data": plan_data.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": ValidationResult.model_json_schema(),
        },
        result_model=ValidationResult,
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
    return ValidationResult.model_validate(enforced)


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
        "description": "Primary: out-of-time holdout on latest periods plus rolling/expanding temporal CV.",
        "why": "Temporal/stability signals require preserving chronology and future-period generalization.",
        "must_preserve_chronology": True,
    }
    updated["secondary_validation"] = {
        "method": "StratifiedGroupKFold",
        "description": "Secondary diagnostic only; not a temporal guarantee.",
        "allowed_only_if": (
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
