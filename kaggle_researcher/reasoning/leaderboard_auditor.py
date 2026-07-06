from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import (
    call_reasoning_json,
    format_retrieved_documents,
    validate_evidence_ids,
)
from kaggle_researcher.schemas import LeaderboardAuditResult, PlanData, RetrievedDocument, ValidationResult


FORBIDDEN_MEASURED_CORRELATION_PHRASES = (
    "lb/cv correlation was measured",
    "lb cv correlation was measured",
    "leaderboard/cv correlation was measured",
    "leaderboard cv correlation was measured",
    "measured lb/cv correlation",
    "measured leaderboard/cv correlation",
    "actual lb/cv correlation",
)


async def audit_leaderboard_risk(
    competition_desc: str,
    plan_data: PlanData,
    validation_result: ValidationResult,
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> LeaderboardAuditResult:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    result = await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Leaderboard Auditor. Assess public/private leaderboard shake-up risk "
            "and submission selection rules. Warn against public LB overfitting. Do not claim "
            "actual LB/CV correlation was measured. Use task_type and metric from plan_data, "
            "and use validation_result as context. Return shake_up_risk, public_lb_trust, "
            "submission_selection_rule, warnings, confidence, and evidence_ids. "
            "Leaderboard/shake-up claims must include provenance where possible and "
            "not_verified_on_data for unmeasured behavior."
        ),
        user_payload={
            "competition_desc": competition_desc,
            "plan_data": plan_data.model_dump(),
            "validation_result": validation_result.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": LeaderboardAuditResult.model_json_schema(),
        },
        result_model=LeaderboardAuditResult,
    )
    unknown_ids = validate_evidence_ids(result, docs)
    if unknown_ids:
        raise ValueError(f"LeaderboardAuditResult contains unknown evidence_ids: {unknown_ids}")
    forbidden_phrase = _find_forbidden_measured_correlation_phrase(result)
    if forbidden_phrase is not None:
        raise ValueError(
            "LeaderboardAuditResult contains a prohibited measured-correlation claim: "
            f"{forbidden_phrase!r}"
        )
    return result


def _find_forbidden_measured_correlation_phrase(result: LeaderboardAuditResult) -> str | None:
    haystack = " ".join(
        [
            result.shake_up_risk,
            result.submission_selection_rule,
            result.public_lb_trust,
            *result.warnings,
        ]
    ).lower()
    return next(
        (phrase for phrase in FORBIDDEN_MEASURED_CORRELATION_PHRASES if phrase in haystack),
        None,
    )
