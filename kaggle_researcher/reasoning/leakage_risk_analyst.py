from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import (
    call_reasoning_json,
    format_retrieved_documents,
    validate_evidence_ids,
)
from kaggle_researcher.schemas import LeakageRiskResult, PlanData, RetrievedDocument


FORBIDDEN_CONFIRMATION_PHRASES = (
    "leakage found",
    "leakage confirmed",
    "confirmed leakage",
    "data leakage found",
    "data leakage confirmed",
)


async def analyze_leakage_risk(
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> LeakageRiskResult:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    result = await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Leakage Risk Analyst. Produce hypotheses and recommended checks only. "
            "Forbid phrases like 'leakage found' or 'leakage confirmed'. Use language such as "
            "'possible risk', 'hypothesis', and 'recommended check'. Leakage hypotheses must "
            "include provenance and not_verified_on_data. Confidence should usually be low or "
            "medium because real data is not visible. Return possible_issues and "
            "recommended_checks. Include evidence_ids where relevant."
        ),
        user_payload={
            "competition_desc": competition_desc,
            "plan_data": plan_data.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": LeakageRiskResult.model_json_schema(),
        },
        result_model=LeakageRiskResult,
    )
    unknown_ids = validate_evidence_ids(result, docs)
    if unknown_ids:
        raise ValueError(f"LeakageRiskResult contains unknown evidence_ids: {unknown_ids}")
    forbidden_phrase = _find_forbidden_confirmation_phrase(result)
    if forbidden_phrase is not None:
        raise ValueError(
            "LeakageRiskResult contains a prohibited confirmed-leakage phrase: "
            f"{forbidden_phrase!r}"
        )
    return result


def _find_forbidden_confirmation_phrase(result: LeakageRiskResult) -> str | None:
    haystack = " ".join(
        [
            result.risk_level,
            *result.possible_issues,
            *result.recommended_checks,
        ]
    ).lower()
    return next((phrase for phrase in FORBIDDEN_CONFIRMATION_PHRASES if phrase in haystack), None)
