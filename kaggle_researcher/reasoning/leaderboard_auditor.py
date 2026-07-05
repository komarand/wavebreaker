from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import call_reasoning_json, format_retrieved_documents
from kaggle_researcher.schemas import LeaderboardAuditResult, PlanData, RetrievedDocument, ValidationResult


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
    return await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Leaderboard Auditor. Assess public/private leaderboard shake-up risk "
            "and submission selection rules. Warn against public LB overfitting. Do not claim "
            "actual LB/CV correlation was measured."
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
