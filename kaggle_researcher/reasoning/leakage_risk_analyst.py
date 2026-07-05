from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import call_reasoning_json, format_retrieved_documents
from kaggle_researcher.schemas import LeakageRiskResult, PlanData, RetrievedDocument


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
    return await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Leakage Risk Analyst. Produce hypotheses and recommended checks only. "
            "Forbid phrases like 'leakage found' or 'leakage confirmed'. Use language such as "
            "'possible risk', 'hypothesis', and 'recommended check'. Leakage hypotheses must "
            "include provenance and not_verified_on_data."
        ),
        user_payload={
            "competition_desc": competition_desc,
            "plan_data": plan_data.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": LeakageRiskResult.model_json_schema(),
        },
        result_model=LeakageRiskResult,
    )
