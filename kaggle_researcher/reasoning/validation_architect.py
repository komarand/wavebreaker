from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import call_reasoning_json, format_retrieved_documents
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
    return await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Validation Architect. Recommend CV/split strategy only. "
            "Do not propose models or feature engineering. Keep confidence below high "
            "unless sources explicitly describe split, time, or group structure."
        ),
        user_payload={
            "competition_desc": competition_desc,
            "plan_data": plan_data.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": ValidationResult.model_json_schema(),
        },
        result_model=ValidationResult,
    )
