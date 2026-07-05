from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import call_reasoning_json, format_retrieved_documents
from kaggle_researcher.schemas import MetricResult, PlanData, RetrievedDocument


async def analyze_metric(
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> MetricResult:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    return await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "You are the Metric Specialist. Explain metric implications. Guidance: "
            "AUC/Gini -> ranking and rank averaging; LogLoss -> calibration and clipping; "
            "F1/Dice -> threshold search; RMSE/RMSLE -> target transforms and clipping; "
            "MAP@K/NDCG -> ranking and candidate generation. Do not claim dataset analysis. "
            "Important metric claims must include provenance labels such as arxiv, kaggle, "
            "heuristic, and not_verified_on_data."
        ),
        user_payload={
            "plan_data": plan_data.model_dump(),
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": MetricResult.model_json_schema(),
        },
        result_model=MetricResult,
    )
