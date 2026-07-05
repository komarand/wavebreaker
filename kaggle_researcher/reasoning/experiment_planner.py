from __future__ import annotations

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import format_retrieved_documents
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeakageRiskResult,
    MetricResult,
    RetrievedDocument,
    ValidationResult,
)


async def plan_experiments(
    validation_result: ValidationResult,
    leakage_result: LeakageRiskResult,
    metric_result: MetricResult,
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> list[ExperimentItem]:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    response = await client.chat_json(
        model=model,
        system_prompt=(
            "Return JSON only as an object with key 'experiments'. Each item must match "
            "ExperimentItem schema. Priorities must be P0, P1, P2, or P3. Include honest "
            "validation and baseline near the top. Do not claim EDA or leakage detection ran."
        ),
        user_prompt="\n\n".join(
            [
                f"Validation: {validation_result.model_dump_json()}",
                f"Leakage: {leakage_result.model_dump_json()}",
                f"Metric: {metric_result.model_dump_json()}",
                f"Retrieved documents:\n{format_retrieved_documents(docs)}",
                f"ExperimentItem schema: {ExperimentItem.model_json_schema()}",
            ]
        ),
        timeout=120,
    )
    raw_items = response.get("experiments", response if isinstance(response, list) else [])
    experiments = [ExperimentItem.model_validate(item) for item in raw_items]
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return sorted(experiments, key=lambda item: priority_order[item.priority])
