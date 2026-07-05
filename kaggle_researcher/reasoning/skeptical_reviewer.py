from __future__ import annotations

from typing import Any

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import call_reasoning_json, format_retrieved_documents
from kaggle_researcher.schemas import RetrievedDocument, ReviewResult


async def review(
    draft_sections: dict[str, Any],
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> ReviewResult:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    return await call_reasoning_json(
        client=client,
        model=model,
        system_prompt=(
            "Act as a critical Kaggle Grandmaster reviewer. Do not add new facts. "
            "Identify unsupported claims, generic advice, and unnecessary experiments. "
            "Return revised_sections with the same high-level keys as draft_sections."
        ),
        user_payload={
            "draft_sections": draft_sections,
            "retrieved_documents": format_retrieved_documents(docs),
            "expected_schema": ReviewResult.model_json_schema(),
        },
        result_model=ReviewResult,
    )
