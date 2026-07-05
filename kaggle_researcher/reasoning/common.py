from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.schemas import RetrievedDocument


SYSTEM_RULES = """Return JSON only.
Separate facts, hypotheses, and recommendations.
Include confidence and evidence_ids.
Do not claim real train/test analysis was performed.
Do not claim leakage was found or confirmed based only on text sources.
Do not include raw chain-of-thought."""

ResultModel = TypeVar("ResultModel", bound=BaseModel)


def format_retrieved_documents(docs: list[RetrievedDocument]) -> str:
    parts: list[str] = []
    for doc in docs:
        snippet = " ".join(doc.content.split())[:1200]
        parts.append(
            "\n".join(
                [
                    f"ID: {doc.id}",
                    f"Source: {doc.source}",
                    f"Title: {doc.title}",
                    f"URL: {doc.url or ''}",
                    f"RRF score: {doc.rrf_score:.4f}",
                    f"Content: {snippet}",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


async def call_reasoning_json(
    client: DeepSeekClient,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    result_model: type[ResultModel],
) -> ResultModel:
    response = await client.chat_json(
        model=model,
        system_prompt=f"{SYSTEM_RULES}\n\n{system_prompt}",
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
        timeout=120,
    )
    return result_model.model_validate(response)


def validate_evidence_ids(result: BaseModel, docs: list[RetrievedDocument]) -> list[str]:
    known_ids = {doc.id for doc in docs}
    evidence_ids = getattr(result, "evidence_ids", [])
    return [evidence_id for evidence_id in evidence_ids if evidence_id not in known_ids]
