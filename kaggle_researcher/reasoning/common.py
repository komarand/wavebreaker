from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_core import ValidationError

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.prompts import SYSTEM_RULES
from kaggle_researcher.schemas import RetrievedDocument


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
    *,
    artifact_dir: Path | str | None = None,
    raw_artifact_name: str | None = None,
) -> ResultModel:
    response = await client.chat_json(
        model=model,
        system_prompt=f"{SYSTEM_RULES}\n\n{system_prompt}",
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
        timeout=120,
    )
    try:
        return result_model.model_validate(response)
    except ValidationError as exc:
        if artifact_dir is not None and raw_artifact_name is not None:
            path = Path(artifact_dir) / raw_artifact_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

        returned_keys: object
        if isinstance(response, dict):
            returned_keys = list(response.keys())
        else:
            returned_keys = type(response).__name__
        raise RuntimeError(
            f"Failed to validate reasoning response from model {model!r} as "
            f"{result_model.__name__}. Returned keys: {returned_keys}. "
            f"Validation error: {exc}"
        ) from exc


def validate_evidence_ids(result: BaseModel | dict[str, Any], docs: list[RetrievedDocument]) -> list[str]:
    known_ids = {doc.id for doc in docs}
    if isinstance(result, dict):
        evidence_ids = result.get("evidence_ids", [])
    else:
        evidence_ids = getattr(result, "evidence_ids", [])
    return [evidence_id for evidence_id in evidence_ids if evidence_id not in known_ids]
