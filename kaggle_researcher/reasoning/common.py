from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

from pydantic import BaseModel
from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.contracts.errors import BoundaryRepairError
from kaggle_researcher.contracts.repair import validate_with_one_repair
from kaggle_researcher.reasoning.prompts import SYSTEM_RULES
from kaggle_researcher.schemas import RetrievedDocument


ResultModel = TypeVar("ResultModel", bound=BaseModel)
CANONICAL_REASONING_EVIDENCE_IDS = (
    "validation_result",
    "leakage_result",
    "metric_result",
)


class ReasoningResponseValidationError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        model: str,
        result_model: type[BaseModel],
        validation_errors: list[dict[str, Any]],
        returned_keys: list[str],
    ) -> None:
        self.stage = stage
        self.model = model
        self.result_model = result_model.__name__
        self.validation_errors = validation_errors
        self.returned_keys = returned_keys
        invalid_fields = ", ".join(".".join(str(part) for part in error.get("loc", ())) for error in validation_errors[:8])
        super().__init__(
            f"Reasoning response validation failed for model {model!r} as {self.result_model} "
            f"at {stage}; Returned keys: {returned_keys}; invalid fields: {invalid_fields or 'unknown'}."
        )


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
    stage: str | None = None,
    artifact_dir: Path | str | None = None,
    raw_artifact_name: str | None = None,
    response_normalizer: Callable[[Any], Any] | None = None,
) -> ResultModel:
    response = await client.chat_json(
        model=model,
        system_prompt=f"{SYSTEM_RULES}\n\n{system_prompt}",
        user_prompt=json.dumps(user_payload, ensure_ascii=False, indent=2),
        timeout=120,
    )
    repaired_response: Any = None

    async def repair_once(repair_input: dict[str, Any]) -> Any:
        nonlocal repaired_response
        if artifact_dir is not None and raw_artifact_name is not None:
            path = Path(artifact_dir) / raw_artifact_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        repaired_response = await client.chat_json(
            model=model,
            system_prompt=(
                f"{SYSTEM_RULES}\n\nCorrect the existing response to the provided schema. "
                "Preserve the original recommendation. Do not invent optional results; use null "
                "when an optional result is unsupported. Return JSON only."
            ),
            user_prompt=json.dumps({
                "validation_errors": repair_input["validation_issues"],
                "expected_schema": result_model.model_json_schema(),
                "invalid_response": response,
            }, ensure_ascii=False, indent=2),
            timeout=120,
        )
        return (
            response_normalizer(repaired_response)
            if response_normalizer is not None
            else repaired_response
        )

    try:
        normalized_response = (
            response_normalizer(response) if response_normalizer is not None else response
        )
        validated = await validate_with_one_repair(
            normalized_response,
            model=result_model,
            repair=repair_once,
            contract_name=stage or result_model.__name__,
        )
        return validated.value
    except BoundaryRepairError as exc:
        returned_keys = list(repaired_response.keys()) if isinstance(repaired_response, dict) else []
        validation_errors = [
            {
                "loc": tuple(part for part in issue.field_path.split(".") if part),
                "type": issue.expected,
                "msg": issue.reason,
            }
            for issue in exc.issues
        ]
        raise ReasoningResponseValidationError(
            stage=stage or result_model.__name__,
            model=model,
            result_model=result_model,
            validation_errors=validation_errors,
            returned_keys=returned_keys,
        ) from exc


def known_evidence_ids(
    docs: Iterable[RetrievedDocument],
    *,
    additional_ids: Iterable[str] = (),
) -> list[str]:
    """Build the canonical evidence registry for a reasoning stage."""
    return sorted({*(document.id for document in docs), *additional_ids})


def validate_evidence_ids(
    result: BaseModel | dict[str, Any],
    docs: list[RetrievedDocument],
    *,
    additional_ids: Iterable[str] = (),
) -> list[str]:
    known_ids = set(known_evidence_ids(docs, additional_ids=additional_ids))
    if isinstance(result, dict):
        evidence_ids = result.get("evidence_ids", [])
    else:
        evidence_ids = getattr(result, "evidence_ids", [])
    return [evidence_id for evidence_id in evidence_ids if evidence_id not in known_ids]
