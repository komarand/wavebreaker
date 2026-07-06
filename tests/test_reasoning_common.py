from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel

from kaggle_researcher.reasoning.common import (
    SYSTEM_RULES,
    call_reasoning_json,
    format_retrieved_documents,
    validate_evidence_ids,
)
from kaggle_researcher.reasoning.prompts import SYSTEM_RULES as PROMPT_SYSTEM_RULES
from kaggle_researcher.schemas import RetrievedDocument, ReasoningBaseResult


def _doc(
    doc_id: str = "doc-1",
    *,
    content: str = "First line.\nSecond line with more detail.",
) -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Discussion thread",
        url="https://example.com/thread",
        content=content,
        score=0.72,
        rrf_score=0.123456,
        metadata={"rank": 1},
    )


def test_format_retrieved_documents_includes_required_fields() -> None:
    formatted = format_retrieved_documents([_doc()])

    assert "ID: doc-1" in formatted
    assert "Source: kaggle" in formatted
    assert "Title: Discussion thread" in formatted
    assert "URL: https://example.com/thread" in formatted
    assert "RRF score: 0.1235" in formatted
    assert "Content: First line. Second line with more detail." in formatted


def test_validate_evidence_ids_detects_unknown_ids_for_models_and_dicts() -> None:
    docs = [_doc("known-1"), _doc("known-2")]
    result = ReasoningBaseResult(confidence="medium", evidence_ids=["known-1", "missing-1"])

    assert validate_evidence_ids(result, docs) == ["missing-1"]
    assert validate_evidence_ids({"evidence_ids": ["known-2", "missing-2"]}, docs) == ["missing-2"]


def test_system_rules_are_shared_and_include_required_guardrails() -> None:
    assert SYSTEM_RULES == PROMPT_SYSTEM_RULES
    for expected in [
        "Return JSON only.",
        "Separate facts, hypotheses, and recommendations.",
        "Include confidence.",
        "Include evidence_ids.",
        "Use retrieved_documents terminology.",
        "Do not claim real train/test analysis was performed.",
        "Do not confirm leakage based only on text sources.",
        "Do not implement or imply data-execution features.",
    ]:
        assert expected in SYSTEM_RULES


def test_call_reasoning_json_sends_rules_and_payload_and_validates_model() -> None:
    class ResultModel(BaseModel):
        confidence: str
        evidence_ids: list[str]
        answer: str

    class FakeClient:
        def __init__(self) -> None:
            self.kwargs = {}

        async def chat_json(self, **kwargs):
            self.kwargs = kwargs
            return {"confidence": "medium", "evidence_ids": ["doc-1"], "answer": "ok"}

    client = FakeClient()

    result = asyncio.run(
        call_reasoning_json(
            client=client,
            model="reasoning-model",
            system_prompt="Module-specific rules.",
            user_payload={"retrieved_documents": "ID: doc-1"},
            result_model=ResultModel,
        )
    )

    assert result.answer == "ok"
    assert client.kwargs["model"] == "reasoning-model"
    assert SYSTEM_RULES in client.kwargs["system_prompt"]
    assert "Module-specific rules." in client.kwargs["system_prompt"]
    assert json.loads(client.kwargs["user_prompt"]) == {"retrieved_documents": "ID: doc-1"}
    assert client.kwargs["timeout"] == 120
