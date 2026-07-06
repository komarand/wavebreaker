from __future__ import annotations

import json

import pytest

from kaggle_researcher.reasoning.skeptical_reviewer import review
from kaggle_researcher.schemas import RetrievedDocument, ReviewResult


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    async def chat_json(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _doc(doc_id: str = "doc-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Review evidence",
        url="https://example.com/review",
        content="Discussion supports temporal validation but not a specific model guarantee.",
        score=0.82,
        rrf_score=0.42,
    )


def _draft_sections() -> dict[str, object]:
    return {
        "validation": {
            "recommended_cv": "Use temporal validation",
            "evidence_ids": ["doc-1"],
            "confidence": "medium",
        },
        "metric": {
            "claim": "AUC-style ranking is important",
        },
        "experiments": [
            {
                "priority": "P0",
                "experiment": "Run baseline",
                "evidence_ids": ["doc-1"],
            }
        ],
    }


def _review_response(evidence_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "confidence": "medium",
        "evidence_ids": evidence_ids or ["doc-1"],
        "unsupported_claims": ["Model guarantee is unsupported."],
        "too_generic": ["Try many models."],
        "unnecessary_experiments": ["Large ensemble before baseline."],
        "revised_sections": {
            "validation": {
                "recommended_cv": "Use temporal validation",
                "evidence_ids": ["doc-1"],
                "confidence": "medium",
            },
            "metric": "AUC-style ranking may matter, but the source basis is weak.",
            "extra": "This key should not be preserved.",
        },
    }


@pytest.mark.asyncio
async def test_review_validates_mock_response_and_preserves_draft_keys() -> None:
    client = FakeClient(_review_response())

    result = await review(
        draft_sections=_draft_sections(),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert isinstance(result, ReviewResult)
    assert result.confidence == "medium"
    assert result.evidence_ids == ["doc-1"]
    assert set(result.revised_sections) == {"validation", "metric", "experiments"}
    assert result.revised_sections["metric"] == "AUC-style ranking may matter, but the source basis is weak."
    assert result.revised_sections["experiments"] == _draft_sections()["experiments"]
    assert "Model guarantee is unsupported." in result.unsupported_claims
    assert any("draft_sections.metric" in claim for claim in result.unsupported_claims)
    assert result.too_generic == ["Try many models."]
    assert result.unnecessary_experiments == ["Large ensemble before baseline."]

    system_prompt = str(client.kwargs["system_prompt"])
    assert "critical Kaggle Grandmaster reviewer" in system_prompt
    assert "Do not add new facts" in system_prompt
    assert "new unsupported claims" in system_prompt
    assert "unsupported_claims, too_generic, and unnecessary_experiments" in system_prompt
    assert "same high-level keys as draft_sections" in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["draft_sections"]["validation"]["recommended_cv"] == "Use temporal validation"
    assert "ID: doc-1" in payload["retrieved_documents"]
    assert "RRF score: 0.4200" in payload["retrieved_documents"]
    assert "expected_schema" in payload


@pytest.mark.asyncio
async def test_review_rejects_unknown_evidence_ids() -> None:
    client = FakeClient(_review_response(evidence_ids=["doc-1", "missing-doc"]))

    with pytest.raises(ValueError, match="unknown evidence_ids"):
        await review(
            draft_sections=_draft_sections(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )


@pytest.mark.asyncio
async def test_review_marks_key_claim_without_evidence_as_unsupported() -> None:
    client = FakeClient(
        {
            "confidence": "medium",
            "evidence_ids": [],
            "unsupported_claims": [],
            "too_generic": [],
            "unnecessary_experiments": [],
            "revised_sections": _draft_sections(),
        }
    )

    result = await review(
        draft_sections={"strategy": {"claim": "This will win the competition."}},
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert any("This will win the competition." in claim for claim in result.unsupported_claims)
