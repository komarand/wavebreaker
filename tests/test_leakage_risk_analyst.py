from __future__ import annotations

import json

import pytest

from kaggle_researcher.reasoning.leakage_risk_analyst import analyze_leakage_risk
from kaggle_researcher.schemas import LeakageRiskResult, PlanData, RetrievedDocument


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    async def chat_json(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _plan() -> PlanData:
    return PlanData(
        task_type="binary_classification",
        metric="auc",
        domain="tabular credit risk",
        kaggle_queries=["credit risk leakage checks"],
    )


def _doc(doc_id: str = "doc-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Leakage discussion",
        url="https://example.com/leakage",
        content="Check whether duplicated entities or post-target timestamps appear in features.",
        score=0.75,
        rrf_score=0.3,
    )


def _leakage_response(evidence_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "confidence": "medium",
        "evidence_ids": evidence_ids or ["doc-1"],
        "risk_level": "medium",
        "possible_issues": [
            "Possible risk: duplicate entities may cross validation folds; not_verified_on_data."
        ],
        "recommended_checks": [
            "Recommended check: inspect entity overlap and timestamp ordering before modeling."
        ],
    }


@pytest.mark.asyncio
async def test_analyze_leakage_risk_validates_mock_response_into_result() -> None:
    client = FakeClient(_leakage_response())

    result = await analyze_leakage_risk(
        competition_desc="Predict credit default from tabular features.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert isinstance(result, LeakageRiskResult)
    assert result.confidence == "medium"
    assert result.evidence_ids == ["doc-1"]
    assert result.risk_level == "medium"
    assert result.possible_issues
    assert result.recommended_checks

    system_prompt = str(client.kwargs["system_prompt"])
    assert "Produce hypotheses and recommended checks only." in system_prompt
    assert "Forbid phrases like 'leakage found' or 'leakage confirmed'" in system_prompt
    assert "'possible risk', 'hypothesis', and 'recommended check'" in system_prompt
    assert "Confidence should usually be low or medium because real data is not visible" in system_prompt
    assert "Return possible_issues and recommended_checks" in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["competition_desc"] == "Predict credit default from tabular features."
    assert payload["plan_data"]["metric"] == "auc"
    assert "ID: doc-1" in payload["retrieved_documents"]
    assert "RRF score: 0.3000" in payload["retrieved_documents"]
    assert "expected_schema" in payload


@pytest.mark.asyncio
async def test_analyze_leakage_risk_rejects_unknown_evidence_ids() -> None:
    client = FakeClient(_leakage_response(evidence_ids=["doc-1", "missing-doc"]))

    with pytest.raises(ValueError, match="unknown evidence_ids"):
        await analyze_leakage_risk(
            competition_desc="Predict credit default from tabular features.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )


@pytest.mark.asyncio
async def test_analyze_leakage_risk_rejects_confirmed_leakage_claims() -> None:
    response = _leakage_response()
    response["possible_issues"] = ["Leakage confirmed through timestamp features."]
    client = FakeClient(response)

    with pytest.raises(ValueError, match="confirmed-leakage phrase"):
        await analyze_leakage_risk(
            competition_desc="Predict credit default from tabular features.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )
