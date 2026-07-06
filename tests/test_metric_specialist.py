from __future__ import annotations

import json

import pytest

from kaggle_researcher.reasoning.metric_specialist import analyze_metric
from kaggle_researcher.schemas import MetricResult, PlanData, RetrievedDocument


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    async def chat_json(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _plan(metric: str = "Gini") -> PlanData:
    return PlanData(
        task_type="binary_classification",
        metric=metric,
        domain="credit risk",
        kaggle_queries=["credit risk gini"],
    )


def _doc(doc_id: str = "doc-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Competition discussion",
        url="https://example.com/discussion",
        content="Gini is related to AUC, so rank quality and ensembling can matter.",
        score=0.8,
        rrf_score=0.25,
    )


def _metric_response(evidence_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "confidence": "medium",
        "evidence_ids": evidence_ids or ["doc-1"],
        "metric_explanation": "Gini optimizes ranking quality and is related to AUC.",
        "needs_calibration": False,
        "rank_averaging_useful": True,
        "threshold_search_needed": False,
        "surrogate_loss_suggestion": "Train with logloss/AUC-oriented validation and select by Gini.",
    }


@pytest.mark.asyncio
async def test_analyze_metric_validates_mock_response_into_metric_result() -> None:
    client = FakeClient(_metric_response())

    result = await analyze_metric(
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert isinstance(result, MetricResult)
    assert result.confidence == "medium"
    assert result.evidence_ids == ["doc-1"]
    assert result.rank_averaging_useful is True
    assert client.kwargs["model"] == "reasoning-model"

    system_prompt = str(client.kwargs["system_prompt"])
    assert "AUC/Gini -> ranking and rank averaging" in system_prompt
    assert "LogLoss -> calibration and clipping" in system_prompt
    assert "F1/Dice -> threshold search" in system_prompt
    assert "RMSE/RMSLE -> target transforms and clipping" in system_prompt
    assert "MAP@K/NDCG -> ranking and candidate generation" in system_prompt
    assert "Do not claim dataset was analyzed." in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["plan_data"]["metric"] == "Gini"
    assert "ID: doc-1" in payload["retrieved_documents"]
    assert "RRF score: 0.2500" in payload["retrieved_documents"]
    assert "expected_schema" in payload


@pytest.mark.asyncio
async def test_analyze_metric_rejects_unknown_evidence_ids() -> None:
    client = FakeClient(_metric_response(evidence_ids=["doc-1", "missing-doc"]))

    with pytest.raises(ValueError, match="unknown evidence_ids"):
        await analyze_metric(
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )
