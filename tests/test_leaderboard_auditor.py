from __future__ import annotations

import json

import pytest

from kaggle_researcher.reasoning.leaderboard_auditor import audit_leaderboard_risk
from kaggle_researcher.schemas import (
    LeaderboardAuditResult,
    PlanData,
    RetrievedDocument,
    ValidationResult,
)


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
        metric="gini",
        domain="credit risk",
        kaggle_queries=["credit risk leaderboard shakeup"],
    )


def _validation() -> ValidationResult:
    return ValidationResult(
        confidence="medium",
        evidence_ids=["doc-1"],
        recommended_cv="Out-of-time holdout plus rolling CV",
        validation_risk="high",
        likely_split="time",
        failure_modes=["Public LB may reward period-specific overfit."],
        reasoning="Temporal validation is recommended from source-backed split risk.",
        primary_validation={"method": "temporal_cv"},
    )


def _doc(doc_id: str = "doc-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Leaderboard discussion",
        url="https://example.com/lb",
        content="Participants warn that public leaderboard can be noisy and shake up privately.",
        score=0.78,
        rrf_score=0.31,
    )


def _audit_response(evidence_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "confidence": "medium",
        "evidence_ids": evidence_ids or ["doc-1"],
        "shake_up_risk": "high",
        "submission_selection_rule": "Select submissions by trusted CV, not public LB peaks.",
        "public_lb_trust": "low",
        "warnings": [
            "Public LB overfitting risk is high; behavior is not_verified_on_data."
        ],
    }


@pytest.mark.asyncio
async def test_audit_leaderboard_risk_validates_mock_response_into_result() -> None:
    client = FakeClient(_audit_response())

    result = await audit_leaderboard_risk(
        competition_desc="Predict credit default. Metric is Gini.",
        plan_data=_plan(),
        validation_result=_validation(),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert isinstance(result, LeaderboardAuditResult)
    assert result.confidence == "medium"
    assert result.evidence_ids == ["doc-1"]
    assert result.shake_up_risk == "high"
    assert result.public_lb_trust == "low"
    assert result.submission_selection_rule.startswith("Select submissions by trusted CV")
    assert result.warnings

    system_prompt = str(client.kwargs["system_prompt"])
    assert "Warn against public LB overfitting" in system_prompt
    assert "Do not claim actual LB/CV correlation was measured" in system_prompt
    assert "Use task_type and metric from plan_data" in system_prompt
    assert "use validation_result as context" in system_prompt
    assert "submission_selection_rule" in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["competition_desc"] == "Predict credit default. Metric is Gini."
    assert payload["plan_data"]["task_type"] == "binary_classification"
    assert payload["plan_data"]["metric"] == "gini"
    assert payload["validation_result"]["recommended_cv"] == "Out-of-time holdout plus rolling CV"
    assert "ID: doc-1" in payload["retrieved_documents"]
    assert "RRF score: 0.3100" in payload["retrieved_documents"]
    assert "expected_schema" in payload


@pytest.mark.asyncio
async def test_audit_leaderboard_risk_rejects_unknown_evidence_ids() -> None:
    client = FakeClient(_audit_response(evidence_ids=["doc-1", "missing-doc"]))

    with pytest.raises(ValueError, match="unknown evidence_ids"):
        await audit_leaderboard_risk(
            competition_desc="Predict credit default. Metric is Gini.",
            plan_data=_plan(),
            validation_result=_validation(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )


@pytest.mark.asyncio
async def test_audit_leaderboard_risk_rejects_measured_correlation_claims() -> None:
    response = _audit_response()
    response["warnings"] = ["Actual LB/CV correlation was measured as weak."]
    client = FakeClient(response)

    with pytest.raises(ValueError, match="measured-correlation claim"):
        await audit_leaderboard_risk(
            competition_desc="Predict credit default. Metric is Gini.",
            plan_data=_plan(),
            validation_result=_validation(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )
