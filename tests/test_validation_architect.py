from __future__ import annotations

import json

import pytest

from kaggle_researcher.reasoning.validation_architect import design_validation
from kaggle_researcher.schemas import PlanData, RetrievedDocument, ValidationResult


class FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    async def chat_json(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _plan(metric: str = "accuracy", domain: str = "image classification") -> PlanData:
    return PlanData(
        task_type="classification",
        metric=metric,
        domain=domain,
        kaggle_queries=["classification validation"],
    )


def _doc(doc_id: str = "doc-1", *, content: str = "Use grouped CV by patient id.") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Validation discussion",
        url="https://example.com/validation",
        content=content,
        score=0.7,
        rrf_score=0.2,
    )


def _validation_response(
    *,
    confidence: str = "medium",
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "confidence": confidence,
        "evidence_ids": evidence_ids or ["doc-1"],
        "recommended_cv": "GroupKFold by patient id",
        "validation_risk": "medium",
        "likely_split": "grouped by patient",
        "failure_modes": ["Same patient appearing in train and validation folds."],
        "reasoning": (
            "Fact: retrieved_documents mention patient grouping. "
            "Hypothesis: leakage risk comes from repeated patient ids. "
            "Recommendation: use group-aware validation."
        ),
        "primary_validation": {"method": "group_kfold", "reason": "Patient groups must remain isolated."},
    }


@pytest.mark.asyncio
async def test_design_validation_validates_mock_response_into_validation_result() -> None:
    client = FakeClient(_validation_response())

    result = await design_validation(
        competition_desc="Classify medical images.",
        plan_data=_plan(domain="medical imaging"),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert isinstance(result, ValidationResult)
    assert result.confidence == "medium"
    assert result.evidence_ids == ["doc-1"]
    assert result.recommended_cv == "GroupKFold by patient id"
    assert client.kwargs["model"] == "reasoning-model"

    system_prompt = str(client.kwargs["system_prompt"])
    assert "Recommend CV/split strategy only." in system_prompt
    assert "Do not propose models or feature engineering." in system_prompt
    assert "Separate facts, hypotheses, and recommendations" in system_prompt
    assert "Every source-backed claim must cite evidence_ids" in system_prompt
    assert "Keep confidence below high unless sources explicitly describe split, time, or group structure" in system_prompt
    assert "secondary_validation may be null" in system_prompt
    assert "Do not duplicate primary_validation" in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["competition_desc"] == "Classify medical images."
    assert payload["plan_data"]["domain"] == "medical imaging"
    assert "ID: doc-1" in payload["retrieved_documents"]
    assert "RRF score: 0.2000" in payload["retrieved_documents"]
    assert "expected_schema" in payload


def test_validation_result_distinguishes_optional_secondary_from_invalid_primary() -> None:
    payload = _validation_response()
    payload["secondary_validation"] = None

    result = ValidationResult.model_validate(payload)

    assert result.secondary_validation is None
    assert result.model_dump(mode="json")["secondary_validation"] is None

    missing_primary = dict(payload)
    missing_primary.pop("primary_validation")
    with pytest.raises(Exception):
        ValidationResult.model_validate(missing_primary)

    null_primary = dict(payload, primary_validation=None)
    with pytest.raises(Exception):
        ValidationResult.model_validate(null_primary)

    invalid_secondary = dict(payload, secondary_validation={})
    with pytest.raises(Exception):
        ValidationResult.model_validate(invalid_secondary)


def test_validation_result_accepts_typed_secondary_and_normalizes_known_null_collections() -> None:
    payload = _validation_response()
    payload.update({
        "secondary_validation": {"method": "time_holdout", "reason": "Robustness check."},
        "failure_modes": None,
        "do_not_use": None,
        "policy_notes": None,
    })

    result = ValidationResult.model_validate(payload)

    assert result.primary_validation.method == "group_kfold"
    assert result.secondary_validation is not None
    assert result.secondary_validation.method == "time_holdout"
    assert result.failure_modes == []
    assert result.do_not_use == []
    assert result.policy_notes == []


@pytest.mark.asyncio
async def test_design_validation_rejects_unknown_evidence_ids() -> None:
    client = FakeClient(_validation_response(evidence_ids=["doc-1", "missing-doc"]))

    with pytest.raises(ValueError, match="unknown evidence_ids"):
        await design_validation(
            competition_desc="Classify images.",
            plan_data=_plan(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )


@pytest.mark.asyncio
async def test_design_validation_demotes_high_confidence_without_split_sources() -> None:
    client = FakeClient(_validation_response(confidence="high"))

    result = await design_validation(
        competition_desc="Classify images.",
        plan_data=_plan(),
        retrieved_documents=[_doc(content="General baseline notebook with augmentation notes.")],
        client=client,
        model="reasoning-model",
    )

    assert result.confidence == "medium"
    assert any("Confidence reduced from high" in note for note in result.policy_notes)
