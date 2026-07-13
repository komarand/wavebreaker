from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kaggle_researcher.reasoning.common import (
    CANONICAL_REASONING_EVIDENCE_IDS,
    known_evidence_ids,
)
from kaggle_researcher.reasoning.experiment_planner import (
    known_experiment_evidence_ids,
    normalize_evidence_ids,
    plan_experiments,
)
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeakageRiskResult,
    MetricResult,
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


class SequentialClient(FakeClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        super().__init__(responses[0])
        self.responses = responses
        self.calls = 0

    async def chat_json(self, **kwargs):
        self.kwargs = kwargs
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _validation() -> ValidationResult:
    return ValidationResult(
        confidence="medium",
        evidence_ids=["doc-1"],
        recommended_cv="Out-of-time holdout plus rolling CV",
        validation_risk="high",
        likely_split="time",
        failure_modes=["Public/private distribution shift."],
        reasoning="Use honest temporal validation before trusting improvements.",
        primary_validation={"method": "temporal_cv"},
    )


def _leakage() -> LeakageRiskResult:
    return LeakageRiskResult(
        confidence="medium",
        evidence_ids=["doc-1"],
        risk_level="medium",
        possible_issues=["Possible timestamp leakage risk; not_verified_on_data."],
        recommended_checks=["Inspect timestamp order and target availability."],
    )


def _metric() -> MetricResult:
    return MetricResult(
        confidence="medium",
        evidence_ids=["doc-1"],
        metric_explanation="Gini rewards ranking quality.",
        needs_calibration=False,
        rank_averaging_useful=True,
        threshold_search_needed=False,
        surrogate_loss_suggestion="Optimize AUC-like ranking with Gini validation.",
    )


def _doc(doc_id: str = "doc-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="comp-1",
        source="kaggle",
        title="Experiment discussion",
        url="https://example.com/experiments",
        content="Start with honest CV and a baseline before costly ensembling.",
        score=0.8,
        rrf_score=0.4,
    )


def _item(priority: str, experiment: str, *, evidence_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "priority": priority,
        "experiment": experiment,
        "why": "High ROI next step based on validation, leakage, and metric context.",
        "cost": "low",
        "expected_gain": "medium",
        "risk": "May not improve if source hypothesis is weak.",
        "evidence_ids": evidence_ids or ["doc-1"],
    }


@pytest.mark.asyncio
async def test_plan_experiments_validates_mock_response_and_sorts_by_priority() -> None:
    client = FakeClient(
        {
            "experiments": [
                _item("P2", "Tune rank-averaging ensemble"),
                _item("P0", "Build baseline model on honest validation"),
                _item("P1", "Run timestamp leakage checks"),
            ]
        }
    )

    result = await plan_experiments(
        validation_result=_validation(),
        leakage_result=_leakage(),
        metric_result=_metric(),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    assert [item.priority for item in result] == ["P0", "P1", "P2"]
    assert all(isinstance(item, ExperimentItem) for item in result)
    assert result[0].experiment == "Build baseline model on honest validation"

    system_prompt = str(client.kwargs["system_prompt"])
    assert "Build a prioritized experiment queue with ROI logic" in system_prompt
    assert "Priorities must be P0, P1, P2, or P3" in system_prompt
    assert "P0 should include honest validation and a baseline" in system_prompt
    assert "Do not present EDA, adversarial validation, leakage checks, or leakage detection as already executed" in system_prompt

    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["validation_result"]["recommended_cv"] == "Out-of-time holdout plus rolling CV"
    assert payload["leakage_result"]["risk_level"] == "medium"
    assert payload["metric_result"]["metric_explanation"] == "Gini rewards ranking quality."
    assert "ID: doc-1" in payload["retrieved_documents"]
    assert "RRF score: 0.4000" in payload["retrieved_documents"]
    assert "expected_schema" in payload


@pytest.mark.asyncio
async def test_plan_experiments_invalid_priority_fails_validation() -> None:
    client = FakeClient({"experiments": [_item("P4", "Invalid priority experiment")]})

    with pytest.raises(ValidationError):
        await plan_experiments(
            validation_result=_validation(),
            leakage_result=_leakage(),
            metric_result=_metric(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )


@pytest.mark.asyncio
async def test_plan_experiments_rejects_unknown_evidence_ids() -> None:
    client = FakeClient({"experiments": [_item("P1", "Run leakage checks", evidence_ids=["missing-doc"])]})

    with pytest.raises(ValueError, match="unknown evidence_ids"):
        await plan_experiments(
            validation_result=_validation(),
            leakage_result=_leakage(),
            metric_result=_metric(),
            retrieved_documents=[_doc()],
            client=client,
            model="reasoning-model",
        )


@pytest.mark.asyncio
async def test_plan_experiments_normalizes_validation_alias_and_exposes_registry() -> None:
    client = FakeClient({"experiments": [_item("P1", "Compare validation folds", evidence_ids=["validation_policy"])]})

    result = await plan_experiments(
        validation_result=_validation(), leakage_result=_leakage(), metric_result=_metric(),
        retrieved_documents=[_doc()], client=client, model="reasoning-model",
    )

    planned = next(item for item in result if item.experiment == "Compare validation folds")
    assert planned.evidence_ids == ["validation_result"]
    payload = json.loads(str(client.kwargs["user_prompt"]))
    assert payload["allowed_evidence_ids"] == known_experiment_evidence_ids([_doc()])
    assert payload["allowed_evidence_ids"] == known_evidence_ids(
        [_doc()], additional_ids=CANONICAL_REASONING_EVIDENCE_IDS,
    )
    assert "Every evidence_ids entry must exactly match" in str(client.kwargs["system_prompt"])
    assert "Do not invent aliases" in str(client.kwargs["system_prompt"])


def test_normalize_evidence_ids_deduplicates_known_aliases_without_fuzzy_matching() -> None:
    assert normalize_evidence_ids(["validation_result"]) == ["validation_result"]
    assert normalize_evidence_ids(["validation_policy", "primary_validation", "validation_result"]) == ["validation_result"]
    assert normalize_evidence_ids(["doc-1", "validation_result"]) == ["doc-1", "validation_result"]
    assert normalize_evidence_ids(["validatoin_policy"]) == ["validatoin_policy"]


@pytest.mark.asyncio
async def test_plan_experiments_repairs_unknown_ids_once_then_fails_hard() -> None:
    client = SequentialClient([
        {"experiments": [_item("P1", "Check validation", evidence_ids=["validatoin_policy"])]},
        {"experiments": [_item("P1", "Check validation", evidence_ids=["still_unknown"])]},
    ])

    with pytest.raises(ValueError, match="final stage"):
        await plan_experiments(
            validation_result=_validation(), leakage_result=_leakage(), metric_result=_metric(),
            retrieved_documents=[_doc()], client=client, model="reasoning-model",
        )
    assert client.calls == 2


@pytest.mark.asyncio
async def test_plan_experiments_accepts_single_repair_to_canonical_id() -> None:
    client = SequentialClient([
        {"experiments": [_item("P1", "Check validation", evidence_ids=["made_up_evidence"])]},
        {"experiments": [_item("P1", "Check validation", evidence_ids=["validation_result"])]},
    ])

    result = await plan_experiments(
        validation_result=_validation(), leakage_result=_leakage(), metric_result=_metric(),
        retrieved_documents=[_doc()], client=client, model="reasoning-model",
    )

    planned = next(item for item in result if item.experiment == "Check validation")
    assert planned.evidence_ids == ["validation_result"]
    assert client.calls == 2


@pytest.mark.asyncio
async def test_plan_experiments_adds_missing_required_p0_items() -> None:
    client = FakeClient({"experiments": [_item("P2", "Try rank-averaging ensemble")]})

    result = await plan_experiments(
        validation_result=_validation(),
        leakage_result=_leakage(),
        metric_result=_metric(),
        retrieved_documents=[_doc()],
        client=client,
        model="reasoning-model",
    )

    p0_experiments = [item.experiment.lower() for item in result if item.priority == "P0"]
    assert any("validation" in experiment for experiment in p0_experiments)
    assert any("baseline" in experiment for experiment in p0_experiments)
    assert [item.priority for item in result] == ["P0", "P0", "P2"]
