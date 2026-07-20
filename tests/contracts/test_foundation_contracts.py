from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.artifacts import validate_contract_definitions
from kaggle_researcher.contracts.eda import EdaTask, EdaTaskPlan
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.contracts.validation import ValidationResult
from kaggle_researcher.contracts.ids import EdaTaskId, ExperimentId, HypothesisId
from kaggle_researcher.contracts.migration import migration_is_idempotent, migrate_research_hypotheses_payload
from kaggle_researcher.contracts.research import ResearchHypotheses
from kaggle_researcher.contracts.repair import validate_with_one_repair
from kaggle_researcher.contracts.versions import CURRENT_CONTRACT_VERSIONS
from kaggle_researcher.main import run
from kaggle_researcher.eda.schemas import EdaEvidencePack as EdaEvidencePackAlias
from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult as FinalStrategyAlias
from kaggle_researcher.schemas import ValidationResult as ValidationResultAlias


pytestmark = pytest.mark.contract


def test_runtime_id_namespaces_remain_json_strings() -> None:
    hypothesis = HypothesisId("hyp_001")
    experiment = ExperimentId("hyp_001")
    assert type(hypothesis) is not type(experiment)
    assert json.dumps({"id": hypothesis}) == '{"id": "hyp_001"}'


def test_compatibility_imports_are_class_identical() -> None:
    assert EdaEvidencePackAlias is EdaEvidencePack
    assert ValidationResultAlias is ValidationResult
    assert FinalStrategyAlias is FinalStrategyResult


def test_strict_contract_and_future_version_policy() -> None:
    with pytest.raises(ValidationError):
        ResearchHypotheses.model_validate({
            "competition_id": "demo", "hypotheses": [], "unexpected": True,
        })
    assert set(CURRENT_CONTRACT_VERSIONS) >= {"research_hypotheses", "final_strategy"}


def test_task_dependencies_reject_cycles() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        EdaTaskPlan(
            competition_id="demo",
            eda_tasks=[
                EdaTask(task_id="a", module="one", priority="P0", dependencies=["b"]),
                EdaTask(task_id="b", module="two", priority="P0", dependencies=["a"]),
            ],
        )
    assert isinstance(EdaTaskId("a"), str)


def test_research_migration_is_idempotent() -> None:
    payload = {
        "competition_id": "demo",
        "hypotheses": [{"id": "h1", "category": "dataset_schema", "claim": "Inspect schema."}],
    }
    assert migration_is_idempotent(payload, migrate_research_hypotheses_payload)


@pytest.mark.asyncio
async def test_validate_contracts_cli_is_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert await run(["validate-contracts"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert validate_contract_definitions()["status"] == "ok"


@pytest.mark.asyncio
async def test_shared_boundary_repair_is_bounded_to_one_attempt() -> None:
    calls: list[dict[str, object]] = []

    async def repair(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"competition_id": "demo", "hypotheses": []}

    result = await validate_with_one_repair(
        {"competition_id": None},
        model=ResearchHypotheses,
        repair=repair,
        contract_name="research_hypotheses",
    )
    assert result.repaired is True
    assert len(calls) == 1
    assert calls[0]["canonical_fields"]
