from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.artifacts import load_eda_task_plan, load_research_hypotheses
from kaggle_researcher.contracts.research_to_eda import validate_research_to_eda_contract
from kaggle_researcher.research_scout.schemas import ResearchScoutOutput
from kaggle_researcher.research_scout.scout import run_research_scout
from kaggle_researcher.schemas import PlanData


pytestmark = [pytest.mark.contract, pytest.mark.offline]


class FailingClient:
    calls = 0

    async def chat_json(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise RuntimeError("offline fixture failure")


@pytest.mark.parametrize(
    ("task_type", "metric"),
    [
        ("binary_classification", "roc_auc"),
        ("regression", "rmse"),
        ("ranking", "ndcg"),
        ("binary_classification", "gini_stability"),
        ("custom", "unknown_competition_metric"),
    ],
)
@pytest.mark.asyncio
async def test_deterministic_fallback_writes_eda_compatible_artifacts(
    tmp_path: Path, task_type: str, metric: str
) -> None:
    client = FailingClient()
    output = await run_research_scout(
        competition_id="fallback-fixture",
        competition_url="https://example.invalid/competition",
        competition_desc="Offline generic fixture.",
        plan_data=PlanData(
            task_type=task_type,
            metric=metric,
            domain="generic_tabular",
            kaggle_queries=[],
            arxiv_queries=[],
            github_queries=[],
        ),
        retrieved_documents=[],
        client=client,  # type: ignore[arg-type]
        model="offline-model",
    )
    paths = output.write_outputs(tmp_path)
    hypotheses, _ = load_research_hypotheses(paths["research_hypotheses"])
    plan, _ = load_eda_task_plan(paths["eda_task_plan"], hypotheses=hypotheses)
    result = validate_research_to_eda_contract(hypotheses, plan)
    assert result.valid
    assert output.models_used["fallback"] is True
    assert client.calls == 1

    validation = next(item for item in hypotheses.hypotheses if item.category == "validation")
    checks = set(validation.expected_eda_checks)
    if metric == "gini_stability":
        assert "validation_analyzer.temporal_cv_feasibility" in checks
    elif task_type not in {"ranking"}:
        assert not any("temporal" in check for check in checks)


def test_published_scout_model_rejects_raw_extras() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ResearchScoutOutput.model_validate({
            "competition_id": "x",
            "eda_task_plan": {"competition_id": "x"},
            "raw_llm_dump": "not part of the contract",
        })

