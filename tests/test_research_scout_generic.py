from __future__ import annotations

import pytest

from kaggle_researcher.eda.schemas import EdaTaskPlan, ResearchHypotheses
from kaggle_researcher.research_scout import run_research_scout
from kaggle_researcher.research_scout.schemas import ResearchScoutOutput
from kaggle_researcher.schemas import PlanData, RetrievedDocument


class FakeScoutClient:
    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response or {}
        self.exc = exc
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


class SequentialScoutClient(FakeScoutClient):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(responses[0])
        self.responses = responses

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]


@pytest.mark.asyncio
async def test_mock_llm_response_validates_and_writes_expected_objects(tmp_path) -> None:
    client = FakeScoutClient(
        {
            "hypotheses": [
                {
                    "hypothesis_id": "schema_001",
                    "category": "schema",
                    "claim": "Schema roles should be inferred generically.",
                    "rationale": "Schema roles are required for downstream checks.",
                    "expected_eda_checks": ["schema_inferer.roles"],
                    "priority": "P0",
                    "confidence_before_eda": "medium",
                    "source_refs": ["doc-1"],
                },
                {
                    "hypothesis_id": "metric_001",
                    "category": "metric",
                    "claim": "ROC AUC requires score predictions.",
                    "rationale": "Metric semantics determine model output format.",
                    "expected_eda_checks": ["metric_analyzer.registry"],
                    "priority": "P0",
                    "confidence_before_eda": "high",
                    "source_refs": ["doc-1"],
                },
            ],
            "eda_task_plan": {
                "recommended_module_sequence": [
                    "file_inventory",
                    "schema_inferer",
                    "metric_analyzer",
                    "validation_analyzer",
                    "leakage_checker",
                ],
                "recommended_human_checklist": ["Confirm metric from competition docs."],
            },
            "structured_findings": [
                {
                    "finding_id": "finding_001",
                    "category": "metric",
                    "finding": "Sources mention ROC AUC.",
                    "source_refs": ["doc-1"],
                }
            ],
            "summary": "# Research Scout Summary\n\nMock summary.\n",
        }
    )

    output = await run_research_scout(
        competition_id="generic-binary",
        competition_url="https://www.kaggle.com/competitions/generic-binary",
        competition_desc="Binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=client,
        model="deepseek-v4-pro",
    )
    paths = output.write_outputs(tmp_path)

    assert isinstance(output, ResearchScoutOutput)
    assert paths["research_hypotheses"].is_file()
    assert paths["eda_task_plan"].is_file()
    assert paths["research_scout_summary"].is_file()
    ResearchHypotheses(**output.to_research_hypotheses_payload())
    EdaTaskPlan(**output.to_eda_task_plan_payload())
    assert {"schema_001", "metric_001", "val_001", "leak_001"} <= {
        hypothesis.hypothesis_id for hypothesis in output.hypotheses
    }


@pytest.mark.asyncio
async def test_prompt_requires_expected_checks_and_no_default_temporal_policy() -> None:
    client = FakeScoutClient({"hypotheses": [], "eda_task_plan": {}})

    await run_research_scout(
        competition_id="generic-binary",
        competition_url="https://www.kaggle.com/competitions/generic-binary",
        competition_desc="Ordinary binary classification with tabular rows.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=client,
        model="deepseek-v4-pro",
    )

    prompt = client.calls[0]["user_prompt"].lower()
    system_prompt = client.calls[0]["system_prompt"].lower()
    assert "every hypothesis must include expected_eda_checks" in prompt
    assert "do not force temporal validation" in prompt
    assert "do not assume temporal validation by default" in system_prompt


@pytest.mark.asyncio
async def test_invalid_task_plan_receives_one_bounded_repair_attempt() -> None:
    client = SequentialScoutClient([
        {"hypotheses": [], "eda_task_plan": {"hypothesis_index": {"schema_001": "invalid"}}},
        {"hypotheses": [], "eda_task_plan": {}},
    ])

    output = await run_research_scout(
        competition_id="generic-binary",
        competition_url="https://www.kaggle.com/competitions/generic-binary",
        competition_desc="Ordinary binary classification with tabular rows.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=client,
    )

    assert len(client.calls) == 2
    assert "canonical_task_schema" in client.calls[1]["user_prompt"]
    assert output.eda_task_plan.hypothesis_index["schema_001"]


@pytest.mark.asyncio
async def test_fallback_output_includes_p0_hypotheses() -> None:
    output = await run_research_scout(
        competition_id="fallback-binary",
        competition_url="https://www.kaggle.com/competitions/fallback-binary",
        competition_desc="Binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=FakeScoutClient(exc=RuntimeError("llm unavailable")),
        model="deepseek-v4-pro",
    )

    by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in output.hypotheses}
    assert {"schema_001", "metric_001", "val_001", "leak_001"} <= set(by_id)
    assert all(
        by_id[hypothesis_id].priority == "P0"
        for hypothesis_id in ["schema_001", "metric_001", "val_001", "leak_001"]
    )
    assert output.models_used["fallback"] is True
    ResearchHypotheses(**output.to_research_hypotheses_payload())
    EdaTaskPlan(**output.to_eda_task_plan_payload())


@pytest.mark.asyncio
async def test_ordinary_binary_fallback_does_not_force_temporal_validation() -> None:
    output = await run_research_scout(
        competition_id="fallback-binary",
        competition_url="https://www.kaggle.com/competitions/fallback-binary",
        competition_desc="Ordinary iid binary classification with no time split described.",
        plan_data=_plan(),
        retrieved_documents=[],
        client=FakeScoutClient(exc=RuntimeError("llm unavailable")),
        model="deepseek-v4-pro",
    )

    validation = next(h for h in output.hypotheses if h.hypothesis_id == "val_001")
    assert "temporal validation is required" not in validation.claim.lower()
    assert "temporal" not in validation.claim.lower()
    assert not any("temporal" in check.lower() for check in validation.expected_eda_checks)


@pytest.mark.asyncio
async def test_every_hypothesis_has_expected_eda_checks() -> None:
    output = await run_research_scout(
        competition_id="generic-binary",
        competition_url="https://www.kaggle.com/competitions/generic-binary",
        competition_desc="Binary classification with ROC AUC.",
        plan_data=_plan(),
        retrieved_documents=[_doc()],
        client=FakeScoutClient(
            {
                "hypotheses": [
                    {
                        "id": "feat_001",
                        "category": "feature",
                        "claim": "Feature ideas may help.",
                        "rationale": "Needs EDA.",
                        "priority": "P1",
                    }
                ]
            }
        ),
        model="deepseek-v4-pro",
    )

    assert output.hypotheses
    assert all(hypothesis.expected_eda_checks for hypothesis in output.hypotheses)


def _plan() -> PlanData:
    return PlanData(
        task_type="binary_classification",
        metric="roc_auc",
        domain="generic_tabular",
        kaggle_queries=["binary classification auc notebook"],
        arxiv_queries=[],
        github_queries=[],
    )


def _doc() -> RetrievedDocument:
    return RetrievedDocument(
        id="doc-1",
        competition_id="generic-binary",
        source="kaggle",
        title="Generic AUC notebook",
        url="https://example.com/notebook",
        content="Notebook discusses StratifiedKFold and ROC AUC scores.",
        score=0.9,
        rrf_score=0.1,
    )
