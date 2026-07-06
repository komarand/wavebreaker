from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kaggle_researcher import main as main_module
from kaggle_researcher.main import run_research
from kaggle_researcher.research_scout import (
    build_research_hypotheses,
    build_research_scout_summary,
    normalize_research_hypotheses,
    split_eda_task_plan,
    validate_research_hypotheses,
)
from kaggle_researcher.research_scout_schemas import (
    EdaTask,
    ResearchHypothesesPayload,
    ResearchHypothesis,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument, SourceDocument


def run(coro):
    return asyncio.run(coro)


def complete_payload() -> dict[str, Any]:
    return normalize_research_hypotheses(
        {
            "competition_id": "home-credit-credit-risk-model-stability",
            "competition_url": "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
            "competition_desc": "Binary classification. Metric: Gini Stability. Tabular credit data.",
            "task_type": "binary_classification",
            "metric": {"name": "gini_stability"},
            "domain": "tabular_credit_risk",
        }
    )


def test_research_hypothesis_schema_valid() -> None:
    hypothesis = ResearchHypothesis(
        id="val_001",
        category="validation",
        priority="P0",
        claim="Use out-of-time validation.",
        why_it_matters="Random CV may overestimate leaderboard performance.",
        how_to_verify=["Find candidate time columns."],
        provenance=["heuristic", "not_verified_on_data"],
        confidence="high",
    )

    assert hypothesis.id == "val_001"


def test_research_hypothesis_rejects_invalid_priority() -> None:
    with pytest.raises(ValidationError):
        ResearchHypothesis(
            id="val_001",
            category="validation",
            priority="P9",
            claim="Bad priority.",
            why_it_matters="It should fail.",
            how_to_verify=["Validate schema."],
            provenance=["heuristic"],
        )


def test_eda_task_schema_valid() -> None:
    task = EdaTask(
        id="eda_val_001",
        priority="P0",
        module="validation_analyzer",
        question="Which split is trustworthy?",
        rationale="Validation controls strategy.",
        expected_outputs=["validation_evidence.recommended_validation"],
    )

    assert task.module == "validation_analyzer"


def test_research_payload_schema_valid() -> None:
    payload = ResearchHypothesesPayload.model_validate(complete_payload())

    assert payload.schema_version == "1.0"


def test_normalize_adds_missing_ids() -> None:
    payload = normalize_research_hypotheses(
        {
            "competition_id": "comp",
            "competition_desc": "Credit risk.",
            "task_type": "binary_classification",
            "metric": "gini",
            "hypotheses": [
                {
                    "category": "validation",
                    "priority": "P0",
                    "claim": "Check validation.",
                    "why_it_matters": "It matters.",
                    "how_to_verify": ["Inspect folds."],
                    "provenance": ["heuristic"],
                }
            ],
        }
    )

    assert payload["hypotheses"][0]["id"] == "val_001"


def test_normalize_adds_not_verified_on_data() -> None:
    payload = complete_payload()

    assert all("not_verified_on_data" in item["provenance"] for item in payload["hypotheses"])


def test_normalize_adds_default_validation_for_stability_metric() -> None:
    payload = complete_payload()

    validation_claims = " ".join(
        item["claim"].lower()
        for item in payload["hypotheses"]
        if item["category"] == "validation"
    )
    assert "out-of-time" in validation_claims or "temporal" in validation_claims


def test_normalize_adds_default_leakage_checks() -> None:
    payload = complete_payload()

    leakage_claims = " ".join(
        item["claim"].lower()
        for item in payload["hypotheses"]
        if item["category"] == "leakage"
    )
    assert "id overlap" in leakage_claims
    assert "target-like" in leakage_claims
    assert "future" in leakage_claims


def test_normalize_ensures_p0_tasks_exist() -> None:
    payload = complete_payload()
    p0_ids = {
        item["id"]
        for item in payload["hypotheses"]
        if item["priority"] == "P0"
    }
    related_ids = {
        hypothesis_id
        for task in payload["eda_tasks"]
        for hypothesis_id in task["related_hypothesis_ids"]
    }

    assert p0_ids <= related_ids


def test_validate_research_hypotheses_accepts_complete_payload() -> None:
    validate_research_hypotheses(complete_payload())


def test_validate_research_hypotheses_rejects_no_validation_hypothesis() -> None:
    payload = complete_payload()
    payload["hypotheses"] = [
        item for item in payload["hypotheses"] if item["category"] != "validation"
    ]

    with pytest.raises(ValueError, match="P0 validation"):
        validate_research_hypotheses(payload)


def test_validate_research_hypotheses_rejects_no_leakage_hypothesis() -> None:
    payload = complete_payload()
    payload["hypotheses"] = [
        item for item in payload["hypotheses"] if item["category"] != "leakage"
    ]

    with pytest.raises(ValueError, match="P0 leakage"):
        validate_research_hypotheses(payload)


def test_validate_research_hypotheses_requires_temporal_for_stability_metric() -> None:
    payload = complete_payload()
    for item in payload["hypotheses"]:
        if item["category"] == "validation":
            item["claim"] = "Use stratified validation."

    with pytest.raises(ValueError, match="temporal"):
        validate_research_hypotheses(payload)


def test_build_research_scout_summary_contains_p0() -> None:
    summary = build_research_scout_summary(complete_payload())

    assert "## P0 EDA checks" in summary
    assert "[P0]" in summary


def test_build_research_scout_summary_contains_limitations() -> None:
    summary = build_research_scout_summary(complete_payload())

    assert "No real EDA was performed" in summary


def test_split_eda_task_plan_is_focused_subset() -> None:
    plan = split_eda_task_plan(complete_payload())

    assert "eda_tasks" in plan
    assert "hypothesis_index" in plan
    assert "hypotheses" not in plan


def test_build_research_hypotheses_uses_mocked_llm_response() -> None:
    class FakeClient:
        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["model"] == "deepseek-v4-pro"
            return {
                "hypotheses": [
                    {
                        "category": "validation",
                        "priority": "P0",
                        "claim": "Identify WEEK_NUM for temporal validation.",
                        "why_it_matters": "Stability scoring depends on time.",
                        "how_to_verify": ["Inspect columns."],
                        "provenance": ["kaggle"],
                        "confidence": "high",
                    }
                ],
                "eda_tasks": [],
            }

    payload = run(
        build_research_hypotheses(
            competition_id="home-credit-credit-risk-model-stability",
            competition_url="https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
            competition_desc="Metric: Gini Stability. Tabular credit data.",
            plan_data={"task_type": "binary_classification", "metric": "gini_stability", "domain": "tabular_credit_risk"},
            retrieved_documents=[],
            client=FakeClient(),
            model="deepseek-v4-pro",
        )
    )

    assert payload["models_used"]["research_scout"] == "deepseek-v4-pro"
    validate_research_hypotheses(payload)


def test_scout_mode_writes_artifacts_and_skips_docx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeSettings:
        deepseek_api_key = "secret"
        deepseek_v4_pro = "deepseek-v4-pro"
        deepseek_v4_flash = "deepseek-v4-flash"
        embed_model = "fake-embedder"
        embed_dim = 2
        max_embed_batch_size = 2
        pg_dsn = "postgresql://example"
        top_k = 2
        max_notebooks = 1
        max_papers = 1
        max_repos = 1
        pdf_cache_dir = "./data/pdfs"
        github_token = None

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            return {"hypotheses": [], "eda_tasks": []}

    class FakeStore:
        def __init__(self, competition_id: str, dsn: str, embed_dim: int) -> None:
            self.competition_id = competition_id

        async def init(self) -> None:
            return None

        async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
            return None

        async def close(self) -> None:
            return None

    class FakeDomainMemory:
        def __init__(self, dsn: str, embed_dim: int) -> None:
            return None

        async def init(self) -> None:
            return None

        async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
            return []

        async def close(self) -> None:
            return None

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="binary_classification",
            metric="gini_stability",
            domain="tabular_credit_risk",
            kaggle_queries=["credit"],
            arxiv_queries=[],
            github_queries=[],
        )

    async def fake_collect_sources(**kwargs: Any) -> list[SourceDocument]:
        return [
            SourceDocument(
                id="doc-1",
                competition_id="comp-1",
                source="kaggle",
                title="Notebook",
                url="https://example.com/notebook",
                content="Uses WEEK_NUM and Gini stability.",
                summary="Uses WEEK_NUM and Gini stability.",
            )
        ]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return kwargs["docs"]

    async def fake_retrieve_documents(**kwargs: Any) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id="doc-1",
                competition_id="comp-1",
                source="kaggle",
                title="Notebook",
                url="https://example.com/notebook",
                content="Uses WEEK_NUM and Gini stability.",
                score=0.9,
                rrf_score=0.1,
            )
        ]

    def fail_generate_report(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("DOCX report should not be generated in scout mode")

    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", FakeDomainMemory)
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(main_module, "_collect_sources", fake_collect_sources)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2]])
    monkeypatch.setattr(main_module, "_retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(main_module, "generate_report", fail_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/comp-1",
            "Metric: Gini Stability. Tabular credit data.",
            mode="scout",
            show_progress=False,
        )
    )

    run_path = Path(result.run_artifacts_path)
    assert result.mode == "scout"
    assert Path(result.research_hypotheses_path).exists()
    assert Path(result.eda_task_plan_path).exists()
    assert Path(result.summary_path).exists()
    assert not any(path.suffix == ".docx" for path in tmp_path.rglob("*"))
    payload = json.loads((run_path / "research_hypotheses.json").read_text(encoding="utf-8"))
    assert any(item["category"] == "validation" and item["priority"] == "P0" for item in payload["hypotheses"])
    assert any(item["category"] == "leakage" and item["priority"] == "P0" for item in payload["hypotheses"])
    assert any(task["module"] == "schema_inferer" for task in payload["eda_tasks"])
    assert any(task["module"] == "validation_analyzer" for task in payload["eda_tasks"])
    assert any(task["module"] == "leakage_checker" for task in payload["eda_tasks"])
    assert any(task["module"] == "metric_analyzer" for task in payload["eda_tasks"])
    assert any(task["module"] == "baseline_runner" for task in payload["eda_tasks"])
    assert payload["models_used"]["research_scout"] == "deepseek-v4-pro"
