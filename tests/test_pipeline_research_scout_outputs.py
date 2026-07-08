from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_researcher import main as main_module
from kaggle_researcher.eda.schemas import EdaTaskPlan, ResearchHypotheses
from kaggle_researcher.main import run_research
from kaggle_researcher.research_scout.schemas import (
    EdaTaskPlanDraft,
    ResearchScoutOutput,
    ScoutEdaTask,
    ScoutHypothesis,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument, SourceDocument


@dataclass(slots=True)
class FakeSettings:
    deepseek_api_key: str = "secret"
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    embed_model: str = "fake-embedder"
    embed_dim: int = 2
    max_embed_batch_size: int = 2
    pg_dsn: str = "postgresql://example"
    top_k: int = 2
    max_notebooks: int = 1
    max_papers: int = 1
    max_repos: int = 1
    pdf_cache_dir: str = "./data/pdfs"
    github_token: str | None = None


class FakeClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


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

    async def close(self) -> None:
        return None


def run(coro):
    return asyncio.run(coro)


def test_parser_accepts_write_eda_plan_flag() -> None:
    args = main_module.build_parser().parse_args(
        [
            "https://www.kaggle.com/competitions/generic-binary",
            "Generic binary classification.",
            "--write-eda-plan",
        ]
    )

    assert args.write_eda_plan is True


def test_pipeline_writes_research_scout_outputs_when_enabled(monkeypatch, tmp_path: Path) -> None:
    scout_calls: list[dict[str, Any]] = []

    async def fake_run_research_scout(**kwargs: Any) -> ResearchScoutOutput:
        scout_calls.append(kwargs)
        return _scout_output(
            competition_id=kwargs["competition_id"],
            competition_url=kwargs["competition_url"],
        )

    _patch_pipeline(monkeypatch, tmp_path, scout_runner=fake_run_research_scout)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/generic-binary",
            "Generic iid binary classification with ROC AUC.",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
        )
    )

    run_path = Path(result.run_artifacts_path)
    hypotheses_path = run_path / "research_hypotheses.json"
    task_plan_path = run_path / "eda_task_plan.json"
    summary_path = run_path / "research_scout_summary.md"
    research_run_payload = json.loads((run_path / "research_run.json").read_text(encoding="utf-8"))

    assert len(scout_calls) == 1
    assert scout_calls[0]["plan_data"].task_type == "binary_classification"
    assert [doc.id for doc in scout_calls[0]["retrieved_documents"]] == ["retrieved-1"]
    assert hypotheses_path.is_file()
    assert task_plan_path.is_file()
    assert summary_path.is_file()
    assert result.research_hypotheses_path == str(hypotheses_path)
    assert result.eda_task_plan_path == str(task_plan_path)
    assert result.summary_path == str(summary_path)
    assert research_run_payload["research_hypotheses_path"] == str(hypotheses_path)
    assert research_run_payload["eda_task_plan_path"] == str(task_plan_path)
    assert research_run_payload["summary_path"] == str(summary_path)
    assert json.loads((Path(result.report_path).parent / "research_run.json").read_text(encoding="utf-8")) == research_run_payload

    hypotheses_payload = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    task_plan_payload = json.loads(task_plan_path.read_text(encoding="utf-8"))
    ResearchHypotheses(**hypotheses_payload)
    EdaTaskPlan(**task_plan_payload)
    assert task_plan_payload["task_type"] == "binary_classification"
    assert task_plan_payload["metric"]["name"] == "roc_auc"
    validation_claims = [
        item["claim"]
        for item in hypotheses_payload["hypotheses"]
        if item["category"] == "validation"
    ]
    assert any("StratifiedKFold" in claim for claim in validation_claims)
    assert all("temporal" not in claim.lower() for claim in validation_claims)


def test_pipeline_does_not_write_research_scout_outputs_by_default(monkeypatch, tmp_path: Path) -> None:
    async def fail_run_research_scout(**kwargs: Any) -> ResearchScoutOutput:
        raise AssertionError("Research Scout should only run behind --write-eda-plan")

    _patch_pipeline(monkeypatch, tmp_path, scout_runner=fail_run_research_scout)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/generic-binary",
            "Generic iid binary classification with ROC AUC.",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
        )
    )

    run_path = Path(result.run_artifacts_path)
    research_run_payload = json.loads((run_path / "research_run.json").read_text(encoding="utf-8"))

    assert not (run_path / "research_hypotheses.json").exists()
    assert not (run_path / "eda_task_plan.json").exists()
    assert not (run_path / "research_scout_summary.md").exists()
    assert result.research_hypotheses_path is None
    assert result.eda_task_plan_path is None
    assert result.summary_path is None
    assert research_run_payload["research_hypotheses_path"] is None
    assert research_run_payload["eda_task_plan_path"] is None
    assert research_run_payload["summary_path"] is None


def _patch_pipeline(monkeypatch, tmp_path: Path, *, scout_runner: Any) -> None:
    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return _plan()

    async def fake_collect_sources(**kwargs: Any) -> list[SourceDocument]:
        return [
            SourceDocument(
                id="source-1",
                competition_id=kwargs["competition_id"],
                source="kaggle",
                title="Generic notebook",
                url="https://example.com/notebook",
                content="Uses generic tabular ROC AUC validation.",
            )
        ]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return kwargs["docs"]

    async def fake_retrieve_documents(**kwargs: Any) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id="retrieved-1",
                competition_id=kwargs["competition_id"],
                source="kaggle",
                title="Retrieved generic evidence",
                url="https://example.com/evidence",
                content="For ordinary iid binary classification, stratified folds are appropriate.",
                score=0.9,
                rrf_score=0.2,
            )
        ]

    def fake_generate_report(
        competition_name: str,
        roadmap_text: str,
        sources: list[RetrievedDocument],
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("docx placeholder", encoding="utf-8")
        return path

    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", FakeDomainMemory)
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(main_module, "_collect_sources", fake_collect_sources)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(main_module, "_retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(main_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(main_module, "run_research_scout", scout_runner)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)


def _plan() -> PlanData:
    return PlanData(
        task_type="binary_classification",
        metric="roc_auc",
        domain="generic_tabular",
        kaggle_queries=["generic roc auc"],
        arxiv_queries=[],
        github_queries=[],
    )


def _scout_output(*, competition_id: str, competition_url: str) -> ResearchScoutOutput:
    hypotheses = [
        ScoutHypothesis(
            hypothesis_id="schema_001",
            category="schema",
            claim="Infer table roles and target columns before downstream checks.",
            rationale="Generic EDA needs schema facts before task-specific analysis.",
            expected_eda_checks=["schema_inferer.roles"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="metric_001",
            category="metric",
            claim="ROC AUC should be evaluated from continuous positive-class scores.",
            rationale="Metric semantics determine prediction format.",
            expected_eda_checks=["metric_analyzer.resolve_metric"],
            priority="P0",
            confidence_before_eda="high",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="val_001",
            category="validation",
            claim="Use StratifiedKFold for ordinary iid binary classification unless data evidence suggests otherwise.",
            rationale="Class balance should be preserved without assuming time order.",
            expected_eda_checks=["validation_analyzer.select_strategy"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="leak_001",
            category="leakage",
            claim="Check for target proxies and train-test schema mismatches before modeling.",
            rationale="Leakage risk is dataset-dependent and must be measured.",
            expected_eda_checks=["leakage_checker.target_proxy_scan"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="drift_001",
            category="drift",
            claim="Measure feature drift between train and test files before trusting validation.",
            rationale="Distribution shift affects generic tabular validation reliability.",
            expected_eda_checks=["drift_analyzer.feature_shift"],
            priority="P1",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
    ]
    tasks = [
        ScoutEdaTask(
            task_id="task_schema",
            module="schema_inferer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["schema_001"],
        ),
        ScoutEdaTask(
            task_id="task_metric",
            module="metric_analyzer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["metric_001"],
        ),
        ScoutEdaTask(
            task_id="task_validation",
            module="validation_analyzer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["val_001"],
        ),
        ScoutEdaTask(
            task_id="task_leakage",
            module="leakage_checker",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["leak_001"],
        ),
    ]
    return ResearchScoutOutput(
        competition_id=competition_id,
        competition_url=competition_url,
        task_type="binary_classification",
        metric={"name": "roc_auc"},
        dataset={},
        hypotheses=hypotheses,
        eda_task_plan=EdaTaskPlanDraft(
            competition_id=competition_id,
            task_type="binary_classification",
            metric={"name": "roc_auc"},
            eda_tasks=tasks,
            hypothesis_index={
                "schema_001": ["task_schema"],
                "metric_001": ["task_metric"],
                "val_001": ["task_validation"],
                "leak_001": ["task_leakage"],
            },
            recommended_module_sequence=[
                "schema_inferer",
                "metric_analyzer",
                "validation_analyzer",
                "leakage_checker",
            ],
            recommended_human_checklist=["Confirm metric and target column from competition metadata."],
            blocking_tasks=["schema_inferer", "validation_analyzer", "leakage_checker"],
        ),
        models_used={"research_scout": "deepseek-v4-pro"},
        summary="# Research Scout Summary\n\nGeneric binary classification EDA plan.\n",
    )
