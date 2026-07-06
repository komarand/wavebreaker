from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_researcher import main as main_module
from kaggle_researcher.main import run_research
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    RetrievedDocument,
    ReviewResult,
    SourceDocument,
    ValidationResult,
)


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
    instances: list["FakeStore"] = []

    def __init__(self, competition_id: str, dsn: str, embed_dim: int) -> None:
        self.competition_id = competition_id
        self.dsn = dsn
        self.embed_dim = embed_dim
        self.initialized = False
        self.closed = False
        self.upserted_docs: list[SourceDocument] = []
        self.upserted_embeddings: list[list[float]] = []
        FakeStore.instances.append(self)

    async def init(self) -> None:
        self.initialized = True

    async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
        self.upserted_docs = docs
        self.upserted_embeddings = embeddings

    async def close(self) -> None:
        self.closed = True


class FakeDomainMemory:
    instances: list["FakeDomainMemory"] = []

    def __init__(self, dsn: str, embed_dim: int) -> None:
        self.dsn = dsn
        self.embed_dim = embed_dim
        self.initialized = False
        self.closed = False
        self.calls: list[dict[str, object]] = []
        FakeDomainMemory.instances.append(self)

    async def init(self) -> None:
        self.initialized = True

    async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls.append({"task_type": task_type, "domain": domain, "top_k": top_k})
        return [{"competition_family": "credit_risk_tabular", "typical_models": ["LightGBM"]}]

    async def close(self) -> None:
        self.closed = True


def _source(doc_id: str, source: str, competition_id: str = "comp-1") -> SourceDocument:
    return SourceDocument(
        id=doc_id,
        competition_id=competition_id,
        source=source,
        title=f"{source} source",
        url="https://example.com/source",
        content="source content with useful competition notes",
    )


def _retrieved(doc_id: str = "doc-1", competition_id: str = "comp-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id=competition_id,
        source="kaggle",
        title="Retrieved evidence",
        url="https://example.com/evidence",
        content="retrieved evidence",
        score=0.9,
        rrf_score=0.2,
    )


def _roadmap() -> str:
    return "\n\n".join(
        f"## {heading}\n"
        "Confidence: medium. _Provenance: Kaggle + heuristic; not verified on data._ "
        + ("Detailed guidance. " * 25)
        for heading in SECTION_HEADINGS
    )


def run(coro):
    return asyncio.run(coro)


def test_full_mocked_pipeline_completes_and_records_github_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeStore.instances = []
    FakeDomainMemory.instances = []
    stage_calls: list[str] = []

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="classification",
            metric="gini",
            domain="credit",
            kaggle_queries=["credit kaggle"],
            arxiv_queries=["credit paper"],
            github_queries=["credit repo"],
        )

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return [doc.model_copy(update={"summary": f"summary {doc.id}"}) for doc in kwargs["docs"]]

    async def fake_search_paper_sources(**kwargs: Any) -> list[dict[str, object]]:
        return []

    async def fake_search_repos(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        raise RuntimeError("github unavailable")

    async def fake_hybrid_search(store: FakeStore, query: str, top_k: int) -> list[RetrievedDocument]:
        return [_retrieved("retrieved-1", store.competition_id)]

    async def fake_analyze_metric(**kwargs: Any) -> MetricResult:
        stage_calls.append("metric")
        return MetricResult(
            confidence="medium",
            evidence_ids=["retrieved-1"],
            metric_explanation="gini",
            needs_calibration=False,
            rank_averaging_useful=True,
            threshold_search_needed=False,
            surrogate_loss_suggestion="auc",
        )

    async def fake_design_validation(**kwargs: Any) -> ValidationResult:
        stage_calls.append("validation")
        return ValidationResult(
            confidence="medium",
            evidence_ids=["retrieved-1"],
            recommended_cv="temporal",
            validation_risk="high",
            likely_split="time",
            reasoning="reason",
        )

    async def fake_analyze_leakage_risk(**kwargs: Any) -> LeakageRiskResult:
        stage_calls.append("leakage")
        return LeakageRiskResult(
            confidence="low",
            evidence_ids=["retrieved-1"],
            risk_level="medium",
        )

    async def fake_audit_leaderboard_risk(**kwargs: Any) -> LeaderboardAuditResult:
        stage_calls.append("leaderboard")
        return LeaderboardAuditResult(
            confidence="medium",
            evidence_ids=["retrieved-1"],
            shake_up_risk="high",
            submission_selection_rule="cv",
            public_lb_trust="low",
        )

    async def fake_plan_experiments(**kwargs: Any) -> list[ExperimentItem]:
        stage_calls.append("experiments")
        return [
            ExperimentItem(
                priority="P0",
                experiment="baseline",
                why="why",
                cost="low",
                expected_gain="medium",
                risk="low",
                evidence_ids=["retrieved-1"],
            )
        ]

    async def fake_review(**kwargs: Any) -> ReviewResult:
        stage_calls.append("review")
        return ReviewResult(confidence="medium", evidence_ids=["retrieved-1"])

    async def fake_compose_report(**kwargs: Any) -> str:
        stage_calls.append("report")
        assert kwargs["domain_patterns"][0]["competition_family"] == "credit_risk_tabular"
        return _roadmap()

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
    monkeypatch.setattr(main_module, "search_notebooks", lambda *args, **kwargs: [{"kernel_ref": "u/k", "content": "nb"}])
    monkeypatch.setattr(main_module, "build_kaggle_documents", lambda raw, competition_id: [_source("kaggle-1", "kaggle", competition_id)])
    monkeypatch.setattr(main_module, "search_arxiv", lambda *args, **kwargs: [{"id": "paper"}])
    monkeypatch.setattr(main_module, "enrich_with_pdf", lambda papers, cache_dir: papers)
    monkeypatch.setattr(main_module, "build_arxiv_documents", lambda raw, competition_id: [_source("arxiv-1", "arxiv", competition_id)] if raw else [])
    monkeypatch.setattr(main_module, "search_paper_sources", fake_search_paper_sources)
    monkeypatch.setattr(main_module, "search_repos", fake_search_repos)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(main_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main_module, "analyze_metric", fake_analyze_metric)
    monkeypatch.setattr(main_module, "design_validation", fake_design_validation)
    monkeypatch.setattr(main_module, "analyze_leakage_risk", fake_analyze_leakage_risk)
    monkeypatch.setattr(main_module, "audit_leaderboard_risk", fake_audit_leaderboard_risk)
    monkeypatch.setattr(main_module, "plan_experiments", fake_plan_experiments)
    monkeypatch.setattr(main_module, "review", fake_review)
    monkeypatch.setattr(main_module, "compose_report", fake_compose_report)
    monkeypatch.setattr(main_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/comp-1",
            "Credit risk competition",
            output_dir=tmp_path,
            show_progress=False,
        )
    )

    assert Path(result.report_path).exists()
    assert result.report_mode == "full"
    assert result.num_documents == 2
    assert any("GitHub source collection failed: github unavailable" in warning for warning in result.warnings)
    assert FakeStore.instances[0].initialized is True
    assert FakeStore.instances[0].closed is True
    assert FakeDomainMemory.instances[0].initialized is True
    assert FakeDomainMemory.instances[0].closed is True
    assert FakeDomainMemory.instances[0].calls == [{"task_type": "classification", "domain": "credit", "top_k": 5}]
    assert stage_calls == ["metric", "validation", "leakage", "leaderboard", "experiments", "review", "report"]

    run_path = Path(result.run_artifacts_path)
    research_run_payload = json.loads((run_path / "research_run.json").read_text(encoding="utf-8"))
    assert research_run_payload["competition_id"] == "comp-1"
    assert research_run_payload["report_path"] == result.report_path
    assert research_run_payload["plan_data"]["metric"] == "gini"
    assert research_run_payload["retrieved_document_ids"] == ["retrieved-1"]
    assert research_run_payload["reasoning_outputs_summary"]["metric"]["confidence"] == "medium"
    assert research_run_payload["reasoning_outputs_summary"]["experiments"]["count"] == 1
    assert json.loads((Path(result.report_path).parent / "research_run.json").read_text(encoding="utf-8")) == research_run_payload
    assert json.loads((run_path / "domain_patterns.json").read_text(encoding="utf-8"))[0]["competition_family"] == "credit_risk_tabular"


def test_no_train_test_dataset_execution_functions_exist() -> None:
    source = Path(main_module.__file__).read_text(encoding="utf-8")
    forbidden_names = [
        "download_train",
        "download_test",
        "load_train",
        "load_test",
        "run_eda",
        "execute_notebook",
        "run_adversarial_validation",
    ]

    assert all(name not in source for name in forbidden_names)
