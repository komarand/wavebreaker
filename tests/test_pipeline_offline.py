from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_researcher import main as main_module
from kaggle_researcher.main import run_research
from kaggle_researcher.quality import (
    validate_reasoning_outputs,
    validate_report_text,
    validate_retrieved_documents,
)
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS
from kaggle_researcher.schemas import ResearchRunResult, RetrievedDocument, SourceDocument
from tests.fixtures import offline_pipeline as fx


@dataclass(slots=True)
class FakeSettings:
    deepseek_api_key: str = "offline-secret"
    deepseek_v4_pro: str = "offline-pro"
    deepseek_v4_flash: str = "offline-flash"
    embed_model: str = "offline-embedder"
    embed_dim: int = 2
    max_embed_batch_size: int = 3
    pg_dsn: str = "postgresql://offline"
    top_k: int = 3
    max_notebooks: int = 3
    max_papers: int = 2
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
        self.upserted_docs: list[SourceDocument] = []
        self.upserted_embeddings: list[list[float]] = []
        self.closed = False
        FakeStore.instances.append(self)

    async def init(self) -> None:
        pass

    async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
        self.upserted_docs = docs
        self.upserted_embeddings = embeddings

    async def close(self) -> None:
        self.closed = True


class FakeDomainMemory:
    instances: list["FakeDomainMemory"] = []

    def __init__(self, dsn: str, embed_dim: int) -> None:
        self.closed = False
        FakeDomainMemory.instances.append(self)

    async def init(self) -> None:
        pass

    async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "competition_family": "credit_risk_tabular",
                "typical_models": ["LightGBM", "CatBoost"],
                "validation_pattern": "stable stratified folds",
            }
        ]

    async def close(self) -> None:
        self.closed = True


def run(coro):
    return asyncio.run(coro)


def _offline_report_text() -> str:
    body = (
        "Confidence: medium. _Provenance: offline fixture; not verified on data._ "
        "Use source-backed validation, compare fold stability, document feature availability, "
        "and keep leaderboard decisions tied to cross-validation evidence. Start with a compact "
        "LightGBM baseline, inspect only public descriptions and notebooks, then add categorical "
        "handling, calibration checks, and conservative ensembling when validation variance is "
        "under control. "
    )
    return "\n\n".join(
        f"## {heading}\n"
        f"{body}"
        f"{body}"
        for heading in SECTION_HEADINGS
    )


def test_full_pipeline_runs_with_offline_fixtures_and_quality_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeStore.instances = []
    FakeDomainMemory.instances = []
    calls: list[str] = []
    generated_reports: list[str] = []
    reasoning = fx.reasoning_outputs()
    original_connect = socket.socket.connect

    def block_external_network_connect(self, address, *args: Any, **kwargs: Any) -> None:
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}:
            return original_connect(self, address, *args, **kwargs)
        raise AssertionError("Offline pipeline test attempted a network connection")

    async def fake_plan(description: str, client: FakeClient, model: str):
        calls.append("planner")
        assert description == fx.competition_desc()
        return fx.plan_data()

    async def fake_search_paper_sources(**kwargs: Any) -> list[dict[str, object]]:
        calls.append("paper_search")
        return []

    async def fake_search_repos(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        calls.append("github_search")
        return [{"full_name": "offline/credit-risk"}]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        calls.append("summarizer")
        return [
            doc.model_copy(update={"summary": f"offline summary for {doc.id}"})
            for doc in kwargs["docs"]
        ]

    async def fake_hybrid_search(store: FakeStore, query: str, top_k: int) -> list[RetrievedDocument]:
        calls.append(f"retrieve:{query}")
        return fx.retrieved_documents()

    async def fake_analyze_metric(**kwargs: Any):
        calls.append("metric")
        return reasoning.metric

    async def fake_design_validation(**kwargs: Any):
        calls.append("validation")
        return reasoning.validation

    async def fake_analyze_leakage_risk(**kwargs: Any):
        calls.append("leakage")
        return reasoning.leakage

    async def fake_audit_leaderboard_risk(**kwargs: Any):
        calls.append("leaderboard")
        return reasoning.leaderboard

    async def fake_plan_experiments(**kwargs: Any):
        calls.append("experiments")
        return reasoning.experiments

    async def fake_review(**kwargs: Any):
        calls.append("review")
        return reasoning.review

    async def fake_compose_report(**kwargs: Any) -> str:
        calls.append("report")
        assert kwargs["metric_result"] == reasoning.metric
        return _offline_report_text()

    def fake_generate_report(
        competition_name: str,
        roadmap_text: str,
        sources: list[RetrievedDocument],
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        calls.append("docx")
        generated_reports.append(roadmap_text)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("offline docx placeholder", encoding="utf-8")
        return path

    monkeypatch.setattr(socket.socket, "connect", block_external_network_connect)
    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", FakeDomainMemory)
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(
        main_module,
        "search_notebooks",
        lambda queries, competition_id, max_notebooks: [
            {"kernel_ref": f"offline/kernel-{index}", "content": "offline notebook"}
            for index in range(1, 4)
        ],
    )
    monkeypatch.setattr(main_module, "build_kaggle_documents", lambda raw, competition_id: fx.kaggle_documents())
    monkeypatch.setattr(
        main_module,
        "search_arxiv",
        lambda queries, max_papers: [{"entry_id": "offline-paper-1"}, {"entry_id": "offline-paper-2"}],
    )
    monkeypatch.setattr(main_module, "enrich_with_pdf", lambda papers, cache_dir: papers)
    monkeypatch.setattr(main_module, "build_arxiv_documents", lambda raw, competition_id: fx.arxiv_documents() if raw else [])
    monkeypatch.setattr(main_module, "search_paper_sources", fake_search_paper_sources)
    monkeypatch.setattr(main_module, "search_repos", fake_search_repos)
    monkeypatch.setattr(main_module, "build_github_documents", lambda raw, competition_id: fx.github_documents())
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: fx.mock_embeddings())
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
            competition_url=fx.COMPETITION_URL,
            competition_desc=fx.competition_desc(),
            output_dir=tmp_path,
            show_progress=False,
        )
    )

    report_path = Path(result.report_path)
    run_path = Path(result.run_artifacts_path)
    summary = json.loads((run_path / "research_run.json").read_text(encoding="utf-8"))
    quality_warnings = [
        *validate_report_text(generated_reports[0]),
        *validate_retrieved_documents(fx.retrieved_documents()),
        *validate_reasoning_outputs(reasoning.as_quality_dict()),
    ]

    assert isinstance(result, ResearchRunResult)
    assert report_path.exists()
    assert report_path.suffix == ".docx"
    assert result.competition_id == fx.COMPETITION_ID
    assert result.report_mode == "full"
    assert result.num_documents == 6
    assert result.num_sources == {
        "kaggle": 3,
        "arxiv": 2,
        "huggingface_papers": 0,
        "papers_with_code_legacy": 0,
        "papers_with_code": 0,
        "github": 1,
    }
    assert result.warnings == [
        "Hugging Face Papers source count is 0. Falling back to arXiv-only academic retrieval."
    ]
    assert quality_warnings == []
    assert [doc.id for doc in FakeStore.instances[0].upserted_docs] == [
        "kaggle-1",
        "kaggle-2",
        "kaggle-3",
        "arxiv-1",
        "arxiv-2",
        "github-1",
    ]
    assert FakeStore.instances[0].upserted_embeddings == fx.mock_embeddings()
    assert FakeStore.instances[0].closed is True
    assert FakeDomainMemory.instances[0].closed is True
    assert summary["retrieved_document_ids"] == [
        "retrieved-kaggle-1",
        "retrieved-arxiv-1",
        "retrieved-github-1",
    ]
    assert summary["reasoning_outputs_summary"]["review"]["confidence"] == "medium"
    assert json.loads((report_path.parent / "research_run.json").read_text(encoding="utf-8")) == summary
    assert "github_search" in calls
    assert "report" in calls
    assert "docx" in calls
