from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_researcher import main as main_module
from kaggle_researcher.main import build_parser, run_research
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
    max_notebooks: int = 10
    max_papers: int = 10
    max_repos: int = 10
    pdf_cache_dir: str = "./data/pdfs"
    github_token: str | None = None


class FakeClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


class FakeStore:
    def __init__(self, competition_id: str, dsn: str, embed_dim: int) -> None:
        self.competition_id = competition_id

    async def init(self) -> None:
        pass

    async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
        pass

    async def close(self) -> None:
        pass


class FakeDomainMemory:
    def __init__(self, dsn: str, embed_dim: int) -> None:
        pass

    async def init(self) -> None:
        pass

    async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
        return []

    async def close(self) -> None:
        pass


def run(coro):
    return asyncio.run(coro)


def test_cli_parses_task25_flags() -> None:
    args = build_parser().parse_args(
        [
            "https://www.kaggle.com/competitions/comp-1",
            "Predict defaults.",
            "--output-dir",
            "./reports",
            "--debug",
            "--no-github",
            "--fast",
        ]
    )

    assert args.output_dir == "./reports"
    assert args.debug is True
    assert args.no_github is True
    assert args.fast is True


def test_no_github_skips_agent_and_writes_research_run_json(monkeypatch, tmp_path: Path) -> None:
    notebook_limits: list[int] = []
    paper_limits: list[int] = []
    github_called = False

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="classification",
            metric="auc",
            domain="credit",
            kaggle_queries=["credit kaggle"],
            arxiv_queries=["credit paper"],
            github_queries=["credit repo"],
        )

    async def fake_search_paper_sources(queries, max_results, warnings):
        paper_limits.append(max_results)
        return []

    async def fake_summarize_documents(
        client: Any,
        docs: list[SourceDocument],
        model: str,
        *,
        show_progress: bool = False,
    ) -> list[SourceDocument]:
        return [doc.model_copy(update={"summary": f"summary {doc.id}"}) for doc in docs]

    async def fake_hybrid_search(store: FakeStore, query: str, top_k: int) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id="retrieved-1",
                competition_id=store.competition_id,
                source="kaggle",
                title="Evidence",
                url="https://example.com/evidence",
                content="retrieved evidence",
                score=0.9,
                rrf_score=0.1,
            )
        ]

    def fake_search_notebooks(queries, competition_id, max_notebooks):
        notebook_limits.append(max_notebooks)
        return [{"kernel_ref": "user/kernel", "content": "notebook text"}]

    def fake_search_repos(*args: Any, **kwargs: Any):
        nonlocal github_called
        github_called = True
        raise AssertionError("GitHub agent should be skipped")

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
    monkeypatch.setattr(main_module, "search_notebooks", fake_search_notebooks)
    monkeypatch.setattr(
        main_module,
        "build_kaggle_documents",
        lambda raw, competition_id: [
            SourceDocument(
                id="kaggle-1",
                competition_id=competition_id,
                source="kaggle",
                title="Notebook",
                url="https://example.com/notebook",
                content="notebook content",
            )
        ],
    )
    monkeypatch.setattr(main_module, "search_arxiv", lambda queries, max_papers: [])
    monkeypatch.setattr(main_module, "enrich_with_pdf", lambda papers, cache_dir: papers)
    monkeypatch.setattr(main_module, "build_arxiv_documents", lambda raw, competition_id: [])
    monkeypatch.setattr(main_module, "search_paper_sources", fake_search_paper_sources)
    monkeypatch.setattr(main_module, "search_repos", fake_search_repos)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2]])
    monkeypatch.setattr(main_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/comp-1",
            "Predict credit defaults.",
            output_dir=tmp_path,
            show_progress=False,
            report_mode="minimal",
            no_github=True,
            fast=True,
        )
    )

    report_path = Path(result.report_path)
    payload = json.loads((report_path.parent / "research_run.json").read_text(encoding="utf-8"))

    assert github_called is False
    assert notebook_limits == [3]
    assert paper_limits == [3]
    assert report_path.exists()
    assert payload["competition_id"] == "comp-1"
    assert payload["plan_data"]["github_queries"] == ["credit repo"]
    assert payload["num_sources"]["github"] == 0
    assert payload["retrieved_document_ids"] == ["retrieved-1"]
    assert payload["reasoning_outputs_summary"] == {}
    assert payload["report_path"] == result.report_path
    assert "GitHub source collection skipped by --no-github." in payload["warnings"]
    assert not any("GitHub source count is 0" in warning for warning in payload["warnings"])
