from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_researcher.main import derive_competition_id, run_research
from kaggle_researcher.schemas import PlanData, RetrievedDocument, SourceDocument


@dataclass(slots=True)
class FakeSettings:
    deepseek_api_key: str = "secret"
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    embed_dim: int = 2
    max_embed_batch_size: int = 4
    pg_dsn: str = "postgresql://example"
    top_k: int = 2
    max_notebooks: int = 2
    max_papers: int = 2
    pdf_cache_dir: str = "./data/pdfs"


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


def make_source(
    doc_id: str,
    source: str,
    competition_id: str = "playground-series-s5e1",
) -> SourceDocument:
    return SourceDocument(
        id=doc_id,
        competition_id=competition_id,
        source=source,
        title=f"{source} document",
        url="https://example.com/source",
        content="A useful source with enough competition guidance.",
    )


def make_retrieved(doc_id: str, source: str = "kaggle") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        competition_id="playground-series-s5e1",
        source=source,
        title=f"Retrieved {doc_id}",
        url="https://example.com/retrieved",
        content="retrieved content",
        score=0.9,
        rrf_score=0.1,
    )


async def fake_plan(description: str, client: Any, model: str) -> PlanData:
    return PlanData(
        task_type="classification",
        metric="auc",
        domain="tabular",
        kaggle_queries=["tabular auc kaggle"],
        arxiv_queries=["tabular auc paper"],
    )


async def fake_summarize_documents(
    client: Any,
    docs: list[SourceDocument],
    model: str,
) -> list[SourceDocument]:
    return [doc.model_copy(update={"summary": f"summary for {doc.id}"}) for doc in docs]


async def fake_hybrid_search(store: FakeStore, query: str, top_k: int) -> list[RetrievedDocument]:
    if "paper" in query:
        raise RuntimeError("retrieval temporarily unavailable")
    return [make_retrieved("retrieved-1"), make_retrieved("retrieved-2", source="arxiv")]


def run(coro):
    return asyncio.run(coro)


def test_minimal_e2e_pipeline_creates_docx(monkeypatch, tmp_path: Path) -> None:
    FakeStore.instances = []
    report_calls: list[dict[str, Any]] = []
    embed_calls: list[dict[str, Any]] = []

    def fake_generate_report(
        competition_name: str,
        roadmap_text: str,
        sources: list[RetrievedDocument],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("offline fake report", encoding="utf-8")
        report_calls.append(
            {
                "competition_name": competition_name,
                "roadmap_text": roadmap_text,
                "sources": sources,
                "output_path": path,
            }
        )
        return path

    def fake_embed_texts(texts: list[str], batch_size: int) -> list[list[float]]:
        embed_calls.append({"texts": texts, "batch_size": batch_size})
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("kaggle_researcher.main.load_config", lambda: FakeSettings())
    monkeypatch.setattr("kaggle_researcher.main.DeepSeekClient", FakeClient)
    monkeypatch.setattr("kaggle_researcher.main.PgStore", FakeStore)
    monkeypatch.setattr("kaggle_researcher.main.plan", fake_plan)
    monkeypatch.setattr(
        "kaggle_researcher.main.search_notebooks",
        lambda queries, competition_id, max_notebooks: [
            {"kernel_ref": "user/kernel", "content": "notebook text"}
        ],
    )
    monkeypatch.setattr(
        "kaggle_researcher.main.build_kaggle_documents",
        lambda raw_results, competition_id: [make_source("kaggle-1", "kaggle", competition_id)],
    )
    monkeypatch.setattr(
        "kaggle_researcher.main.search_arxiv",
        lambda queries, max_papers: [{"entry_id": "paper", "content": "paper text"}],
    )
    monkeypatch.setattr("kaggle_researcher.main.enrich_with_pdf", lambda papers, cache_dir: papers)
    monkeypatch.setattr(
        "kaggle_researcher.main.build_arxiv_documents",
        lambda papers, competition_id: [make_source("arxiv-1", "arxiv", competition_id)],
    )
    monkeypatch.setattr("kaggle_researcher.main.summarize_documents", fake_summarize_documents)
    monkeypatch.setattr("kaggle_researcher.main.embed_texts", fake_embed_texts)
    monkeypatch.setattr("kaggle_researcher.main.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("kaggle_researcher.main.generate_report", fake_generate_report)

    result = run(
        run_research(
            competition_url="https://www.kaggle.com/competitions/playground-series-s5e1",
            competition_desc="Classify tabular examples with AUC.",
            output_dir=tmp_path,
        )
    )

    report_path = Path(result.report_path)
    assert report_path.exists()
    assert report_path.suffix == ".docx"
    assert result.competition_id == "playground-series-s5e1"
    assert result.num_documents == 2
    assert result.num_sources == {"kaggle": 1, "arxiv": 1}
    assert result.warnings == ["Retrieval failed for query 'tabular auc paper': retrieval temporarily unavailable"]
    assert result.duration_sec >= 0
    assert FakeStore.instances[0].initialized is True
    assert FakeStore.instances[0].closed is True
    assert [doc.id for doc in FakeStore.instances[0].upserted_docs] == ["kaggle-1", "arxiv-1"]
    assert len(FakeStore.instances[0].upserted_embeddings) == 2
    assert embed_calls == [{"texts": ["summary for kaggle-1", "summary for arxiv-1"], "batch_size": 4}]
    assert report_calls[0]["competition_name"] == "playground-series-s5e1"
    assert [source.id for source in report_calls[0]["sources"]] == ["retrieved-1", "retrieved-2"]


def test_derive_competition_id_from_kaggle_url() -> None:
    assert (
        derive_competition_id("https://www.kaggle.com/competitions/playground-series-s5e1/overview")
        == "playground-series-s5e1"
    )
