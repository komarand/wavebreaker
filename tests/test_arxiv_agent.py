from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from kaggle_researcher.agents import arxiv_agent
from kaggle_researcher.agents.arxiv_agent import (
    build_arxiv_documents,
    enrich_with_pdf,
    search_arxiv,
    search_papers_with_code,
)
from kaggle_researcher.schemas import SourceDocument


def test_search_arxiv_deduplicates_by_entry_id(monkeypatch: pytest.MonkeyPatch) -> None:
    first = SimpleNamespace(
        entry_id="https://arxiv.org/abs/1",
        pdf_url="https://arxiv.org/pdf/1",
        title="First",
        summary="First abstract",
        authors=["Ada"],
        published="2025-01-01",
    )
    duplicate = SimpleNamespace(
        entry_id="https://arxiv.org/abs/1",
        pdf_url="https://arxiv.org/pdf/1",
        title="Duplicate",
        summary="Duplicate abstract",
        authors=[],
        published="2025-01-02",
    )
    second = SimpleNamespace(
        entry_id="https://arxiv.org/abs/2",
        pdf_url="https://arxiv.org/pdf/2",
        title="Second",
        summary="Second abstract",
        authors=["Grace"],
        published="2025-01-03",
    )

    class FakeClient:
        def results(self, search: Any):
            return iter([first, duplicate, second])

    class FakeArxiv:
        Client = FakeClient

        class Search:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class SortCriterion:
            Relevance = "relevance"

    monkeypatch.setattr(arxiv_agent, "arxiv", FakeArxiv)

    papers = search_arxiv(["query"], max_papers=10)

    assert [paper["entry_id"] for paper in papers] == [
        "https://arxiv.org/abs/1",
        "https://arxiv.org/abs/2",
    ]
    assert papers[0]["title"] == "First"


def test_enrich_with_pdf_uses_parsed_content(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def fake_download_pdf(url: str, paper_id: str, cache_dir: str):
        return tmp_path / "paper.pdf"

    monkeypatch.setattr(arxiv_agent, "download_pdf", fake_download_pdf)
    monkeypatch.setattr(arxiv_agent, "parse_pdf", lambda path: "parsed pdf text")

    enriched = enrich_with_pdf(
        [{"entry_id": "paper-1", "pdf_url": "https://example.com/paper.pdf", "abstract": "abstract"}],
        cache_dir=str(tmp_path),
    )

    assert enriched[0]["content"] == "parsed pdf text"


def test_pdf_failure_falls_back_to_abstract(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def fake_download_pdf(url: str, paper_id: str, cache_dir: str):
        return None

    monkeypatch.setattr(arxiv_agent, "download_pdf", fake_download_pdf)

    enriched = enrich_with_pdf(
        [{"entry_id": "paper-1", "pdf_url": "https://example.com/paper.pdf", "abstract": "fallback abstract"}],
        cache_dir=str(tmp_path),
    )

    assert enriched[0]["content"] == "fallback abstract"


def test_search_papers_with_code_uses_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        requests.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "results": [
                    {
                        "id": "pwc-1",
                        "title": "Paper",
                        "abstract": "PWC abstract",
                        "url_abs": "https://paperswithcode.com/paper/paper",
                    }
                ]
            },
        )

    monkeypatch.setattr(arxiv_agent.httpx, "get", fake_get)

    papers = search_papers_with_code("tabular auc")

    assert papers[0]["source"] == "papers_with_code"
    assert papers[0]["content"] == "PWC abstract"
    assert requests[0]["params"] == {"q": "tabular auc", "page_size": 10}


def test_build_arxiv_documents_creates_valid_source_documents() -> None:
    documents = build_arxiv_documents(
        [
            {
                "entry_id": "https://arxiv.org/abs/1",
                "title": "Arxiv Paper",
                "abstract": "abstract",
                "content": "parsed",
                "pdf_url": "https://arxiv.org/pdf/1",
                "url": "https://arxiv.org/abs/1",
                "source": "arxiv",
            },
            {
                "entry_id": "https://arxiv.org/abs/1",
                "title": "Duplicate Arxiv Paper",
                "abstract": "duplicate",
                "source": "arxiv",
            },
            {
                "id": "pwc-1",
                "title": "PWC Paper",
                "content": "pwc text",
                "url": "https://paperswithcode.com/paper/pwc",
                "source": "papers_with_code",
            },
        ],
        competition_id="comp-1",
    )

    assert len(documents) == 2
    assert all(isinstance(document, SourceDocument) for document in documents)
    assert documents[0].source == "arxiv"
    assert documents[0].content == "parsed"
    assert documents[1].source == "papers_with_code"
