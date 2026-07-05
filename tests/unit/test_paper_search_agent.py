from __future__ import annotations

from typing import Any

import pytest

from kaggle_researcher.agents import paper_search_agent
from kaggle_researcher.agents.paper_search_agent import (
    _search_legacy_papers_with_code,
    search_huggingface_papers,
    search_paper_sources,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        json_error: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error
        self.headers = headers or {}

    def json(self) -> Any:
        if self.json_error:
            raise ValueError("not json")
        return self.payload


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def run(coro):
    import asyncio

    return asyncio.run(coro)


def test_search_huggingface_papers_parses_json_list_response() -> None:
    client = FakeAsyncClient(
        [
            FakeResponse(
                payload=[
                    {
                        "id": "2401.12345",
                        "title": "Credit default prediction",
                        "summary": "A useful abstract.",
                        "arxivId": "2401.12345",
                    }
                ]
            )
        ]
    )

    docs = run(search_huggingface_papers("credit default", client=client))

    assert docs[0]["source"] == "huggingface_papers"
    assert docs[0]["id"] == "hf_paper:2401.12345"
    assert docs[0]["metadata"]["arxiv_id"] == "2401.12345"
    assert "A useful abstract." in docs[0]["content"]


def test_search_huggingface_papers_parses_object_results_response() -> None:
    client = FakeAsyncClient(
        [
            FakeResponse(
                payload={
                    "results": [
                        {
                            "paper": {
                                "id": "paper-1",
                                "title": "Gini optimization",
                                "abstract": "Metric details.",
                            }
                        }
                    ]
                }
            )
        ]
    )

    docs = run(search_huggingface_papers("gini", client=client))

    assert docs[0]["title"] == "Gini optimization"
    assert docs[0]["metadata"]["paper_id"] == "paper-1"


def test_search_huggingface_papers_uses_title_as_content_when_abstract_missing() -> None:
    client = FakeAsyncClient([FakeResponse(payload=[{"id": "paper-1", "title": "Only title"}])])

    docs = run(search_huggingface_papers("query", client=client))

    assert docs[0]["content"] == "Only title"


def test_search_huggingface_papers_skips_empty_records() -> None:
    client = FakeAsyncClient([FakeResponse(payload=[{"id": ""}, {"title": "Valid"}])])

    docs = run(search_huggingface_papers("query", client=client))

    assert len(docs) == 1
    assert docs[0]["title"] == "Valid"


def test_search_huggingface_papers_adds_hf_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    client = FakeAsyncClient([FakeResponse(payload=[{"title": "Paper"}])])

    run(search_huggingface_papers("query", client=client))

    assert client.calls[0]["headers"]["Authorization"] == "Bearer hf-secret"


def test_search_huggingface_papers_omits_hf_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = FakeAsyncClient([FakeResponse(payload=[{"title": "Paper"}])])

    run(search_huggingface_papers("query", client=client))

    assert "Authorization" not in client.calls[0]["headers"]


def test_search_huggingface_papers_handles_429_with_warning() -> None:
    warnings: list[str] = []
    client = FakeAsyncClient([FakeResponse(status_code=429, payload={})])

    docs = run(search_huggingface_papers("query", client=client, warnings=warnings))

    assert docs == []
    assert warnings == ["Hugging Face Papers rate-limited query 'query' (429)."]


def test_search_huggingface_papers_handles_non_json_with_warning() -> None:
    warnings: list[str] = []
    client = FakeAsyncClient([FakeResponse(payload=None, json_error=True)])

    docs = run(search_huggingface_papers("query", client=client, warnings=warnings))

    assert docs == []
    assert warnings == ["Hugging Face Papers search returned non-JSON response for query 'query'."]


def test_search_paper_sources_does_not_call_legacy_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls = 0

    async def fake_hf(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def fake_legacy(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal legacy_calls
        legacy_calls += 1
        return []

    async def fake_arxiv(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.delenv("ENABLE_LEGACY_PWC", raising=False)
    monkeypatch.setattr(paper_search_agent, "search_huggingface_papers", fake_hf)
    monkeypatch.setattr(paper_search_agent, "_search_legacy_papers_with_code", fake_legacy)
    monkeypatch.setattr(paper_search_agent, "_search_arxiv_fallback", fake_arxiv)

    run(search_paper_sources(["query"], max_results=10, warnings=[]))

    assert legacy_calls == 0


def test_search_paper_sources_calls_legacy_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_calls = 0

    async def fake_hf(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def fake_legacy(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal legacy_calls
        legacy_calls += 1
        return []

    async def fake_arxiv(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setenv("ENABLE_LEGACY_PWC", "1")
    monkeypatch.setattr(paper_search_agent, "search_huggingface_papers", fake_hf)
    monkeypatch.setattr(paper_search_agent, "_search_legacy_papers_with_code", fake_legacy)
    monkeypatch.setattr(paper_search_agent, "_search_arxiv_fallback", fake_arxiv)

    run(search_paper_sources(["query"], max_results=10, warnings=[]))

    assert legacy_calls == 1


def test_legacy_redirect_warning_is_concise_once() -> None:
    warnings: list[str] = []
    client = FakeAsyncClient(
        [
            FakeResponse(status_code=302, headers={"location": "https://huggingface.co/papers/trending"}),
            FakeResponse(status_code=302, headers={"location": "https://huggingface.co/papers/trending"}),
        ]
    )

    run(_search_legacy_papers_with_code("query 1", 10, client, warnings))
    run(_search_legacy_papers_with_code("query 2", 10, client, warnings))

    assert warnings == [
        "Legacy Papers with Code endpoint redirected to Hugging Face; "
        "legacy PWC disabled/fallback skipped."
    ]


def test_search_paper_sources_falls_back_to_arxiv_when_hf_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_hf(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def fake_arxiv(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "arxiv-1",
                "entry_id": "2401.00001",
                "title": "Fallback paper",
                "content": "abstract",
                "source": "arxiv",
            }
        ]

    monkeypatch.setattr(paper_search_agent, "search_huggingface_papers", fake_hf)
    monkeypatch.setattr(paper_search_agent, "_search_arxiv_fallback", fake_arxiv)

    docs = run(search_paper_sources(["query"], max_results=10, warnings=[]))

    assert docs[0]["source"] == "arxiv"
    assert docs[0]["title"] == "Fallback paper"


def test_search_paper_sources_deduplicates_by_title_and_arxiv_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_hf(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "hf-1",
                "title": "Same Paper",
                "content": "one",
                "source": "huggingface_papers",
                "metadata": {"arxiv_id": "2401.00001"},
            },
            {
                "id": "hf-2",
                "title": "Same Paper",
                "content": "two",
                "source": "huggingface_papers",
                "metadata": {"arxiv_id": "2401.00001"},
            },
            {
                "id": "hf-3",
                "title": "Same Paper",
                "content": "three",
                "source": "huggingface_papers",
                "metadata": {},
            },
        ]

    monkeypatch.setattr(paper_search_agent, "search_huggingface_papers", fake_hf)

    docs = run(search_paper_sources(["query"], max_results=10, warnings=[]))

    assert len(docs) == 1
    assert docs[0]["id"] == "hf-1"
