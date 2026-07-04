from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from kaggle_researcher.agents.github_agent import build_github_documents, search_repos
from kaggle_researcher.schemas import SourceDocument


def run(coro):
    return asyncio.run(coro)


def test_search_repos_uses_mocked_github_responses(monkeypatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/search/repositories":
            return httpx.Response(
                200,
                request=request,
                headers={"x-ratelimit-remaining": "59", "x-ratelimit-reset": "123"},
                json={
                    "items": [
                        {
                            "full_name": "alice/good",
                            "description": "good repo",
                            "html_url": "https://github.com/alice/good",
                            "stargazers_count": 10,
                            "language": "Python",
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, request=request, text="# README\nUseful code.")

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("kaggle_researcher.agents.github_agent.httpx.AsyncClient", async_client_factory)

    repos = run(search_repos(["tabular auc"], max_repos=5))

    assert len(repos) == 1
    assert repos[0]["full_name"] == "alice/good"
    assert repos[0]["readme"] == "# README\nUseful code."
    assert repos[0]["rate_limit_remaining"] == "59"
    assert requests[0].headers.get("authorization") is None
    assert requests[0].url.params["q"] == "tabular auc"
    assert requests[1].url.path == "/repos/alice/good/readme"


def test_search_repos_deduplicates_by_full_name_and_sorts_by_stars(monkeypatch) -> None:
    search_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal search_calls
        if request.url.path == "/search/repositories":
            search_calls += 1
            if search_calls == 1:
                items = [
                    {"full_name": "alice/repo", "description": "old", "stargazers_count": 5},
                    {"full_name": "bob/repo", "description": "best", "stargazers_count": 20},
                ]
            else:
                items = [
                    {"full_name": "alice/repo", "description": "new", "stargazers_count": 12},
                    {"full_name": "cara/repo", "description": "low", "stargazers_count": 1},
                ]
            return httpx.Response(200, request=request, json={"items": items})
        return httpx.Response(404, request=request)

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("kaggle_researcher.agents.github_agent.httpx.AsyncClient", async_client_factory)

    repos = run(search_repos(["one", "two"], max_repos=10))

    assert [repo["full_name"] for repo in repos] == ["bob/repo", "alice/repo", "cara/repo"]
    assert repos[1]["description"] == "new"
    assert repos[1]["stars"] == 12


def test_search_repos_adds_token_and_rate_limit_warning(monkeypatch) -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/search/repositories":
            return httpx.Response(
                200,
                request=request,
                headers={"x-ratelimit-remaining": "0"},
                json={"items": [{"full_name": "alice/repo", "stargazers_count": 1}]},
            )
        return httpx.Response(404, request=request)

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("kaggle_researcher.agents.github_agent.httpx.AsyncClient", async_client_factory)

    repos_without_token = run(search_repos(["query"], token=None, max_repos=1))
    repos_with_token = run(search_repos(["query"], token="secret", max_repos=1))

    assert repos_without_token[0]["warning"] == "GitHub unauthenticated rate limit exhausted"
    assert "warning" not in repos_with_token[0]
    assert captured_requests[-2].headers["authorization"] == "Bearer secret"


def test_search_repos_skips_failed_github_queries(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request, json={"message": "rate limited"})

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("kaggle_researcher.agents.github_agent.httpx.AsyncClient", async_client_factory)

    assert run(search_repos(["query"], max_repos=1)) == []


def test_search_repos_decodes_json_base64_readme(monkeypatch) -> None:
    encoded = base64.b64encode(b"# JSON README").decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/repositories":
            return httpx.Response(
                200,
                request=request,
                json={"items": [{"full_name": "alice/repo", "stargazers_count": 1}]},
            )
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={"encoding": "base64", "content": encoded},
        )

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("kaggle_researcher.agents.github_agent.httpx.AsyncClient", async_client_factory)

    repos = run(search_repos(["query"], max_repos=1))

    assert repos[0]["readme"] == "# JSON README"


def test_build_github_documents_creates_valid_documents_with_metadata() -> None:
    documents = build_github_documents(
        [
            {
                "full_name": "alice/repo",
                "description": "description",
                "readme": "# readme",
                "url": "https://github.com/alice/repo",
                "stars": 7,
                "language": "Python",
                "updated_at": "2026-01-01T00:00:00Z",
            },
            {
                "full_name": "alice/repo",
                "description": "duplicate",
                "stars": 1,
            },
        ],
        competition_id="comp-1",
    )

    assert len(documents) == 1
    assert isinstance(documents[0], SourceDocument)
    assert documents[0].source == "github"
    assert documents[0].content == "description\n\n# readme"
    assert documents[0].metadata["stars"] == 7
    assert documents[0].metadata["language"] == "Python"
    assert documents[0].metadata["full_name"] == "alice/repo"
