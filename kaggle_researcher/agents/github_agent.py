from __future__ import annotations

import base64
import hashlib
from typing import Any

import httpx

from kaggle_researcher.logging_utils import get_logger
from kaggle_researcher.schemas import SourceDocument


logger = get_logger(__name__)


async def search_repos(
    queries: list[str],
    token: str | None = None,
    max_repos: int = 10,
) -> list[dict[str, Any]]:
    if max_repos <= 0:
        return []

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "KaggleResearcher",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repos_by_full_name: dict[str, dict[str, Any]] = {}

    async with httpx.AsyncClient(
        base_url="https://api.github.com",
        headers=headers,
        timeout=30,
    ) as client:
        for query in queries:
            response = await client.get(
                "/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": max_repos,
                },
            )
            if response.status_code >= 400:
                logger.warning("GitHub repository search failed for query %r: %s", query, response.status_code)
                continue

            rate_limit_context = _rate_limit_context(response=response, token=token)
            payload = response.json()
            items = payload.get("items", []) if isinstance(payload, dict) else []

            for item in items:
                repo = await _repo_from_search_item(
                    item=item,
                    client=client,
                    rate_limit_context=rate_limit_context,
                )
                full_name = repo.get("full_name")
                if not full_name:
                    continue

                existing = repos_by_full_name.get(full_name)
                if existing is None or repo.get("stars", 0) > existing.get("stars", 0):
                    repos_by_full_name[full_name] = repo

    return sorted(
        repos_by_full_name.values(),
        key=lambda repo: repo.get("stars", 0),
        reverse=True,
    )[:max_repos]


def build_github_documents(
    raw_repos: list[dict[str, Any]],
    competition_id: str,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    repos_by_full_name: dict[str, dict[str, Any]] = {}

    for raw_repo in raw_repos:
        full_name = str(raw_repo.get("full_name") or raw_repo.get("fullName") or "")
        if not full_name:
            continue
        stars = _parse_int(raw_repo.get("stars", raw_repo.get("stargazers_count", 0)))
        existing = repos_by_full_name.get(full_name)
        if existing is None or stars > _parse_int(existing.get("stars", existing.get("stargazers_count", 0))):
            repos_by_full_name[full_name] = raw_repo

    sorted_repos = sorted(
        repos_by_full_name.values(),
        key=lambda repo: _parse_int(repo.get("stars", repo.get("stargazers_count", 0))),
        reverse=True,
    )

    for repo in sorted_repos:
        full_name = str(repo.get("full_name") or repo.get("fullName"))
        description = str(repo.get("description") or "")
        readme = str(repo.get("readme") or repo.get("readme_text") or "")
        content = "\n\n".join(part for part in (description, readme) if part.strip())
        stars = _parse_int(repo.get("stars", repo.get("stargazers_count", 0)))
        language = repo.get("language")
        updated_at = repo.get("updated_at")
        metadata = dict(repo.get("metadata") or {})
        metadata.update(
            {
                "stars": stars,
                "language": language,
                "full_name": full_name,
                "updated_at": updated_at,
            }
        )
        if "rate_limit_remaining" in repo:
            metadata["rate_limit_remaining"] = repo.get("rate_limit_remaining")
        if "rate_limit_reset" in repo:
            metadata["rate_limit_reset"] = repo.get("rate_limit_reset")
        if repo.get("warning"):
            metadata["warning"] = repo["warning"]

        documents.append(
            SourceDocument(
                id=_document_id(competition_id=competition_id, full_name=full_name),
                competition_id=competition_id,
                source="github",
                title=full_name,
                url=str(repo.get("url") or repo.get("html_url") or f"https://github.com/{full_name}"),
                content=content,
                metadata=metadata,
            )
        )

    return documents


async def _repo_from_search_item(
    item: dict[str, Any],
    client: httpx.AsyncClient,
    rate_limit_context: dict[str, Any],
) -> dict[str, Any]:
    full_name = str(item.get("full_name") or "")
    readme = await _fetch_readme(client=client, full_name=full_name) if full_name else ""

    repo = {
        "full_name": full_name,
        "description": item.get("description") or "",
        "url": item.get("html_url") or f"https://github.com/{full_name}",
        "stars": _parse_int(item.get("stargazers_count", item.get("stars", 0))),
        "language": item.get("language"),
        "updated_at": item.get("updated_at"),
        "readme": readme,
        "source": "github",
    }
    repo.update(rate_limit_context)
    return repo


async def _fetch_readme(client: httpx.AsyncClient, full_name: str) -> str:
    response = await client.get(
        f"/repos/{full_name}/readme",
        headers={"Accept": "application/vnd.github.raw"},
    )
    if response.status_code == 404:
        return ""
    if response.status_code >= 400:
        logger.warning("GitHub README fetch failed for %s: %s", full_name, response.status_code)
        return ""

    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower():
        payload = response.json()
        encoded = payload.get("content", "") if isinstance(payload, dict) else ""
        if payload.get("encoding") == "base64" and encoded:
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
    return response.text


def _rate_limit_context(response: httpx.Response, token: str | None) -> dict[str, Any]:
    remaining = response.headers.get("x-ratelimit-remaining")
    reset = response.headers.get("x-ratelimit-reset")
    context: dict[str, Any] = {}

    if remaining is not None:
        context["rate_limit_remaining"] = remaining
    if reset is not None:
        context["rate_limit_reset"] = reset
    if token is None and remaining == "0":
        context["warning"] = "GitHub unauthenticated rate limit exhausted"

    return context


def _parse_int(value: Any) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return 0


def _document_id(competition_id: str, full_name: str) -> str:
    digest = hashlib.sha1(f"{competition_id}:{full_name}".encode("utf-8")).hexdigest()[:16]
    return f"github-{digest}"
