from __future__ import annotations

import hashlib
import os
from typing import Any

import httpx

from kaggle_researcher.agents.arxiv_agent import search_arxiv


HF_PAPERS_SEARCH_URL = "https://huggingface.co/api/papers/search"
LEGACY_PWC_SEARCH_URL = "https://paperswithcode.com/api/v1/papers/"
ENABLE_LEGACY_PWC = os.getenv("ENABLE_LEGACY_PWC", "0") == "1"


async def search_huggingface_papers(
    query: str,
    limit: int = 10,
    client: httpx.AsyncClient | None = None,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    headers = {"Accept": "application/json"}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async def fetch(active_client: httpx.AsyncClient) -> httpx.Response:
        last_response: httpx.Response | None = None
        for attempt in range(2):
            response = await active_client.get(
                HF_PAPERS_SEARCH_URL,
                params={"q": query, "limit": limit},
                headers=headers,
            )
            last_response = response
            if response.status_code >= 500 and attempt == 0:
                continue
            return response
        return last_response  # type: ignore[return-value]

    if client is None:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as active_client:
            response = await fetch(active_client)
    else:
        response = await fetch(client)

    if response.status_code == 429:
        _append_warning(warnings, f"Hugging Face Papers rate-limited query {query!r} (429).")
        return []
    if response.status_code >= 400:
        _append_warning(
            warnings,
            f"Hugging Face Papers search failed for query {query!r}: status {response.status_code}.",
        )
        return []

    try:
        payload = response.json()
    except ValueError:
        _append_warning(
            warnings,
            f"Hugging Face Papers search returned non-JSON response for query {query!r}.",
        )
        return []

    records = _extract_records(payload)
    documents = [_normalize_hf_record(record, query=query) for record in records]
    return [document for document in documents if document is not None][:limit]


async def search_paper_sources(
    queries: list[str],
    max_results: int = 10,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as hf_client:
        legacy_client: httpx.AsyncClient | None = None
        try:
            if _legacy_pwc_enabled():
                legacy_client = httpx.AsyncClient(follow_redirects=False, timeout=30)

            for query in queries:
                if len(documents) >= max_results:
                    break

                query_documents = await search_huggingface_papers(
                    query=query,
                    limit=max_results,
                    client=hf_client,
                    warnings=warnings,
                )

                if not query_documents and legacy_client is not None:
                    query_documents = await _search_legacy_papers_with_code(
                        query=query,
                        limit=max_results,
                        client=legacy_client,
                        warnings=warnings,
                    )

                if not query_documents:
                    query_documents = await _search_arxiv_fallback(
                        query=query,
                        limit=max_results,
                        warnings=warnings,
                    )

                for document in query_documents:
                    keys = _dedup_keys(document)
                    if any(key in seen_keys for key in keys):
                        continue
                    seen_keys.update(keys)
                    documents.append(document)
                    if len(documents) >= max_results:
                        break
        finally:
            if legacy_client is not None:
                await legacy_client.aclose()

    return documents


async def _search_legacy_papers_with_code(
    query: str,
    limit: int,
    client: httpx.AsyncClient,
    warnings: list[str] | None,
) -> list[dict[str, Any]]:
    response = await client.get(
        LEGACY_PWC_SEARCH_URL,
        params={"q": query, "page_size": limit},
    )
    if response.status_code in {301, 302, 307, 308}:
        location = response.headers.get("location", "")
        if "huggingface.co/papers" in location:
            _append_warning_once(
                warnings,
                "Legacy Papers with Code endpoint redirected to Hugging Face; "
                "legacy PWC disabled/fallback skipped.",
            )
            return []
    if response.status_code >= 400:
        _append_warning(warnings, f"Legacy Papers with Code failed for query {query!r}: status {response.status_code}.")
        return []

    try:
        payload = response.json()
    except ValueError:
        _append_warning(warnings, f"Legacy Papers with Code returned non-JSON response for query {query!r}.")
        return []

    records = _extract_records(payload)
    documents = [_normalize_legacy_pwc_record(record, query=query) for record in records]
    return [document for document in documents if document is not None][:limit]


async def _search_arxiv_fallback(
    query: str,
    limit: int,
    warnings: list[str] | None,
) -> list[dict[str, Any]]:
    try:
        return await asyncio_to_thread(search_arxiv, [query], limit)
    except Exception as exc:
        _append_warning(warnings, f"arXiv fallback failed for paper query {query!r}: {exc}")
        return []


async def asyncio_to_thread(func: Any, *args: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(func, *args)


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("results", "papers", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    return []


def _normalize_hf_record(record: dict[str, Any], query: str) -> dict[str, Any] | None:
    paper = record.get("paper") if isinstance(record.get("paper"), dict) else {}
    title = _first_text(record, paper, "title", "name")
    abstract = _first_text(record, paper, "summary", "abstract", "description")
    arxiv_id = _first_text(record, paper, "arxivId", "arxiv_id", "arxiv")
    paper_id = _first_text(record, paper, "id", "paper_id", "paperId") or arxiv_id or _stable_hash(title)
    content = _content_from_title_and_abstract(title, abstract, record)

    if not title.strip() and not content.strip():
        return None

    url = _paper_url(paper_id=paper_id, arxiv_id=arxiv_id, source="huggingface_papers")
    return {
        "id": f"hf_paper:{paper_id}",
        "entry_id": f"hf_paper:{paper_id}",
        "title": title or paper_id,
        "content": content,
        "abstract": abstract,
        "url": url,
        "source": "huggingface_papers",
        "metadata": {
            "query": query,
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "raw_source": "huggingface_papers",
        },
    }


def _normalize_legacy_pwc_record(record: dict[str, Any], query: str) -> dict[str, Any] | None:
    title = str(record.get("title") or "").strip()
    abstract = str(record.get("abstract") or "").strip()
    paper_id = str(record.get("id") or record.get("arxiv_id") or record.get("url_abs") or _stable_hash(title))
    content = _content_from_title_and_abstract(title, abstract, record)
    if not title.strip() and not content.strip():
        return None
    return {
        "id": f"legacy_pwc:{paper_id}",
        "entry_id": f"legacy_pwc:{paper_id}",
        "title": title or paper_id,
        "content": content,
        "abstract": abstract,
        "url": record.get("url_abs") or record.get("url_pdf"),
        "source": "papers_with_code_legacy",
        "metadata": {
            "query": query,
            "paper_id": paper_id,
            "arxiv_id": record.get("arxiv_id"),
            "raw_source": "papers_with_code_legacy",
        },
    }


def _first_text(record: dict[str, Any], paper: dict[str, Any], *keys: str) -> str:
    for container in (record, paper):
        for key in keys:
            value = container.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _content_from_title_and_abstract(title: str, abstract: str, record: dict[str, Any]) -> str:
    if abstract.strip():
        return "\n\n".join(part for part in (title, abstract) if part.strip())
    metadata_parts = [
        str(record.get(key)).strip()
        for key in ("authors", "publishedAt", "published_at", "venue")
        if record.get(key)
    ]
    return "\n\n".join(part for part in [title, *metadata_parts] if part.strip())


def _paper_url(paper_id: str, arxiv_id: str, source: str) -> str | None:
    if source == "huggingface_papers" and paper_id:
        return f"https://huggingface.co/papers/{paper_id}"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


def _dedup_keys(document: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    arxiv_id = str(metadata.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id}")
    title = _normalize_title(str(document.get("title") or ""))
    if title:
        keys.add(f"title:{title}")
    url = str(document.get("url") or "").strip().lower()
    if url:
        keys.add(f"url:{url}")
    if not keys:
        keys.add(str(document.get("id") or ""))
    return keys


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16] if value else ""


def _legacy_pwc_enabled() -> bool:
    return os.getenv("ENABLE_LEGACY_PWC", "0") == "1"


def _append_warning(warnings: list[str] | None, message: str) -> None:
    if warnings is not None:
        warnings.append(message)


def _append_warning_once(warnings: list[str] | None, message: str) -> None:
    if warnings is not None and message not in warnings:
        warnings.append(message)
