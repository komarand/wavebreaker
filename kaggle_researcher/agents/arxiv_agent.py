from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

import httpx

from kaggle_researcher.logging_utils import get_logger
from kaggle_researcher.parsers.pdf_parser import download_pdf, parse_pdf
from kaggle_researcher.parsers.pdf_parser import PDF_PARSER_VERSION
from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.source_registry.fingerprints import build_parser_fingerprint
from kaggle_researcher.source_registry.hashing import sha256_bytes

try:  # pragma: no cover - optional dependency is mocked in unit tests.
    import arxiv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    arxiv = None  # type: ignore[assignment]


logger = get_logger(__name__)


def search_arxiv(queries: list[str], max_papers: int) -> list[dict[str, Any]]:
    if max_papers <= 0:
        return []
    if arxiv is None:
        raise RuntimeError("arxiv package is required for search_arxiv")

    papers_by_entry_id: dict[str, dict[str, Any]] = {}
    client = arxiv.Client()

    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            paper = _paper_from_arxiv_result(result)
            entry_id = paper["entry_id"]
            papers_by_entry_id.setdefault(entry_id, paper)
            if len(papers_by_entry_id) >= max_papers:
                break
        if len(papers_by_entry_id) >= max_papers:
            break

    return list(papers_by_entry_id.values())[:max_papers]


def enrich_with_pdf(papers: list[dict[str, Any]], cache_dir: str) -> list[dict[str, Any]]:
    return asyncio.run(_enrich_with_pdf_async(papers=papers, cache_dir=cache_dir))


def search_papers_with_code(query: str) -> list[dict[str, Any]]:
    response = httpx.get(
        "https://paperswithcode.com/api/v1/papers/",
        params={"q": query, "page_size": 10},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", []) if isinstance(payload, dict) else []

    papers: list[dict[str, Any]] = []
    for item in results:
        paper_id = str(item.get("id") or item.get("arxiv_id") or item.get("url_abs") or item.get("title"))
        title = str(item.get("title") or "Untitled Papers with Code paper")
        abstract = str(item.get("abstract") or "")
        url = str(item.get("url_abs") or item.get("url_pdf") or "")
        papers.append(
            {
                "id": paper_id,
                "entry_id": paper_id,
                "title": title,
                "abstract": abstract,
                "content": abstract,
                "url": url,
                "source": "papers_with_code",
                "metadata": {"paper_id": paper_id},
            }
        )

    return papers


def build_arxiv_documents(
    papers: list[dict[str, Any]],
    competition_id: str,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    seen_entry_ids: set[str] = set()

    for paper in papers:
        source = str(paper.get("source") or "arxiv")
        entry_id = str(paper.get("entry_id") or paper.get("id") or paper.get("url") or paper.get("title"))
        dedup_key = f"{source}:{entry_id}"
        if source == "arxiv" and entry_id in seen_entry_ids:
            continue
        if source == "arxiv":
            seen_entry_ids.add(entry_id)

        abstract = str(paper.get("abstract") or "")
        content = str(paper.get("content") or abstract)
        title = str(paper.get("title") or entry_id)
        url = paper.get("url") or paper.get("entry_id")
        metadata = dict(paper.get("metadata") or {})
        metadata.update(
            {
                "entry_id": entry_id,
                "pdf_url": paper.get("pdf_url"),
                "source_revision": paper.get("source_revision") or metadata.get("source_revision"),
                "revision_is_reliable": bool(
                    paper.get("revision_is_reliable", metadata.get("revision_is_reliable", False))
                ),
            }
        )

        documents.append(
            SourceDocument(
                id=_document_id(competition_id=competition_id, key=dedup_key),
                competition_id=competition_id,
                source=source,
                title=title,
                url=str(url) if url else None,
                content=content,
                metadata=metadata,
            )
        )

    return documents


async def _enrich_with_pdf_async(papers: list[dict[str, Any]], cache_dir: str) -> list[dict[str, Any]]:
    enriched = await asyncio.gather(*[_enrich_one(paper, cache_dir) for paper in papers])
    return list(enriched)


async def _enrich_one(paper: dict[str, Any], cache_dir: str) -> dict[str, Any]:
    enriched = dict(paper)
    abstract = str(enriched.get("abstract") or "")
    pdf_url = enriched.get("pdf_url")
    paper_id = str(enriched.get("entry_id") or enriched.get("id") or enriched.get("title") or "paper")

    if not pdf_url:
        enriched["content"] = abstract
        return enriched

    try:
        pdf_path = await download_pdf(str(pdf_url), paper_id=paper_id, cache_dir=cache_dir)
        if pdf_path is None:
            enriched["content"] = abstract
            return enriched

        parsed_content = parse_pdf(pdf_path)
        enriched["content"] = parsed_content if parsed_content.strip() else abstract
        raw_pdf_hash = sha256_bytes(pdf_path.read_bytes()) if pdf_path.is_file() else None
        enriched["metadata"] = {
            **dict(enriched.get("metadata") or {}),
            "raw_pdf_hash": raw_pdf_hash,
            "raw_pdf_size_bytes": pdf_path.stat().st_size if pdf_path.is_file() else None,
            "content_location": str(pdf_path) if pdf_path.is_file() else None,
            "content_mime_type": "application/pdf",
            "parser_fingerprint": arxiv_pdf_parser_fingerprint(),
            "parsed_with_fingerprint": arxiv_pdf_parser_fingerprint(),
        }
    except Exception as exc:
        logger.warning("Failed to enrich paper %s with PDF content: %s", paper_id, exc)
        enriched["content"] = abstract

    return enriched


def arxiv_pdf_parser_fingerprint() -> str:
    return build_parser_fingerprint(
        processor_name="arxiv_pdf_parser",
        processor_version=PDF_PARSER_VERSION,
    ).fingerprint


def _paper_from_arxiv_result(result: Any) -> dict[str, Any]:
    entry_id = str(getattr(result, "entry_id", "") or getattr(result, "id", ""))
    pdf_url = str(getattr(result, "pdf_url", "") or "")
    title = str(getattr(result, "title", "") or "Untitled arXiv paper")
    abstract = str(getattr(result, "summary", "") or "")
    authors = [str(author) for author in getattr(result, "authors", [])]

    revision_match = re.search(r"(v\d+)(?:\.pdf)?$", entry_id, flags=re.I)
    source_revision = revision_match.group(1).lower() if revision_match else None
    return {
        "id": entry_id,
        "entry_id": entry_id,
        "title": title,
        "abstract": abstract,
        "content": abstract,
        "pdf_url": pdf_url,
        "url": entry_id,
        "source": "arxiv",
        "source_revision": source_revision,
        "revision_is_reliable": source_revision is not None,
        "metadata": {
            "authors": authors,
            "published": str(getattr(result, "published", "") or ""),
            "source_revision": source_revision,
            "revision_is_reliable": source_revision is not None,
        },
    }


def _document_id(competition_id: str, key: str) -> str:
    digest = hashlib.sha1(f"{competition_id}:{key}".encode("utf-8")).hexdigest()[:16]
    return f"paper-{digest}"
