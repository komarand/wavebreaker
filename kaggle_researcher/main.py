from __future__ import annotations

import argparse
import asyncio
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from kaggle_researcher.agents.arxiv_agent import (
    build_arxiv_documents,
    enrich_with_pdf,
    search_arxiv,
)
from kaggle_researcher.agents.kaggle_agent import build_kaggle_documents, search_notebooks
from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.config import load_config
from kaggle_researcher.embedder import embed_texts
from kaggle_researcher.planner import fallback_plan, plan
from kaggle_researcher.report.docx_generator import generate_report
from kaggle_researcher.retriever import hybrid_search
from kaggle_researcher.schemas import PlanData, ResearchRunResult, RetrievedDocument, SourceDocument
from kaggle_researcher.store.pg_store import PgStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_researcher",
        description="KaggleResearcher minimal research pipeline.",
    )
    parser.add_argument("competition_url", nargs="?", help="Kaggle competition URL")
    parser.add_argument("competition_desc", nargs="?", help="Competition description")
    parser.add_argument("--competition-id", dest="competition_id", help="Optional competition identifier")
    parser.add_argument("--output-dir", default="./reports", help="Directory for the generated report")
    return parser


async def run_research(
    competition_url: str,
    competition_desc: str,
    *,
    competition_id: str | None = None,
    output_dir: str | Path = "./reports",
) -> ResearchRunResult:
    started_at = time.perf_counter()
    settings = load_config()
    resolved_competition_id = competition_id or derive_competition_id(competition_url)
    warnings: list[str] = []
    store = PgStore(
        competition_id=resolved_competition_id,
        dsn=settings.pg_dsn,
        embed_dim=settings.embed_dim,
    )

    try:
        await store.init()
        client = DeepSeekClient(api_key=settings.deepseek_api_key)
        plan_data = await _build_plan(
            competition_desc=competition_desc,
            client=client,
            model=settings.deepseek_v4_pro,
            warnings=warnings,
        )
        source_documents = await _collect_sources(
            plan_data=plan_data,
            competition_id=resolved_competition_id,
            settings=settings,
            warnings=warnings,
        )
        if not source_documents:
            raise RuntimeError("No source documents were collected")

        summarized_documents = await summarize_documents(
            client=client,
            docs=source_documents,
            model=settings.deepseek_v4_flash,
        )
        indexed_texts = [document.summary or document.content for document in summarized_documents]
        embeddings = embed_texts(indexed_texts, batch_size=settings.max_embed_batch_size)
        if len(embeddings) != len(summarized_documents):
            raise RuntimeError("Embedding count does not match document count")

        await store.upsert(summarized_documents, embeddings)
        retrieved_documents = await _retrieve_documents(
            store=store,
            plan_data=plan_data,
            top_k=settings.top_k,
            warnings=warnings,
        )
        report_path = _report_output_path(
            output_dir=output_dir,
            competition_id=resolved_competition_id,
        )
        report_text = _build_minimal_report_text(
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
        )
        generate_report(
            competition_name=resolved_competition_id,
            roadmap_text=report_text,
            sources=retrieved_documents,
            output_path=report_path,
        )

        return ResearchRunResult(
            competition_id=resolved_competition_id,
            report_path=str(report_path),
            num_documents=len(summarized_documents),
            num_sources=dict(Counter(document.source for document in summarized_documents)),
            warnings=warnings,
            duration_sec=round(time.perf_counter() - started_at, 3),
        )
    finally:
        await store.close()


async def summarize_documents(
    client: DeepSeekClient,
    docs: list[SourceDocument],
    model: str,
) -> list[SourceDocument]:
    from kaggle_researcher.summarizer import summarize_all

    return await summarize_all(client=client, docs=docs, model=model)


def derive_competition_id(competition_url: str) -> str:
    parsed_url = urlparse(competition_url)
    path_parts = [part for part in parsed_url.path.split("/") if part]
    competition_markers = {"competitions", "c"}

    for index, part in enumerate(path_parts):
        if part in competition_markers and index + 1 < len(path_parts):
            return _slugify(path_parts[index + 1])

    if path_parts:
        return _slugify(path_parts[-1])

    host_slug = _slugify(parsed_url.netloc)
    return host_slug or "unknown-competition"


async def _build_plan(
    competition_desc: str,
    client: DeepSeekClient,
    model: str,
    warnings: list[str],
) -> PlanData:
    try:
        return await plan(competition_desc, client, model)
    except Exception as exc:
        warnings.append(f"Planner failed; used fallback plan: {exc}")
        return fallback_plan(competition_desc)


async def _collect_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    documents.extend(
        await asyncio.to_thread(
            _collect_kaggle_sources,
            plan_data,
            competition_id,
            settings,
            warnings,
        )
    )
    documents.extend(
        await asyncio.to_thread(
            _collect_arxiv_sources,
            plan_data,
            competition_id,
            settings,
            warnings,
        )
    )
    return documents


def _collect_kaggle_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
) -> list[SourceDocument]:
    try:
        raw_notebooks = search_notebooks(plan_data.kaggle_queries, max_notebooks=settings.max_notebooks)
        return build_kaggle_documents(raw_notebooks, competition_id=competition_id)
    except Exception as exc:
        warnings.append(f"Kaggle source collection failed: {exc}")
        return []


def _collect_arxiv_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
) -> list[SourceDocument]:
    try:
        papers = search_arxiv(plan_data.arxiv_queries, max_papers=settings.max_papers)
        enriched_papers = enrich_with_pdf(papers, cache_dir=settings.pdf_cache_dir)
        return build_arxiv_documents(enriched_papers, competition_id=competition_id)
    except Exception as exc:
        warnings.append(f"arXiv source collection failed: {exc}")
        return []


async def _retrieve_documents(
    store: PgStore,
    plan_data: PlanData,
    top_k: int,
    warnings: list[str],
) -> list[RetrievedDocument]:
    retrieved_by_id: dict[str, RetrievedDocument] = {}
    for query in _retrieval_queries(plan_data):
        try:
            results = await hybrid_search(store, query, top_k=top_k)
        except Exception as exc:
            warnings.append(f"Retrieval failed for query {query!r}: {exc}")
            continue
        for document in results:
            existing = retrieved_by_id.get(document.id)
            if existing is None or document.rrf_score > existing.rrf_score:
                retrieved_by_id[document.id] = document

    return sorted(
        retrieved_by_id.values(),
        key=lambda document: document.rrf_score,
        reverse=True,
    )


def _retrieval_queries(plan_data: PlanData) -> list[str]:
    queries = [
        *plan_data.kaggle_queries[:2],
        *plan_data.arxiv_queries[:2],
        f"{plan_data.domain} {plan_data.task_type} {plan_data.metric}".strip(),
    ]
    return [query for query in dict.fromkeys(queries) if query]


def _build_minimal_report_text(
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
) -> str:
    source_lines = [
        f"- {document.title} ({document.source})"
        for document in retrieved_documents[:10]
    ]
    if not source_lines:
        source_lines = ["- No retrieved documents were available for the minimal report."]

    return "\n".join(
        [
            "# Minimal Research Report",
            "",
            "## Competition Summary",
            competition_desc.strip(),
            "",
            "## Plan",
            f"- Task type: {plan_data.task_type}",
            f"- Metric: {plan_data.metric}",
            f"- Domain: {plan_data.domain}",
            "",
            "## Retrieved Sources",
            *source_lines,
            "",
            "## Next Steps",
            "- Build a trustworthy validation baseline before trying model complexity.",
            "- Use the retrieved sources as hypotheses, not as proof of dataset-specific behavior.",
        ]
    )


def _report_output_path(output_dir: str | Path, competition_id: str) -> Path:
    return Path(output_dir) / f"{competition_id}_research_report.docx"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown-competition"


async def run() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.competition_url or not args.competition_desc:
        print("Provide competition_url and competition_desc to run the minimal pipeline.")
        return 0

    result = await run_research(
        competition_url=args.competition_url,
        competition_desc=args.competition_desc,
        competition_id=args.competition_id,
        output_dir=args.output_dir,
    )
    print(result.model_dump_json(indent=2))
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
