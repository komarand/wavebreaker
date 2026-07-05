from __future__ import annotations

import argparse
import asyncio
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from tqdm.auto import tqdm

from kaggle_researcher.agents.arxiv_agent import (
    build_arxiv_documents,
    enrich_with_pdf,
    search_arxiv,
)
from kaggle_researcher.agents.kaggle_agent import (
    build_kaggle_documents,
    get_notebook_content,
    search_notebooks,
)
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
    parser.add_argument("--report-path", help="Optional exact report output path")
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Overwrite the exact report path instead of creating a safe new name",
    )
    parser.add_argument(
        "--report-naming-strategy",
        choices=("timestamp", "increment"),
        default="timestamp",
        help="Safe report naming strategy when the target already exists",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and stage messages",
    )
    return parser


async def run_research(
    competition_url: str,
    competition_desc: str,
    *,
    competition_id: str | None = None,
    output_dir: str | Path = "./reports",
    report_path: str | Path | None = None,
    overwrite_report: bool = False,
    report_naming_strategy: str = "timestamp",
    show_progress: bool = True,
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
        _stage("[1/8] Planning search queries...", show_progress)
        await store.init()
        client = DeepSeekClient(api_key=settings.deepseek_api_key)
        plan_data = await _build_plan(
            competition_desc=competition_desc,
            client=client,
            model=settings.deepseek_v4_pro,
            warnings=warnings,
        )
        _stage("[2/8] Collecting Kaggle notebooks...", show_progress)
        source_documents = await _collect_sources(
            plan_data=plan_data,
            competition_id=resolved_competition_id,
            settings=settings,
            warnings=warnings,
            show_progress=show_progress,
        )
        if not source_documents:
            raise RuntimeError("No source documents were collected")

        _stage("[4/8] Summarizing documents...", show_progress)
        summarized_documents = await summarize_documents(
            client=client,
            docs=source_documents,
            model=settings.deepseek_v4_flash,
            show_progress=show_progress,
        )
        _stage("[5/8] Embedding and indexing...", show_progress)
        indexed_texts = [document.summary or document.content for document in summarized_documents]
        embeddings = _embed_documents(
            indexed_texts,
            batch_size=settings.max_embed_batch_size,
            show_progress=show_progress,
        )
        if len(embeddings) != len(summarized_documents):
            raise RuntimeError("Embedding count does not match document count")

        _stage("Indexing documents in pgvector...", show_progress)
        await store.upsert(summarized_documents, embeddings)
        _stage("[6/8] Retrieving evidence...", show_progress)
        retrieved_documents = await _retrieve_documents(
            store=store,
            plan_data=plan_data,
            top_k=settings.top_k,
            warnings=warnings,
            show_progress=show_progress,
        )
        _stage("[7/8] Running reasoning chain...", show_progress)
        _stage("Skipping full reasoning chain in minimal pipeline.", show_progress)
        target_report_path = Path(report_path) if report_path is not None else _report_output_path(
            output_dir=output_dir,
            competition_id=resolved_competition_id,
        )
        report_text = _build_minimal_report_text(
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
        )
        _stage("[8/8] Generating DOCX report...", show_progress)
        actual_report_path = generate_report(
            competition_name=resolved_competition_id,
            roadmap_text=report_text,
            sources=retrieved_documents,
            output_path=target_report_path,
            overwrite=overwrite_report,
            naming_strategy=report_naming_strategy,
        )
        _stage(f"Report saved to: {actual_report_path}", show_progress)

        return ResearchRunResult(
            competition_id=resolved_competition_id,
            report_path=str(actual_report_path),
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
    *,
    show_progress: bool = True,
) -> list[SourceDocument]:
    from kaggle_researcher.summarizer import summarize_one

    semaphore = asyncio.Semaphore(8)
    results: list[SourceDocument | None] = [None] * len(docs)

    async def summarize_with_index(index: int, doc: SourceDocument) -> tuple[int, SourceDocument]:
        async with semaphore:
            return index, await summarize_one(client=client, doc=doc, model=model)

    tasks = [summarize_with_index(index, doc) for index, doc in enumerate(docs)]
    progress = tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Summarizing documents",
        unit="doc",
        disable=not show_progress,
    )
    for completed in progress:
        index, summarized_doc = await completed
        results[index] = summarized_doc

    return [doc for doc in results if doc is not None]


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
    show_progress: bool,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    documents.extend(
        await asyncio.to_thread(
            _collect_kaggle_sources,
            plan_data,
            competition_id,
            settings,
            warnings,
            show_progress,
        )
    )
    _stage("[3/8] Collecting papers and repos...", show_progress)
    _stage("Processing arXiv PDFs...", show_progress)
    documents.extend(
        await asyncio.to_thread(
            _collect_arxiv_sources,
            plan_data,
            competition_id,
            settings,
            warnings,
            show_progress,
        )
    )
    return documents


def _collect_kaggle_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
) -> list[SourceDocument]:
    raw_notebooks = search_notebooks(
        plan_data.kaggle_queries,
        competition_id=competition_id,
        max_notebooks=settings.max_notebooks,
    )
    usable_notebooks: list[dict[str, object]] = []

    for notebook in tqdm(
        raw_notebooks,
        desc="Pulling Kaggle notebooks",
        unit="notebook",
        disable=not show_progress,
    ):
        notebook_id = _notebook_id(notebook)
        if not notebook_id:
            warnings.append(f"Kaggle notebook metadata missing id/ref: {notebook}")
            continue

        content = str(notebook.get("content") or "")
        if not content:
            try:
                content = get_notebook_content(notebook_id)
            except Exception as exc:
                warnings.append(f"Kaggle notebook pull failed for {notebook_id}: {exc}")
                continue

        if not content.strip():
            warnings.append(f"Kaggle notebook {notebook_id} produced empty extracted content")
            continue

        notebook_with_content = dict(notebook)
        notebook_with_content["content"] = content
        usable_notebooks.append(notebook_with_content)

    if not usable_notebooks:
        warnings.append(
            "Kaggle agent returned 0 usable notebooks. CLI auth works, so check "
            "kaggle_agent.py pull/content extraction."
        )
        raise RuntimeError(
            "Kaggle retrieval returned 0 usable notebooks. "
            "CLI auth works, so check Kaggle pull/content extraction."
        )

    return build_kaggle_documents(usable_notebooks, competition_id=competition_id)


def _collect_arxiv_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
) -> list[SourceDocument]:
    try:
        papers = search_arxiv(plan_data.arxiv_queries, max_papers=settings.max_papers)
        enriched_papers = enrich_with_pdf(papers, cache_dir=settings.pdf_cache_dir)
        return build_arxiv_documents(enriched_papers, competition_id=competition_id)
    except Exception as exc:
        warnings.append(f"arXiv source collection failed: {exc}")
        return []


def _notebook_id(notebook: dict[str, object]) -> str | None:
    for key in ("id", "kernel_ref", "ref", "kernelRef"):
        value = notebook.get(key)
        if value:
            return str(value)
    metadata = notebook.get("metadata")
    if isinstance(metadata, dict):
        ref = metadata.get("ref")
        if ref:
            return str(ref)
    return None


async def _retrieve_documents(
    store: PgStore,
    plan_data: PlanData,
    top_k: int,
    warnings: list[str],
    show_progress: bool = True,
) -> list[RetrievedDocument]:
    retrieved_by_id: dict[str, RetrievedDocument] = {}
    queries = _retrieval_queries(plan_data)
    for query in tqdm(
        queries,
        desc="Retrieving evidence",
        unit="query",
        disable=not show_progress,
    ):
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


def _embed_documents(texts: list[str], batch_size: int, show_progress: bool) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    batches = [
        texts[start : start + batch_size]
        for start in range(0, len(texts), batch_size)
    ]
    for batch in tqdm(
        batches,
        desc="Embedding documents",
        unit="batch",
        disable=not show_progress,
    ):
        embeddings.extend(embed_texts(batch, batch_size=batch_size))
    return embeddings


def _stage(message: str, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(message)


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
        report_path=args.report_path,
        overwrite_report=args.overwrite_report,
        report_naming_strategy=args.report_naming_strategy,
        show_progress=not args.no_progress,
    )
    print(f"Report saved to: {result.report_path}")
    print(result.model_dump_json(indent=2))
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
