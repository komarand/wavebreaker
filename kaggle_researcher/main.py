from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel
from tqdm.auto import tqdm

from kaggle_researcher.agents.arxiv_agent import (
    build_arxiv_documents,
    enrich_with_pdf,
    search_arxiv,
)
from kaggle_researcher.agents.github_agent import (
    build_github_documents,
    search_repos,
)
from kaggle_researcher.agents.kaggle_agent import (
    build_kaggle_documents,
    get_notebook_content,
    search_notebooks,
)
from kaggle_researcher.agents.paper_search_agent import search_paper_sources
from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.config import load_config
from kaggle_researcher.embedder import embed_texts
from kaggle_researcher.planner import fallback_plan, plan
from kaggle_researcher.report.docx_generator import generate_report
from kaggle_researcher.reasoning.experiment_planner import plan_experiments
from kaggle_researcher.reasoning.leaderboard_auditor import audit_leaderboard_risk
from kaggle_researcher.reasoning.leakage_risk_analyst import analyze_leakage_risk
from kaggle_researcher.reasoning.metric_specialist import analyze_metric
from kaggle_researcher.reasoning.provenance import attach_default_provenance, provenance_summary
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS, compose_report
from kaggle_researcher.reasoning.skeptical_reviewer import review
from kaggle_researcher.reasoning.validation_architect import design_validation
from kaggle_researcher.retrieval.source_quality import rerank_by_source_quality, source_quality_summary
from kaggle_researcher.retriever import hybrid_search
from kaggle_researcher.schemas import PlanData, ResearchRunResult, RetrievedDocument, SourceDocument
from kaggle_researcher.store.pg_store import PgStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_researcher",
        description="KaggleResearcher research pipeline.",
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
    parser.add_argument(
        "--report-mode",
        choices=("full", "minimal"),
        default="full",
        help="Report mode. Full is the default; minimal must be requested explicitly.",
    )
    parser.add_argument(
        "--allow-minimal-fallback",
        action="store_true",
        help="Allow a minimal report if full reasoning fails.",
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
    report_mode: Literal["full", "minimal"] = "full",
    allow_minimal_fallback: bool = False,
) -> ResearchRunResult:
    started_at = time.perf_counter()
    settings = load_config()
    resolved_competition_id = competition_id or derive_competition_id(competition_url)
    warnings: list[str] = []
    run_dir = _create_run_dir(resolved_competition_id)
    models_used = _models_used(settings)
    _write_json_artifact(run_dir, "models_used.json", models_used)
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
        _write_json_artifact(run_dir, "plan.json", plan_data)
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
        num_sources = _source_counts(source_documents)
        _add_missing_source_warnings(num_sources, warnings)
        _write_json_artifact(run_dir, "source_counts.json", num_sources)

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
        _write_json_artifact(run_dir, "documents_indexed.json", summarized_documents)
        _stage("[6/8] Retrieving evidence...", show_progress)
        retrieved_documents = await _retrieve_documents(
            store=store,
            plan_data=plan_data,
            competition_id=resolved_competition_id,
            run_dir=run_dir,
            top_k=settings.top_k,
            warnings=warnings,
            show_progress=show_progress,
        )
        _write_json_artifact(run_dir, "retrieved_documents.json", retrieved_documents)
        _stage("[7/8] Running reasoning chain...", show_progress)
        target_report_path = Path(report_path) if report_path is not None else _report_output_path(
            output_dir=output_dir,
            competition_id=resolved_competition_id,
        )
        report_text = await _build_report_text(
            report_mode=report_mode,
            allow_minimal_fallback=allow_minimal_fallback,
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            client=client,
            model=settings.deepseek_v4_pro,
            run_dir=run_dir,
            warnings=warnings,
            show_progress=show_progress,
        )
        _write_text_artifact(run_dir, "roadmap.md", report_text)
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
        _write_json_artifact(run_dir, "warnings.json", warnings)

        return ResearchRunResult(
            competition_id=resolved_competition_id,
            report_path=str(actual_report_path),
            num_documents=len(summarized_documents),
            num_sources=num_sources,
            warnings=warnings,
            duration_sec=round(time.perf_counter() - started_at, 3),
            report_mode=report_mode,
            run_artifacts_path=str(run_dir),
            retrieved_evidence_count=len(retrieved_documents),
        )
    except Exception:
        _write_json_artifact(run_dir, "warnings.json", warnings)
        raise
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
    documents.extend(
        await _collect_paper_sources(
            plan_data=plan_data,
            competition_id=competition_id,
            settings=settings,
            warnings=warnings,
        )
    )
    documents.extend(
        await _collect_github_sources(
            plan_data=plan_data,
            competition_id=competition_id,
            settings=settings,
            warnings=warnings,
            show_progress=show_progress,
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


async def _collect_paper_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
) -> list[SourceDocument]:
    paper_queries = [*plan_data.arxiv_queries]
    if not paper_queries:
        return []
    raw_papers = await search_paper_sources(
        queries=paper_queries,
        max_results=settings.max_papers,
        warnings=warnings,
    )
    return build_arxiv_documents(raw_papers, competition_id=competition_id)


async def _collect_github_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
) -> list[SourceDocument]:
    if not plan_data.github_queries:
        return []
    _stage("Collecting GitHub repos...", show_progress)
    try:
        raw_repos = await search_repos(
            plan_data.github_queries,
            token=settings.github_token,
            max_repos=settings.max_repos,
        )
        return build_github_documents(raw_repos, competition_id=competition_id)
    except Exception as exc:
        warnings.append(f"GitHub source collection failed: {exc}")
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
    competition_id: str,
    run_dir: Path,
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

    fused_documents = sorted(
        retrieved_by_id.values(),
        key=lambda document: document.rrf_score,
        reverse=True,
    )
    reranked = rerank_by_source_quality(
        fused_documents,
        competition_id=competition_id,
        plan_data=plan_data.model_dump(),
    )
    _write_json_artifact(run_dir, "source_quality_summary.json", source_quality_summary(reranked))
    return reranked


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


async def _build_report_text(
    report_mode: Literal["full", "minimal"],
    allow_minimal_fallback: bool,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    client: DeepSeekClient,
    model: str,
    run_dir: Path,
    warnings: list[str],
    show_progress: bool,
) -> str:
    if report_mode == "minimal":
        return _build_minimal_report_text(competition_desc, plan_data, retrieved_documents)

    try:
        return await _build_full_report_text(
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
            run_dir=run_dir,
            show_progress=show_progress,
        )
    except Exception as exc:
        if allow_minimal_fallback:
            warnings.append(f"Full report failed; generated minimal fallback: {exc}")
            return _build_minimal_report_text(competition_desc, plan_data, retrieved_documents)
        raise


async def _build_full_report_text(
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    client: DeepSeekClient,
    model: str,
    run_dir: Path,
    show_progress: bool,
) -> str:
    domain_patterns: list[dict[str, Any]] = []
    _write_json_artifact(run_dir, "domain_patterns.json", domain_patterns)
    provenance_sections: dict[str, Any] = {}

    validation_result = await _run_reasoning_stage(
        "validation_architect",
        lambda: design_validation(
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "validation_result.json", validation_result)
    provenance_sections["validation"] = attach_default_provenance(
        "validation",
        validation_result.model_dump(mode="json"),
        retrieved_documents,
    )

    leakage_result = await _run_reasoning_stage(
        "leakage_risk_analyst",
        lambda: analyze_leakage_risk(
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "leakage_result.json", leakage_result)
    provenance_sections["leakage"] = attach_default_provenance(
        "leakage",
        leakage_result.model_dump(mode="json"),
        retrieved_documents,
    )

    metric_result = await _run_reasoning_stage(
        "metric_specialist",
        lambda: analyze_metric(
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "metric_result.json", metric_result)
    provenance_sections["metric"] = attach_default_provenance(
        "metric",
        metric_result.model_dump(mode="json"),
        retrieved_documents,
    )

    experiments = await _run_reasoning_stage(
        "experiment_planner",
        lambda: plan_experiments(
            validation_result=validation_result,
            leakage_result=leakage_result,
            metric_result=metric_result,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "experiments.json", experiments)
    provenance_sections["experiments"] = attach_default_provenance(
        "experiments",
        [item.model_dump(mode="json") for item in experiments],
        retrieved_documents,
    )

    lb_audit = await _run_reasoning_stage(
        "leaderboard_auditor",
        lambda: audit_leaderboard_risk(
            competition_desc=competition_desc,
            plan_data=plan_data,
            validation_result=validation_result,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "leaderboard_audit.json", lb_audit)
    provenance_sections["leaderboard"] = attach_default_provenance(
        "leaderboard",
        lb_audit.model_dump(mode="json"),
        retrieved_documents,
    )

    draft_sections = {
        "validation": provenance_sections["validation"],
        "leakage": provenance_sections["leakage"],
        "metric": provenance_sections["metric"],
        "experiments": provenance_sections["experiments"],
        "leaderboard": provenance_sections["leaderboard"],
    }
    review_result = await _run_reasoning_stage(
        "skeptical_reviewer",
        lambda: review(
            draft_sections=draft_sections,
            retrieved_documents=retrieved_documents,
            client=client,
            model=model,
            artifact_dir=run_dir,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "review_result.json", review_result)
    provenance_sections["review"] = attach_default_provenance(
        "review",
        review_result.model_dump(mode="json"),
        retrieved_documents,
    )
    _write_json_artifact(run_dir, "provenance_summary.json", provenance_summary(provenance_sections))

    roadmap_text = await _run_reasoning_stage(
        "report_composer",
        lambda: compose_report(
            competition_desc=competition_desc,
            plan_data=plan_data,
            domain_patterns=domain_patterns,
            validation_result=validation_result,
            leakage_result=leakage_result,
            metric_result=metric_result,
            experiments=experiments,
            lb_audit=lb_audit,
            review=review_result,
            client=client,
            model=model,
        ),
        show_progress,
    )
    try:
        validate_full_roadmap(roadmap_text)
    except Exception as exc:
        raise RuntimeError(f"Full report generation failed at stage 'roadmap_validation': {exc}") from exc
    return roadmap_text


async def _run_reasoning_stage(stage_name: str, factory: Any, show_progress: bool) -> Any:
    _stage(f"Running {stage_name}...", show_progress)
    try:
        return await factory()
    except RuntimeError as exc:
        if str(exc).startswith("Full report generation failed at stage"):
            raise
        raise RuntimeError(
            f"Full report generation failed at stage '{stage_name}': {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Full report generation failed at stage '{stage_name}': {exc}"
        ) from exc


def validate_full_roadmap(roadmap_text: str) -> None:
    if "Minimal Research Report" in roadmap_text:
        raise RuntimeError("report looks like a minimal/fallback report")
    matched_sections = [
        heading for heading in SECTION_HEADINGS if heading.lower() in roadmap_text.lower()
    ]
    if len(matched_sections) < 10:
        raise RuntimeError(
            "report is missing expected v4 sections "
            f"({len(matched_sections)}/10 matched)"
        )
    if len(roadmap_text) < 4000:
        raise RuntimeError("report is too short for a full v4 roadmap")


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


def _create_run_dir(competition_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{competition_id}_{timestamp}"
    counter = 2
    while run_dir.exists():
        run_dir = Path("runs") / f"{competition_id}_{timestamp}_{counter:03d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json_artifact(run_dir: Path, filename: str, value: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_text_artifact(run_dir: Path, filename: str, value: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(value, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _source_counts(documents: list[SourceDocument]) -> dict[str, int]:
    counts = Counter(document.source for document in documents)
    legacy_count = counts.get("papers_with_code_legacy", 0)
    hf_count = counts.get("huggingface_papers", 0)
    return {
        "kaggle": counts.get("kaggle", 0),
        "arxiv": counts.get("arxiv", 0),
        "huggingface_papers": hf_count,
        "papers_with_code_legacy": legacy_count,
        "papers_with_code": hf_count + legacy_count,
        "github": counts.get("github", 0),
    }


def _models_used(settings: Any) -> dict[str, str]:
    reasoning_model = settings.deepseek_v4_pro
    return {
        "planner": settings.deepseek_v4_pro,
        "summarizer": settings.deepseek_v4_flash,
        "validation_architect": reasoning_model,
        "leakage_risk_analyst": reasoning_model,
        "metric_specialist": reasoning_model,
        "experiment_planner": reasoning_model,
        "leaderboard_auditor": reasoning_model,
        "skeptical_reviewer": reasoning_model,
        "report_composer": reasoning_model,
        "embedder": settings.embed_model,
    }


def _add_missing_source_warnings(num_sources: dict[str, int], warnings: list[str]) -> None:
    if num_sources.get("github", 0) == 0:
        warnings.append("GitHub source count is 0. Check GITHUB_TOKEN or query quality.")
    if num_sources.get("huggingface_papers", 0) == 0:
        warnings.append(
            "Hugging Face Papers source count is 0. Falling back to arXiv-only academic retrieval."
        )


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
        report_mode=args.report_mode,
        allow_minimal_fallback=args.allow_minimal_fallback,
    )
    print("Research run complete.")
    print(f"Report mode: {result.report_mode}")
    print(f"Report saved to: {result.report_path}")
    print(f"Run artifacts saved to: {result.run_artifacts_path}")
    print("Source counts:")
    for source, count in result.num_sources.items():
        print(f"  {source}: {count}")
    print(f"Retrieved evidence: {result.retrieved_evidence_count}")
    if result.run_artifacts_path:
        models_path = Path(result.run_artifacts_path) / "models_used.json"
        if models_path.exists():
            models_used = json.loads(models_path.read_text(encoding="utf-8"))
            print("Models used:")
            print(f"  planner: {models_used.get('planner')}")
            print(f"  summarizer: {models_used.get('summarizer')}")
            print(f"  reasoning/report: {models_used.get('report_composer')}")
            print(f"  embedder: {models_used.get('embedder')}")
    print(f"Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  - {warning}")
    print(result.model_dump_json(indent=2))
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
