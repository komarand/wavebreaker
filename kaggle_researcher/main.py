from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel
from tqdm.auto import tqdm

from kaggle_researcher.agents.arxiv_agent import (
    arxiv_pdf_parser_fingerprint,
    build_arxiv_documents,
    enrich_with_pdf,
    search_arxiv,
)
from kaggle_researcher.agents.github_agent import (
    build_github_documents,
    search_repos,
    github_readme_parser_fingerprint,
)
from kaggle_researcher.agents.kaggle_agent import (
    build_kaggle_documents,
    get_notebook_content,
    search_notebooks,
    kaggle_notebook_parser_fingerprint,
)
from kaggle_researcher.agents.paper_search_agent import search_paper_sources
from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.contracts.research_hypotheses import (
    ResearchHypotheses,
    migrate_research_hypotheses_payload,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.contracts.eda_task_plan import (
    EdaTaskPlan,
    migrate_eda_task_plan_payload,
    validate_research_artifact_bundle,
    write_eda_task_plan_atomic,
)
from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    ReasoningStageResult,
    ResearchStageResult,
    load_eda_publication_bundle,
    load_experiment_plan,
    load_skeptical_review,
    write_json_atomic,
)
from kaggle_researcher.contracts.registries import build_contract_registries
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.contracts.experiments import ExperimentPlan
from kaggle_researcher.config import load_config
from kaggle_researcher.eda import orchestrator as eda_orchestrator
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaRunConfig, ResearchHypotheses
from kaggle_researcher.contracts.research_to_eda import (
    canonicalize_research_to_eda_contract,
    require_valid_research_to_eda_contract,
)
from kaggle_researcher.embedder import embed_texts
from kaggle_researcher.planner import fallback_plan, plan
from kaggle_researcher.research_scout import (
    build_deterministic_research_scout_fallback,
    build_research_hypotheses,
    build_research_scout_summary,
    run_research_scout,
    split_eda_task_plan,
    validate_research_hypotheses,
)
from kaggle_researcher.report.docx_generator import generate_report
from kaggle_researcher.reasoning.experiment_planner import plan_experiments
from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyResult,
    render_final_strategy,
    render_final_strategy_summary,
    synthesize_final_strategy,
    validate_rendered_strategy_quality,
)
from kaggle_researcher.reasoning.leaderboard_auditor import audit_leaderboard_risk
from kaggle_researcher.reasoning.leakage_risk_analyst import analyze_leakage_risk
from kaggle_researcher.reasoning.metric_specialist import analyze_metric
from kaggle_researcher.reasoning.provenance import attach_default_provenance, provenance_summary
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS, compose_report
from kaggle_researcher.reasoning.skeptical_reviewer import review
from kaggle_researcher.reasoning.validation_architect import design_validation
from kaggle_researcher.retrieval.source_quality import rerank_by_source_quality, source_quality_summary
from kaggle_researcher.retriever import hybrid_search
from kaggle_researcher.schemas import (
    PlanData,
    ResearchRunResult,
    RetrievedDocument,
    ReviewResult,
    SourceDocument,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    ValidationResult,
)
from kaggle_researcher.store.domain_memory import DomainMemory
from kaggle_researcher.store.pg_store import PgStore
from kaggle_researcher.store.source_registry_store import SourceRegistryStore
from kaggle_researcher.source_registry.processing_cache import process_source_documents
from kaggle_researcher.source_registry.schemas import (
    CachePolicy,
    CacheRunTelemetry,
    SearchCacheEntry,
    SourceRefreshMode,
)
from kaggle_researcher.source_registry.fingerprints import build_search_request_fingerprint
from kaggle_researcher.source_registry.hashing import sha256_text
from kaggle_researcher.source_registry.identity import canonicalize_source_identity
from kaggle_researcher.source_registry.search_cache import normalize_search_query
from kaggle_researcher.source_registry.telemetry import write_cache_report
from kaggle_researcher.workflow import (
    FinalSynthesisDegradedError,
    FinalSynthesisStageStatus,
    WorkflowStatus,
)


FAST_MAX_NOTEBOOKS = 3
FAST_MAX_PAPERS = 3
FAST_MAX_REPOS = 2
FAST_MAX_RETRIEVAL_QUERIES = 2
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_researcher",
        description="KaggleResearcher research pipeline. Use 'full-run --help' for the canonical end-to-end command.",
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
        help="Compatibility alias for --mode full|minimal. Full is the default.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "scout", "minimal"),
        default="full",
        help="Pipeline mode. Scout writes Research Scout JSON artifacts without a DOCX report.",
    )
    parser.add_argument(
        "--allow-minimal-fallback",
        action="store_true",
        help="Allow a minimal report if full reasoning fails.",
    )
    parser.add_argument(
        "--allow-partial-scout-output",
        action="store_true",
        help="Write research_hypotheses_partial.json when Scout validation fails.",
    )
    parser.add_argument(
        "--write-eda-plan",
        action="store_true",
        help="Write Research Scout hypotheses and EDA task plan artifacts during a research run.",
    )
    parser.add_argument(
        "--run-eda",
        dest="execute_eda",
        action="store_true",
        help="Run the EDA Engine after writing Research Scout outputs.",
    )
    parser.add_argument(
        "--local-dataset-path",
        type=Path,
        help="Path to a local competition dataset for optional EDA execution.",
    )
    parser.add_argument(
        "--eda-output-dir",
        type=Path,
        help="Directory for optional EDA Engine run outputs.",
    )
    parser.add_argument(
        "--download-dataset",
        dest="download_dataset",
        action="store_true",
        default=True,
        help="Allow EDA to download the Kaggle dataset when no local dataset path is provided.",
    )
    parser.add_argument(
        "--no-download-dataset",
        dest="download_dataset",
        action="store_false",
        help="Disable Kaggle dataset download during EDA.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force a fresh Kaggle dataset download before EDA.",
    )
    parser.add_argument(
        "--enable-p1-modules",
        action="store_true",
        help="Enable P1 EDA modules such as relationships, drift, and feature probes.",
    )
    parser.add_argument(
        "--enable-baseline",
        action="store_true",
        help="Enable the optional EDA baseline runner.",
    )
    parser.add_argument(
        "--enable-baseline-ablations",
        action="store_true",
        help="Enable optional fold-safe EDA baseline feature-block ablations.",
    )
    parser.add_argument(
        "--enable-interaction-diagnostics",
        action="store_true",
        help="Enable optional bounded EDA interaction diagnostics.",
    )
    parser.add_argument(
        "--enable-source-claim-validation",
        action="store_true",
        help="Validate collected source claims against current EDA evidence.",
    )
    parser.add_argument(
        "--enable-visual-diagnostics",
        action="store_true",
        help="Generate optional bounded EDA visual diagnostic artifacts.",
    )
    parser.add_argument(
        "--enable-slice-diagnostics",
        action="store_true",
        help="Enable fold-safe EDA slice performance diagnostics.",
    )
    parser.add_argument(
        "--research-hypotheses-path",
        type=Path,
        help="Existing research_hypotheses.json to use for EDA or final synthesis.",
    )
    parser.add_argument(
        "--eda-task-plan-path",
        type=Path,
        help="Existing eda_task_plan.json to use for EDA.",
    )
    parser.add_argument(
        "--eda-evidence-pack-path",
        type=Path,
        help="Existing EDA evidence pack JSON to use for final synthesis.",
    )
    parser.add_argument(
        "--eda-summary-path",
        type=Path,
        help="Existing EDA summary Markdown to include in final synthesis context.",
    )
    parser.add_argument(
        "--final-synthesis",
        action="store_true",
        help="Run final strategy synthesis after EDA evidence is available.",
    )
    parser.add_argument(
        "--final-output-dir",
        type=Path,
        help="Directory for final strategy outputs. Defaults to the research run artifact directory.",
    )
    parser.add_argument(
        "--require-valid-final-synthesis",
        action="store_true",
        help=(
            "Fail after writing artifacts when final synthesis can only produce a "
            "deterministic degraded fallback."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Skip GitHub repository collection.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce collection limits and skip the skeptical reviewer pass.",
    )
    parser.add_argument(
        "--source-refresh", choices=("auto", "always", "never"), default=None,
        help="Source discovery/content refresh policy. CLI value overrides SOURCE_REFRESH_MODE.",
    )
    parser.add_argument(
        "--rebuild-source-artifacts", default="",
        help="Comma-separated stages: parsed,summaries,embeddings,static-analysis,all.",
    )
    parser.add_argument(
        "--no-source-cache", action="store_true",
        help="Bypass compatible artifact reuse for this run without deleting stored cache records.",
    )
    parser.add_argument(
        "--source-cache-report", action="store_true",
        help="Write source_cache_report.json with aggregate and per-source cache decisions.",
    )
    parser.add_argument(
        "--source-registry-migrate", action="store_true",
        help="Migrate legacy documents into the source registry before the run.",
    )
    return parser


def build_full_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_researcher full-run",
        description="Run Research Scout, EDA, reasoning, and final strategy synthesis as one reproducible run.",
    )
    parser.add_argument("--competition-id", required=True, help="Kaggle competition slug or identifier")
    parser.add_argument("--competition-url", help="Kaggle competition URL")
    parser.add_argument("--competition-description", default="", help="Competition description and metric context")
    parser.add_argument("--local-dataset-path", type=Path, help="Path to local competition data")
    parser.add_argument("--output-root", type=Path, default=Path("runs"), help="Parent directory for full runs")
    parser.add_argument("--resume-run-dir", type=Path, help="Reuse a completed full-run directory when its config and artifacts match")
    parser.add_argument("--force-rerun-stage", action="append", default=[], help="Stage ID to rerun; may be repeated")
    parser.add_argument("--profile", choices=("minimal", "standard", "full"), default="standard", help="Execution profile")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after an optional-stage failure")
    parser.add_argument(
        "--require-valid-final-synthesis",
        action="store_true",
        help="Fail after preserving artifacts when final synthesis is degraded.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")
    parser.add_argument("--download-dataset", dest="download_dataset", action="store_true", default=True, help="Allow dataset download when local data is absent")
    parser.add_argument("--no-download-dataset", dest="download_dataset", action="store_false", help="Disable dataset download")
    for flag, help_text in (
        ("--enable-p1-modules", "Enable optional P1 EDA modules."),
        ("--enable-baseline", "Enable the EDA baseline runner."),
        ("--enable-baseline-ablations", "Enable fold-safe baseline ablations."),
        ("--enable-interaction-diagnostics", "Enable bounded interaction diagnostics."),
        ("--enable-slice-diagnostics", "Enable fold-safe slice diagnostics."),
        ("--enable-source-claim-validation", "Validate source claims against EDA evidence."),
        ("--enable-visual-diagnostics", "Generate EDA visual diagnostics."),
    ):
        parser.add_argument(flag, action="store_true", help=help_text)
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
    mode: Literal["full", "scout", "minimal"] = "full",
    allow_minimal_fallback: bool = False,
    allow_partial_scout_output: bool = False,
    write_eda_plan: bool = False,
    execute_eda: bool = False,
    local_dataset_path: str | Path | None = None,
    eda_output_dir: str | Path | None = None,
    download_dataset: bool = True,
    force_download: bool = False,
    enable_p1_modules: bool = False,
    enable_baseline: bool = False,
    enable_baseline_ablations: bool = False,
    enable_interaction_diagnostics: bool = False,
    enable_source_claim_validation: bool = False,
    enable_visual_diagnostics: bool = False,
    enable_slice_diagnostics: bool = False,
    research_hypotheses_path: str | Path | None = None,
    eda_task_plan_path: str | Path | None = None,
    eda_evidence_pack_path: str | Path | None = None,
    eda_summary_path: str | Path | None = None,
    final_synthesis: bool = False,
    final_output_dir: str | Path | None = None,
    require_valid_final_synthesis: bool = False,
    debug: bool = False,
    no_github: bool = False,
    fast: bool = False,
    source_refresh: Literal["auto", "always", "never"] | None = None,
    rebuild_source_artifacts: set[str] | list[str] | tuple[str, ...] | str | None = None,
    no_source_cache: bool = False,
    source_cache_report: bool = False,
    **workflow_options: Any,
) -> ResearchRunResult:
    legacy_execute_eda = bool(workflow_options.pop("run" "_eda", False))
    if workflow_options:
        unexpected = ", ".join(sorted(workflow_options))
        raise TypeError(f"Unexpected workflow option(s): {unexpected}")
    execute_eda = execute_eda or legacy_execute_eda
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("kaggle_researcher").setLevel(logging.DEBUG)
    if mode == "minimal":
        report_mode = "minimal"
    elif report_mode == "minimal":
        mode = "minimal"
    provided_scout_paths = _provided_scout_paths(
        research_hypotheses_path=research_hypotheses_path,
        eda_task_plan_path=eda_task_plan_path,
    )
    provided_eda_evidence_pack_path = Path(eda_evidence_pack_path) if eda_evidence_pack_path else None
    provided_eda_summary_path = Path(eda_summary_path) if eda_summary_path else None
    effective_write_eda_plan = (
        write_eda_plan
        or execute_eda
        or (final_synthesis and not provided_scout_paths.get("research_hypotheses"))
    )
    if execute_eda and not effective_write_eda_plan and not provided_scout_paths:
        raise ValueError("--run-eda requires --write-eda-plan or existing Scout output paths.")
    if final_synthesis and not execute_eda and provided_eda_evidence_pack_path is None:
        raise ValueError(
            "--final-synthesis requires --run-eda or --eda-evidence-pack-path."
        )

    started_at = time.perf_counter()
    settings = load_config()
    resolved_competition_id = competition_id or derive_competition_id(competition_url)
    warnings: list[str] = []
    run_dir = _create_run_dir(resolved_competition_id)
    registry_enabled = bool(getattr(settings, "source_registry_enabled", False))
    cache_policy = _build_source_cache_policy(
        settings,
        source_refresh=source_refresh,
        rebuild_source_artifacts=rebuild_source_artifacts,
        no_source_cache=no_source_cache,
        source_cache_report=source_cache_report,
    )
    cache_telemetry = CacheRunTelemetry(
        run_id=run_dir.name,
        competition_id=resolved_competition_id,
    )
    source_cache_report_path: Path | None = None
    models_used = _models_used(settings)
    if mode == "scout":
        models_used["research_scout"] = settings.deepseek_v4_pro
    _write_json_artifact(run_dir, "models_used.json", models_used)
    store = PgStore(
        competition_id=resolved_competition_id,
        dsn=settings.pg_dsn,
        embed_dim=settings.embed_dim,
    )
    domain_memory = DomainMemory(dsn=settings.pg_dsn, embed_dim=settings.embed_dim)
    registry: SourceRegistryStore | None = None

    try:
        _stage("[1/8] Planning search queries...", show_progress)
        if registry_enabled:
            registry = SourceRegistryStore(
                dsn=settings.pg_dsn,
                embed_dim=settings.embed_dim,
                competition_id=resolved_competition_id,
            )
            await registry.init()
        else:
            await store.init()
        await domain_memory.init()
        client = DeepSeekClient(api_key=settings.deepseek_api_key)
        plan_data = await _build_plan(
            competition_desc=competition_desc,
            client=client,
            model=settings.deepseek_v4_pro,
            warnings=warnings,
        )
        _write_json_artifact(run_dir, "plan.json", plan_data)
        _stage("[2/8] Collecting Kaggle notebooks...", show_progress)
        discovery_requests = _source_discovery_requests(
            plan_data, resolved_competition_id, settings, no_github=no_github, fast=fast
        )
        cached_source_ids: list[str] | None = None
        if registry is not None:
            cached_source_ids = await _cached_discovery_source_ids(
                registry, discovery_requests, cache_policy, cache_telemetry
            )
        discovery_refreshed = cached_source_ids is None
        if cached_source_ids is not None:
            source_documents = await _load_cached_source_documents(
                registry, resolved_competition_id, source_ids=cached_source_ids
            )
        else:
            source_documents = await _collect_sources(
                plan_data=plan_data,
                competition_id=resolved_competition_id,
                settings=settings,
                warnings=warnings,
                show_progress=show_progress,
                no_github=no_github,
                fast=fast,
                registry=registry,
                cache_policy=cache_policy,
            )
        if not source_documents:
            raise RuntimeError("No source documents were collected")
        num_sources = _source_counts(source_documents)
        _add_missing_source_warnings(num_sources, warnings, no_github=no_github)
        _write_json_artifact(run_dir, "source_counts.json", num_sources)

        _stage("[4/8] Summarizing documents...", show_progress)
        _stage("[5/8] Embedding and indexing...", show_progress)
        if registry is not None:
            from kaggle_researcher.summarizer import summarize_one

            async def cached_summarize_one(document: SourceDocument) -> SourceDocument:
                return await summarize_one(client=client, doc=document, model=settings.deepseek_v4_flash)

            def cached_embed_one(text: str) -> list[float]:
                values = _embed_documents(
                    [text], batch_size=settings.max_embed_batch_size, show_progress=show_progress
                )
                if not values:
                    raise RuntimeError("No embedding was generated")
                return values[0]

            def cached_embed_many(texts: list[str]) -> list[list[float]]:
                return _embed_documents(
                    texts, batch_size=settings.max_embed_batch_size, show_progress=show_progress
                )

            summarized_documents, embeddings, processing_results = await process_source_documents(
                source_documents,
                competition_id=resolved_competition_id,
                run_id=run_dir.name,
                registry=registry,
                cache_policy=cache_policy,
                summarize_one=cached_summarize_one,
                embed_one=cached_embed_one,
                embed_many=cached_embed_many,
                summary_model=settings.deepseek_v4_flash,
                embed_model=settings.embed_model,
                telemetry=cache_telemetry,
            )
            warnings.extend(
                warning for processing_result in processing_results
                for warning in processing_result.warnings if warning not in warnings
            )
            if discovery_refreshed:
                await _save_discovery_cache(
                    registry,
                    discovery_requests,
                    summarized_documents,
                    cache_policy,
                )
        else:
            summarized_documents = await summarize_documents(
                client=client,
                docs=source_documents,
                model=settings.deepseek_v4_flash,
                show_progress=show_progress,
            )
            indexed_texts = [document.summary or document.content for document in summarized_documents]
            embeddings = _embed_documents(
                indexed_texts,
                batch_size=settings.max_embed_batch_size,
                show_progress=show_progress,
            )
        if not embeddings:
            raise RuntimeError("No embeddings were generated")
        if len(embeddings) != len(summarized_documents):
            raise RuntimeError("Embedding count does not match document count")

        _stage("Indexing documents in pgvector...", show_progress)
        retrieval_store: Any = store
        if registry is None:
            await store.upsert(summarized_documents, embeddings)
        else:
            retrieval_store = registry
        _write_json_artifact(run_dir, "documents_indexed.json", summarized_documents)
        if cache_policy.write_cache_telemetry:
            source_cache_report_path = write_cache_report(
                run_dir / "source_cache_report.json", cache_telemetry, cache_policy
            )
        _stage("[6/8] Retrieving evidence...", show_progress)
        retrieved_documents = await _retrieve_documents(
            store=retrieval_store,
            plan_data=plan_data,
            competition_id=resolved_competition_id,
            run_dir=run_dir,
            top_k=settings.top_k,
            warnings=warnings,
            show_progress=show_progress,
            fast=fast,
        )
        _write_json_artifact(run_dir, "retrieved_documents.json", retrieved_documents)
        if registry is not None:
            for document in retrieved_documents:
                await registry.record_run_source(
                    run_dir.name,
                    resolved_competition_id,
                    document.id,
                    version_id=document.metadata.get("version_id"),
                    selected_for_retrieval=True,
                    retrieval_score=document.score,
                    rrf_score=document.rrf_score,
                )
        scout_output_paths: dict[str, Path] = dict(provided_scout_paths)
        scout_num_hypotheses = 0
        scout_num_eda_tasks = 0
        if effective_write_eda_plan and mode != "scout" and (write_eda_plan or not scout_output_paths):
            _stage("[7/8] Running Research Scout...", show_progress)
            scout_output_paths, scout_num_hypotheses, scout_num_eda_tasks = (
                await _write_research_scout_outputs(
                    competition_id=resolved_competition_id,
                    competition_url=competition_url,
                    competition_desc=competition_desc,
                    plan_data=plan_data,
                    retrieved_documents=retrieved_documents,
                    client=client,
                    model=settings.deepseek_v4_pro,
                    run_dir=run_dir,
                )
            )
        if mode == "scout":
            _stage("[7/8] Running Research Scout...", show_progress)
            domain_patterns = await domain_memory.find_similar(
                task_type=plan_data.task_type,
                domain=plan_data.domain,
                top_k=5,
            )
            _write_json_artifact(run_dir, "domain_patterns.json", domain_patterns)
            scout_payload, scout_raw_payload = await build_research_hypotheses(
                competition_id=resolved_competition_id,
                competition_url=competition_url,
                competition_desc=competition_desc,
                plan_data=plan_data.model_dump(mode="json"),
                retrieved_documents=[
                    document.model_dump(mode="json") for document in retrieved_documents
                ],
                source_quality_summary=source_quality_summary(retrieved_documents),
                domain_patterns=domain_patterns,
                client=client,
                model=settings.deepseek_v4_pro,
                return_raw=True,
            )
            validation_payload = {"ok": True, "errors": []}
            try:
                validate_research_hypotheses(scout_payload)
            except ValueError as exc:
                validation_payload = {"ok": False, "errors": str(exc).splitlines()}
                _write_json_artifact(run_dir, "research_scout_validation.json", validation_payload)
                if allow_partial_scout_output:
                    partial_payload = {**scout_payload, "validation_errors": validation_payload["errors"]}
                    _write_json_artifact(run_dir, "research_hypotheses_partial.json", partial_payload)
                    warnings.append(f"Research Scout validation failed; wrote partial output: {exc}")
                else:
                    raise
            else:
                _write_json_artifact(run_dir, "research_scout_validation.json", validation_payload)

            # The legacy Scout result is a rich pre-publication envelope.  Project it
            # explicitly into the legacy hypothesis payload before canonical migration;
            # published schema 1.0 artifacts themselves remain strict/extra-forbid.
            research_boundary_payload = {
                key: scout_payload[key]
                for key in (
                    "competition_id", "created_at", "hypotheses", "eda_tasks",
                    "structured_findings", "scout_limitations", "models_used",
                )
                if key in scout_payload
            }
            research_boundary_payload["hypotheses"] = _adapt_legacy_scout_hypotheses(
                list(scout_payload.get("hypotheses") or [])
            )
            migration = migrate_research_hypotheses_payload(research_boundary_payload)
            canonical_hypotheses_payload = migration.canonical_payload
            scout_summary_override: str | None = None
            if migration.migrated:
                warnings.extend(migration.warnings)
                _write_json_artifact(
                    run_dir,
                    "research_hypotheses_migration.json",
                    {
                        "source_schema_version": migration.source_schema_version,
                        "target_schema_version": migration.target_schema_version,
                        "applied_migrations": migration.applied_migrations,
                        "warnings": migration.warnings,
                    },
                )
            eda_task_plan = split_eda_task_plan(scout_payload)
            adapted_tasks = _adapt_legacy_scout_tasks(
                list(eda_task_plan.get("eda_tasks") or [])
            )
            task_plan_boundary_payload = {
                key: eda_task_plan[key]
                for key in (
                    "competition_id", "task_type", "metric", "dataset", "eda_tasks",
                    "hypothesis_index", "recommended_module_sequence",
                    "recommended_human_checklist", "blocking_tasks",
                )
                if key in eda_task_plan
            }
            task_plan_boundary_payload["eda_tasks"] = adapted_tasks
            task_plan_boundary_payload["hypothesis_index"] = _derive_hypothesis_index(
                adapted_tasks
            )
            task_plan_boundary_payload["recommended_module_sequence"] = [
                _legacy_scout_module_name(value)
                for value in eda_task_plan.get("recommended_module_sequence") or []
            ]
            task_plan_boundary_payload["blocking_tasks"] = list(dict.fromkeys(
                task["module"] for task in adapted_tasks if task.get("blocking")
            ))
            task_plan_migration = migrate_eda_task_plan_payload(task_plan_boundary_payload)
            canonical_task_plan_payload = task_plan_migration.canonical_payload
            try:
                canonicalization = canonicalize_research_to_eda_contract(
                    canonical_hypotheses_payload,
                    canonical_task_plan_payload,
                )
                canonical_hypotheses = canonicalization.research_hypotheses
                canonical_task_plan = canonicalization.eda_task_plan
                validate_research_artifact_bundle(canonical_hypotheses, canonical_task_plan)
                require_valid_research_to_eda_contract(canonical_hypotheses, canonical_task_plan)
            except ValueError as exc:
                canonical_debug = {
                    "research_hypotheses": canonical_hypotheses_payload,
                    "eda_task_plan": canonical_task_plan_payload,
                }
                error_payload = (
                    exc.as_manifest_error()
                    if hasattr(exc, "as_manifest_error")
                    else {"error_type": type(exc).__name__, "message": str(exc)[:2000]}
                )
                _write_json_artifact(run_dir, "research_scout_raw_output.json", scout_raw_payload)
                _write_json_artifact(
                    run_dir, "research_scout_canonical_output.json", canonical_debug,
                )
                _write_json_artifact(
                    run_dir, "research_to_eda_validation_errors.json", error_payload,
                )
                fallback_output = build_deterministic_research_scout_fallback(
                    competition_id=resolved_competition_id,
                    competition_url=competition_url,
                    competition_desc=competition_desc,
                    plan_data=plan_data,
                    retrieved_documents=retrieved_documents,
                    model=settings.deepseek_v4_pro,
                    reason=str(exc)[:1000],
                )
                fallback_canonicalization = canonicalize_research_to_eda_contract(
                    fallback_output.to_research_hypotheses_payload(),
                    fallback_output.to_eda_task_plan_payload(),
                )
                canonical_hypotheses = fallback_canonicalization.research_hypotheses
                canonical_task_plan = fallback_canonicalization.eda_task_plan
                validate_research_artifact_bundle(canonical_hypotheses, canonical_task_plan)
                require_valid_research_to_eda_contract(canonical_hypotheses, canonical_task_plan)
                scout_summary_override = fallback_output.to_summary_markdown()
                warnings.append(
                    "Research Scout output failed semantic publication validation; "
                    "a deterministic generic fallback was published."
                )
            scout_summary = scout_summary_override or build_research_scout_summary(scout_payload)
            _write_json_artifact(run_dir, "research_scout_raw.json", scout_raw_payload)
            hypotheses_path = run_dir / "research_hypotheses.json"
            eda_task_plan_path = run_dir / "eda_task_plan.json"
            summary_path = run_dir / "research_scout_summary.md"
            write_research_hypotheses_atomic(hypotheses_path, canonical_hypotheses)
            write_eda_task_plan_atomic(eda_task_plan_path, canonical_task_plan)
            if task_plan_migration.migrated:
                _write_json_artifact(
                    run_dir,
                    "eda_task_plan_migration.json",
                    {
                        "source_schema_version": task_plan_migration.source_schema_version,
                        "target_schema_version": task_plan_migration.target_schema_version,
                        "applied_migrations": task_plan_migration.applied_migrations,
                        "warnings": task_plan_migration.warnings,
                    },
                )
            _write_json_artifact(
                run_dir,
                "research_scout_category_corrections.json",
                scout_payload.get("category_corrections", []),
            )
            summary_path.write_text(scout_summary, encoding="utf-8")
            _write_json_artifact(run_dir, "warnings.json", warnings)
            result = ResearchRunResult(
                competition_id=resolved_competition_id,
                mode="scout",
                report_mode="scout",
                num_documents=len(summarized_documents),
                num_sources=num_sources,
                warnings=warnings,
                duration_sec=round(time.perf_counter() - started_at, 3),
                run_artifacts_path=str(run_dir),
                retrieved_evidence_count=len(retrieved_documents),
                research_hypotheses_path=str(hypotheses_path),
                eda_task_plan_path=str(eda_task_plan_path),
                summary_path=str(summary_path),
                num_hypotheses=len(scout_payload["hypotheses"]),
                num_eda_tasks=len(scout_payload["eda_tasks"]),
                source_cache_report_path=str(source_cache_report_path) if source_cache_report_path else None,
                num_new_sources=cache_telemetry.sources_new,
                num_reused_sources=cache_telemetry.sources_reused,
                num_changed_sources=cache_telemetry.sources_changed,
                num_reused_embeddings=cache_telemetry.embeddings_reused,
                num_computed_embeddings=cache_telemetry.embeddings_computed,
            )
            run_summary = _build_research_run_summary(
                result=result,
                plan_data=plan_data,
                retrieved_documents=retrieved_documents,
                run_dir=run_dir,
            )
            _write_json_artifact(run_dir, "research_run.json", run_summary)
            _stage("Research Scout complete.", show_progress)
            _stage(f"Hypotheses saved to: {hypotheses_path}", show_progress)
            _stage(f"EDA task plan saved to: {eda_task_plan_path}", show_progress)
            _stage(f"Summary saved to: {summary_path}", show_progress)
            return result
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
            domain_memory=domain_memory,
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
        eda_evidence_pack_path: Path | None = provided_eda_evidence_pack_path
        eda_summary_path: Path | None = provided_eda_summary_path
        final_strategy_path: Path | None = None
        final_strategy_summary_path: Path | None = None
        final_synthesis_diagnostics_path: Path | None = None
        final_synthesis_status: str | None = None
        final_synthesis_degraded = False
        final_synthesis_stage_status: FinalSynthesisStageStatus | None = None
        workflow_status: WorkflowStatus = "success"
        degraded_stages: list[str] = []
        limitations: list[str] = []
        if execute_eda:
            _require_scout_paths_for_eda(scout_output_paths)
            if not scout_output_paths:
                raise RuntimeError("EDA requested but Research Scout outputs were not written.")
            _stage("Running EDA Engine...", show_progress)
            eda_result = await _run_optional_eda(
                competition_id=resolved_competition_id,
                competition_url=competition_url,
                scout_output_paths=scout_output_paths,
                local_dataset_path=local_dataset_path,
                eda_output_dir=eda_output_dir or run_dir / "eda_runs",
                download_dataset=download_dataset,
                force_download=force_download,
                enable_p1_modules=enable_p1_modules,
                enable_baseline=enable_baseline,
                enable_baseline_ablations=enable_baseline_ablations,
                enable_interaction_diagnostics=enable_interaction_diagnostics,
                enable_source_claim_validation=enable_source_claim_validation,
                enable_visual_diagnostics=enable_visual_diagnostics,
                enable_slice_diagnostics=enable_slice_diagnostics,
            )
            eda_evidence_pack_path = eda_result.evidence_pack_path
            eda_summary_path = eda_result.summary_path
            _stage(f"EDA evidence pack saved to: {eda_evidence_pack_path}", show_progress)
        if final_synthesis:
            if eda_evidence_pack_path is None:
                raise RuntimeError("Final synthesis requested but no EDA evidence pack is available.")
            if not eda_evidence_pack_path.is_file():
                raise FileNotFoundError(f"EDA evidence pack file does not exist: {eda_evidence_pack_path}")
            if eda_summary_path is not None and not eda_summary_path.is_file():
                raise FileNotFoundError(f"EDA summary file does not exist: {eda_summary_path}")
            research_hypotheses_for_final = _require_research_hypotheses_for_final(scout_output_paths)
            eda_pack_for_final = _load_json_model(
                eda_evidence_pack_path,
                EdaEvidencePack,
            )
            _stage("Running final strategy synthesis...", show_progress)
            final_hypotheses = _load_json_model(
                research_hypotheses_for_final, ResearchHypotheses
            )
            final_task_plan_path = Path(scout_output_paths["eda_task_plan"])
            final_task_plan = _load_json_model(final_task_plan_path, EdaTaskPlan)
            research_stage = ResearchStageResult(
                final_hypotheses,
                final_task_plan,
                research_hypotheses_for_final,
                final_task_plan_path,
                plan_data,
                tuple(retrieved_documents),
                tuple(_load_domain_patterns(run_dir)),
            )
            published_eda_bundle, publication_warnings = load_eda_publication_bundle(
                eda_evidence_pack_path.parent
            )
            eda_stage = EdaStageResult(
                published_eda_bundle.evidence_pack,
                eda_evidence_pack_path,
                eda_summary_path or eda_evidence_pack_path.with_suffix(".md"),
                evidence_manifest=published_eda_bundle.evidence_manifest,
                published_bundle=published_eda_bundle,
                publication_migration_warnings=publication_warnings,
            )
            reasoning_stage = _load_reasoning_stage_for_synthesis(
                run_dir, eda_pack_for_final
            )
            registries = build_contract_registries(
                research=research_stage, eda=eda_stage, reasoning=reasoning_stage
            )
            synthesis_context = build_final_synthesis_context(
                competition_desc=competition_desc,
                research=research_stage,
                published_eda_bundle=published_eda_bundle,
                reasoning=reasoning_stage,
                registries=registries,
                eda_summary_text=_load_optional_text(eda_summary_path),
            )
            final_strategy_output_dir = (
                Path(final_output_dir) if final_output_dir is not None else run_dir
            )
            final_strategy = await synthesize_final_strategy(
                context=synthesis_context,
                registries=registries,
                client=client,
                model=settings.deepseek_v4_pro,
                diagnostics_dir=final_strategy_output_dir,
            )
            final_strategy_path, final_strategy_summary_path = _write_final_strategy_outputs(
                final_strategy_output_dir,
                final_strategy,
                eda_evidence_pack=eda_pack_for_final.model_dump(mode="json"),
            )
            final_synthesis_diagnostics_path = (
                final_strategy_output_dir / "final_synthesis_diagnostics.json"
            )
            final_synthesis_status = final_strategy.synthesis_status
            final_synthesis_degraded = final_strategy.fallback_used
            if final_strategy.synthesis_status == "llm_success":
                final_synthesis_stage_status = "success"
            elif final_strategy.synthesis_status == "repaired_success":
                final_synthesis_stage_status = "repaired_success"
                warnings.append(
                    "Final synthesis required deterministic contract repair and "
                    "then passed validation."
                )
            else:
                degraded_stages.append("final_synthesis")
                warnings.append(
                    "Final synthesis completed with a deterministic degraded fallback; "
                    "this is not a successful LLM synthesis."
                )
                limitations.append(
                    "The final strategy was assembled deterministically because the "
                    "LLM output did not satisfy the final strategy contract."
                )
                if require_valid_final_synthesis:
                    workflow_status = "failed"
                    final_synthesis_stage_status = "failed"
                else:
                    workflow_status = "completed_with_degradation"
                    final_synthesis_stage_status = "degraded_fallback"
            _stage(f"Final strategy saved to: {final_strategy_path}", show_progress)
        result = ResearchRunResult(
            competition_id=resolved_competition_id,
            mode=mode,
            report_path=str(actual_report_path),
            num_documents=len(summarized_documents),
            num_sources=num_sources,
            warnings=warnings,
            limitations=limitations,
            workflow_status=workflow_status,
            degraded_stages=degraded_stages,
            duration_sec=round(time.perf_counter() - started_at, 3),
            report_mode=report_mode,
            run_artifacts_path=str(run_dir),
            retrieved_evidence_count=len(retrieved_documents),
            research_hypotheses_path=str(scout_output_paths["research_hypotheses"])
            if scout_output_paths
            else None,
            eda_task_plan_path=str(scout_output_paths["eda_task_plan"])
            if scout_output_paths
            else None,
            summary_path=str(scout_output_paths["research_scout_summary"])
            if scout_output_paths
            else None,
            num_hypotheses=scout_num_hypotheses,
            num_eda_tasks=scout_num_eda_tasks,
            eda_evidence_pack_path=str(eda_evidence_pack_path)
            if eda_evidence_pack_path
            else None,
            eda_summary_path=str(eda_summary_path) if eda_summary_path else None,
            final_strategy_path=str(final_strategy_path)
            if final_strategy_path
            else None,
            final_strategy_summary_path=str(final_strategy_summary_path)
            if final_strategy_summary_path
            else None,
            final_synthesis_diagnostics_path=str(final_synthesis_diagnostics_path)
            if final_synthesis_diagnostics_path
            else None,
            final_synthesis_status=final_synthesis_status,
            final_synthesis_degraded=final_synthesis_degraded,
            final_synthesis_stage_status=final_synthesis_stage_status,
            source_cache_report_path=str(source_cache_report_path) if source_cache_report_path else None,
            num_new_sources=cache_telemetry.sources_new,
            num_reused_sources=cache_telemetry.sources_reused,
            num_changed_sources=cache_telemetry.sources_changed,
            num_reused_embeddings=cache_telemetry.embeddings_reused,
            num_computed_embeddings=cache_telemetry.embeddings_computed,
        )
        run_summary = _build_research_run_summary(
            result=result,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            run_dir=run_dir,
        )
        _write_json_artifact(run_dir, "warnings.json", warnings)
        _write_json_artifact(run_dir, "research_run.json", run_summary)
        _write_json_file(actual_report_path.parent / "research_run.json", run_summary)
        if workflow_status == "failed" and final_synthesis_degraded:
            raise FinalSynthesisDegradedError(
                final_synthesis_diagnostics_path
                or run_dir / "final_synthesis_diagnostics.json",
                result=result,
            )
        return result
    except Exception:
        if cache_policy.write_cache_telemetry:
            write_cache_report(run_dir / "source_cache_report.json", cache_telemetry, cache_policy)
        _write_json_artifact(run_dir, "warnings.json", warnings)
        raise
    finally:
        if registry is not None:
            await registry.close()
        await store.close()
        await domain_memory.close()


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
    no_github: bool = False,
    fast: bool = False,
    registry: SourceRegistryStore | None = None,
    cache_policy: CachePolicy | None = None,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    try:
        documents.extend(
            await _collect_kaggle_sources_cached(
                _collect_kaggle_sources,
                plan_data=plan_data,
                competition_id=competition_id,
                settings=settings,
                warnings=warnings,
                show_progress=show_progress,
                fast=fast,
                registry=registry,
            )
        )
    except Exception as exc:
        warnings.append(f"Kaggle source collection failed: {exc}")
    _stage("[3/8] Collecting papers and repos...", show_progress)
    _stage("Processing arXiv PDFs...", show_progress)
    documents.extend(
        await _collect_arxiv_sources_cached(
            plan_data=plan_data,
            competition_id=competition_id,
            settings=settings,
            warnings=warnings,
            show_progress=show_progress,
            fast=fast,
            registry=registry,
        )
    )
    documents.extend(
        await _collect_paper_sources(
            plan_data=plan_data,
            competition_id=competition_id,
            settings=settings,
            warnings=warnings,
            fast=fast,
        )
    )
    if no_github:
        warnings.append("GitHub source collection skipped by --no-github.")
    else:
        documents.extend(
            await _collect_github_sources(
                plan_data=plan_data,
                competition_id=competition_id,
                settings=settings,
                warnings=warnings,
                show_progress=show_progress,
                fast=fast,
                registry=registry,
            )
        )
    return documents


async def _collect_kaggle_sources_cached(
    collector: Any,
    *,
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
    fast: bool,
    registry: SourceRegistryStore | None,
) -> list[SourceDocument]:
    if registry is None:
        return await asyncio.to_thread(
            collector, plan_data, competition_id, settings, warnings, show_progress, fast
        )
    raw_notebooks = await asyncio.to_thread(
        search_notebooks,
        plan_data.kaggle_queries,
        competition_id,
        _fast_limit(settings.max_notebooks, FAST_MAX_NOTEBOOKS, fast),
    )
    usable: list[dict[str, Any]] = []
    for notebook in raw_notebooks:
        notebook_id = _notebook_id(notebook)
        if not notebook_id:
            continue
        metadata = dict(notebook.get("metadata") or {})
        revision = metadata.get("source_revision")
        cached_content = await _cached_content_for_revision(
            registry, "kaggle", notebook_id, notebook.get("url"), revision,
            bool(metadata.get("revision_is_reliable")),
            expected_parser_fingerprint=kaggle_notebook_parser_fingerprint(),
        )
        item = dict(notebook)
        if cached_content is not None:
            item["content"] = cached_content
            item["metadata"] = {**metadata, "content_from_registry_cache": True}
        else:
            try:
                item["content"] = await asyncio.to_thread(get_notebook_content, notebook_id)
            except Exception as exc:
                warnings.append(f"Kaggle notebook pull failed for {notebook_id}: {exc}")
                continue
        if str(item.get("content") or "").strip():
            usable.append(item)
    if not usable:
        raise RuntimeError("Kaggle retrieval returned 0 usable notebooks")
    return build_kaggle_documents(usable, competition_id=competition_id)


async def _collect_arxiv_sources_cached(
    *,
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
    fast: bool,
    registry: SourceRegistryStore | None,
) -> list[SourceDocument]:
    try:
        papers = await asyncio.to_thread(
            search_arxiv,
            plan_data.arxiv_queries,
            _fast_limit(settings.max_papers, FAST_MAX_PAPERS, fast),
        )
        if registry is None:
            enriched = await asyncio.to_thread(enrich_with_pdf, papers, settings.pdf_cache_dir)
            return build_arxiv_documents(enriched, competition_id=competition_id)
        reusable: list[dict[str, Any]] = []
        needs_content: list[dict[str, Any]] = []
        for paper in papers:
            metadata = dict(paper.get("metadata") or {})
            revision = paper.get("source_revision") or metadata.get("source_revision")
            content = await _cached_content_for_revision(
                registry, "arxiv", paper.get("entry_id"), paper.get("url"), revision,
                bool(paper.get("revision_is_reliable", metadata.get("revision_is_reliable", False))),
                expected_parser_fingerprint=arxiv_pdf_parser_fingerprint(),
            )
            if content is None:
                needs_content.append(paper)
            else:
                identity = canonicalize_source_identity(
                    "arxiv",
                    str(paper.get("entry_id") or "") or None,
                    str(paper.get("url") or "") or None,
                )
                cached_source = await registry.get_source(identity.source_id)
                cached_metadata = cached_source.metadata if cached_source is not None else {}
                reusable.append({
                    **paper,
                    "content": content,
                    "metadata": {
                        **cached_metadata,
                        **metadata,
                        "content_from_registry_cache": True,
                    },
                })
        if needs_content:
            reusable.extend(await asyncio.to_thread(enrich_with_pdf, needs_content, settings.pdf_cache_dir))
        return build_arxiv_documents(reusable, competition_id=competition_id)
    except Exception as exc:
        warnings.append(f"arXiv source collection failed: {exc}")
        return []


async def _cached_content_for_revision(
    registry: SourceRegistryStore,
    source_type: str,
    external_id: Any,
    url: Any,
    revision: Any,
    revision_is_reliable: bool,
    expected_parser_fingerprint: str | None = None,
) -> str | None:
    if not revision_is_reliable or revision is None:
        return None
    identity = canonicalize_source_identity(
        source_type,
        str(external_id) if external_id else None,
        str(url) if url else None,
    )
    current = await registry.get_current_version(identity.source_id)
    if current is None or current.source_revision != str(revision):
        return None
    if current.content_location and not Path(current.content_location).is_file():
        return None
    parsed = await registry.find_latest_artifact(current.version_id, "parsed_text")
    if expected_parser_fingerprint and (
        parsed is None or parsed.processor_fingerprint != expected_parser_fingerprint
    ):
        return None
    if current.raw_content:
        return current.raw_content
    if parsed and isinstance(parsed.payload, dict):
        return str(parsed.payload.get("text") or "") or None
    return None


async def _load_cached_source_documents(
    registry: SourceRegistryStore,
    competition_id: str,
    *,
    source_ids: list[str] | None = None,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    selected_ids = source_ids if source_ids is not None else await registry.list_competition_source_ids(competition_id)
    for source_id in selected_ids:
        source = await registry.get_source(source_id)
        version = await registry.get_current_version(source_id)
        if source is None or version is None:
            continue
        if version.content_location and not Path(version.content_location).is_file():
            continue
        parsed = await registry.find_latest_artifact(version.version_id, "parsed_text")
        payload = parsed.payload if parsed is not None else None
        if isinstance(payload, dict):
            content = str(payload.get("text") or payload.get("summary") or "")
        elif isinstance(payload, str):
            content = payload
        else:
            content = version.raw_content or ""
        if not content:
            continue
        current_metadata = dict(source.metadata)
        if source.source_type == "arxiv" and current_metadata.get("raw_pdf_hash"):
            current_metadata["parser_fingerprint"] = arxiv_pdf_parser_fingerprint()
        elif source.source_type == "kaggle":
            current_metadata["parser_fingerprint"] = kaggle_notebook_parser_fingerprint()
        elif source.source_type == "github":
            current_metadata["parser_fingerprint"] = github_readme_parser_fingerprint()
        documents.append(
            SourceDocument(
                id=source.source_id,
                competition_id=competition_id,
                source=source.source_type,
                title=source.title or source.source_id,
                url=source.canonical_url,
                content=content,
                metadata={
                    **current_metadata,
                    "source_id": source.source_id,
                    "source_revision": version.source_revision,
                    "revision_is_reliable": bool(version.metadata.get("revision_is_reliable")),
                    "version_id": str(version.version_id),
                    "content_from_registry_cache": True,
                },
            )
        )
    if not documents:
        from kaggle_researcher.source_registry.errors import SourceOfflineCacheMissError

        raise SourceOfflineCacheMissError(
            f"No cached sources are associated with competition {competition_id!r}"
        )
    return documents


def _source_discovery_requests(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    *,
    no_github: bool,
    fast: bool,
) -> list[dict[str, Any]]:
    paper_query = " | ".join(plan_data.arxiv_queries)
    requests = [
        {
            "provider": "kaggle",
            "query": f"competition:{competition_id} | {' | '.join(plan_data.kaggle_queries)}",
            "limit": _fast_limit(settings.max_notebooks, FAST_MAX_NOTEBOOKS, fast),
            "sort_mode": "voteCount",
        }
    ]
    if paper_query:
        requests.extend([
            {"provider": "arxiv", "query": paper_query, "limit": _fast_limit(settings.max_papers, FAST_MAX_PAPERS, fast), "sort_mode": "relevance"},
            {"provider": "papers_with_code", "query": paper_query, "limit": _fast_limit(settings.max_papers, FAST_MAX_PAPERS, fast), "sort_mode": "provider_default"},
        ])
    if not no_github and plan_data.github_queries:
        requests.append({
            "provider": "github", "query": " | ".join(plan_data.github_queries),
            "limit": _fast_limit(settings.max_repos, FAST_MAX_REPOS, fast), "sort_mode": "stars",
        })
    return requests


async def _cached_discovery_source_ids(
    registry: SourceRegistryStore,
    requests: list[dict[str, Any]],
    policy: CachePolicy,
    telemetry: CacheRunTelemetry,
) -> list[str] | None:
    if policy.source_refresh_mode == SourceRefreshMode.ALWAYS:
        telemetry.search_cache_misses += len(requests)
        telemetry.provider_calls += len(requests)
        return None
    now = datetime.now(timezone.utc)
    entries: list[SearchCacheEntry] = []
    for request in requests:
        normalized = normalize_search_query(request["query"])
        fingerprint = build_search_request_fingerprint(
            provider=request["provider"], normalized_query=normalized,
            result_limit=request["limit"], sort_mode=request["sort_mode"],
        ).fingerprint
        entry = await registry.get_search_cache(request["provider"], sha256_text(normalized), fingerprint)
        fresh = entry is not None and entry.expires_at > now
        if entry is None or (not fresh and policy.source_refresh_mode == SourceRefreshMode.AUTO):
            if policy.source_refresh_mode == SourceRefreshMode.NEVER:
                from kaggle_researcher.source_registry.errors import SourceOfflineCacheMissError

                raise SourceOfflineCacheMissError(
                    f"No cached {request['provider']} discovery result is available in offline mode"
                )
            telemetry.search_cache_misses += 1
            telemetry.provider_calls += len(requests)
            return None
        if not fresh:
            if not policy.allow_stale_search_cache_when_offline:
                from kaggle_researcher.source_registry.errors import SourceOfflineCacheMissError

                raise SourceOfflineCacheMissError(
                    f"Cached {request['provider']} discovery result is stale in offline mode"
                )
            telemetry.search_stale_hits += 1
            telemetry.warnings.append(
                f"Used stale {request['provider']} discovery cache while source refresh was disabled."
            )
        else:
            telemetry.search_cache_hits += 1
        entries.append(entry)
    return list(dict.fromkeys(source_id for entry in entries for source_id in entry.result_source_ids))


async def _save_discovery_cache(
    registry: SourceRegistryStore,
    requests: list[dict[str, Any]],
    documents: list[SourceDocument],
    policy: CachePolicy,
) -> None:
    now = datetime.now(timezone.utc)
    for request in requests:
        provider = request["provider"]
        if provider in {"arxiv", "papers_with_code"}:
            accepted_sources = {"arxiv", "papers_with_code", "papers_with_code_legacy", "huggingface_papers"}
        else:
            accepted_sources = {provider}
        source_ids = [document.id for document in documents if document.source in accepted_sources]
        normalized = normalize_search_query(request["query"])
        fingerprint = build_search_request_fingerprint(
            provider=provider, normalized_query=normalized, result_limit=request["limit"],
            sort_mode=request["sort_mode"],
        ).fingerprint
        ttl = policy.search_ttl_by_provider.get(provider)
        if ttl is None:
            continue
        await registry.save_search_cache(SearchCacheEntry(
            provider=provider, query_hash=sha256_text(normalized), normalized_query=normalized,
            request_fingerprint=fingerprint, result_source_ids=list(dict.fromkeys(source_ids)),
            raw_result_metadata={"result_count": len(source_ids)}, fetched_at=now, expires_at=now + ttl,
        ))


def _build_source_cache_policy(
    settings: object,
    *,
    source_refresh: str | None,
    rebuild_source_artifacts: set[str] | list[str] | tuple[str, ...] | str | None,
    no_source_cache: bool,
    source_cache_report: bool,
) -> CachePolicy:
    if isinstance(rebuild_source_artifacts, str):
        rebuild = {item.strip() for item in rebuild_source_artifacts.split(",") if item.strip()}
    else:
        rebuild = set(rebuild_source_artifacts or ())
    if hasattr(settings, "source_search_ttls"):
        ttls = settings.source_search_ttls()
    else:
        from datetime import timedelta

        ttls = {
            "kaggle": timedelta(hours=int(getattr(settings, "source_search_ttl_kaggle_hours", 24))),
            "github": timedelta(hours=int(getattr(settings, "source_search_ttl_github_hours", 24))),
            "arxiv": timedelta(hours=int(getattr(settings, "source_search_ttl_arxiv_hours", 168))),
            "papers_with_code": timedelta(
                hours=int(getattr(settings, "source_search_ttl_papers_with_code_hours", 168))
            ),
        }
    return CachePolicy(
        source_refresh_mode=SourceRefreshMode(
            source_refresh or str(getattr(settings, "source_refresh_mode", "auto"))
        ),
        rebuild_artifacts=rebuild,
        search_ttl_by_provider=ttls,
        allow_stale_search_cache_when_offline=bool(
            getattr(settings, "source_cache_allow_stale_offline", True)
        ),
        write_cache_telemetry=source_cache_report,
        cache_enabled=not no_source_cache,
    )


def _collect_kaggle_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
    fast: bool = False,
) -> list[SourceDocument]:
    raw_notebooks = search_notebooks(
        plan_data.kaggle_queries,
        competition_id=competition_id,
        max_notebooks=_fast_limit(settings.max_notebooks, FAST_MAX_NOTEBOOKS, fast),
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
    fast: bool = False,
) -> list[SourceDocument]:
    try:
        papers = search_arxiv(
            plan_data.arxiv_queries,
            max_papers=_fast_limit(settings.max_papers, FAST_MAX_PAPERS, fast),
        )
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
    fast: bool = False,
) -> list[SourceDocument]:
    paper_queries = [*plan_data.arxiv_queries]
    if not paper_queries:
        return []
    raw_papers = await search_paper_sources(
        queries=paper_queries,
        max_results=_fast_limit(settings.max_papers, FAST_MAX_PAPERS, fast),
        warnings=warnings,
    )
    return build_arxiv_documents(raw_papers, competition_id=competition_id)


async def _collect_github_sources(
    plan_data: PlanData,
    competition_id: str,
    settings: object,
    warnings: list[str],
    show_progress: bool,
    fast: bool = False,
    registry: SourceRegistryStore | None = None,
) -> list[SourceDocument]:
    if not plan_data.github_queries:
        return []
    _stage("Collecting GitHub repos...", show_progress)
    try:
        async def cached_github_content(full_name: str, commit_sha: str) -> str | None:
            if registry is None:
                return None
            return await _cached_content_for_revision(
                registry, "github", full_name, f"https://github.com/{full_name}",
                commit_sha, True,
                expected_parser_fingerprint=github_readme_parser_fingerprint(),
            )

        raw_repos = await search_repos(
            plan_data.github_queries,
            token=settings.github_token,
            max_repos=_fast_limit(settings.max_repos, FAST_MAX_REPOS, fast),
            content_cache_lookup=cached_github_content if registry is not None else None,
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
    fast: bool = False,
) -> list[RetrievedDocument]:
    retrieved_by_id: dict[str, RetrievedDocument] = {}
    queries = _retrieval_queries(plan_data)
    if fast:
        queries = queries[:FAST_MAX_RETRIEVAL_QUERIES]
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
    domain_memory: DomainMemory,
    client: DeepSeekClient,
    model: str,
    run_dir: Path,
    warnings: list[str],
    show_progress: bool,
    fast: bool = False,
) -> str:
    if report_mode == "minimal":
        return _build_minimal_report_text(competition_desc, plan_data, retrieved_documents)

    try:
        return await _build_full_report_text(
            competition_desc=competition_desc,
            plan_data=plan_data,
            retrieved_documents=retrieved_documents,
            domain_memory=domain_memory,
            client=client,
            model=model,
            run_dir=run_dir,
            show_progress=show_progress,
            skip_reviewer=fast,
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
    domain_memory: DomainMemory,
    client: DeepSeekClient,
    model: str,
    run_dir: Path,
    show_progress: bool,
    skip_reviewer: bool = False,
) -> str:
    domain_patterns = await _run_reasoning_stage(
        "domain_memory",
        lambda: domain_memory.find_similar(
            task_type=plan_data.task_type,
            domain=plan_data.domain,
            top_k=5,
        ),
        show_progress,
    )
    _write_json_artifact(run_dir, "domain_patterns.json", domain_patterns)
    provenance_sections: dict[str, Any] = {}

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

    draft_sections = {
        "validation": provenance_sections["validation"],
        "leakage": provenance_sections["leakage"],
        "metric": provenance_sections["metric"],
        "experiments": provenance_sections["experiments"],
        "leaderboard": provenance_sections["leaderboard"],
    }
    if skip_reviewer:
        review_result = ReviewResult(
            confidence="low",
            evidence_ids=[],
            too_generic=["Skipped by --fast."],
        )
    else:
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
    _write_json_file(run_dir / filename, value)


def _write_json_file(path: Path, value: Any) -> None:
    write_json_atomic(path, _jsonable(value))


def _write_text_artifact(run_dir: Path, filename: str, value: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / filename).write_text(value, encoding="utf-8")


def _provided_scout_paths(
    *,
    research_hypotheses_path: str | Path | None,
    eda_task_plan_path: str | Path | None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if research_hypotheses_path is not None:
        paths["research_hypotheses"] = Path(research_hypotheses_path)
    if eda_task_plan_path is not None:
        paths["eda_task_plan"] = Path(eda_task_plan_path)
    return paths


def _require_scout_paths_for_eda(paths: dict[str, Path]) -> None:
    missing = [
        label
        for label in ("research_hypotheses", "eda_task_plan")
        if label not in paths
    ]
    if missing:
        raise RuntimeError(
            "EDA requested but required Scout output path(s) are missing: "
            + ", ".join(missing)
            + ". Use --write-eda-plan or provide --research-hypotheses-path and --eda-task-plan-path."
        )
    for label in ("research_hypotheses", "eda_task_plan"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"Scout output file does not exist: {paths[label]}")


def _require_research_hypotheses_for_final(paths: dict[str, Path]) -> Path:
    path = paths.get("research_hypotheses")
    if path is None:
        raise RuntimeError(
            "Final synthesis requested but no research_hypotheses.json is available. "
            "Use --write-eda-plan or provide --research-hypotheses-path."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Research hypotheses file does not exist: {path}")
    return path


async def _write_research_scout_outputs(
    *,
    competition_id: str,
    competition_url: str,
    competition_desc: str,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    client: DeepSeekClient,
    model: str,
    run_dir: Path,
) -> tuple[dict[str, Path], int, int]:
    scout_output = await run_research_scout(
        competition_id=competition_id,
        competition_url=competition_url,
        competition_desc=competition_desc,
        plan_data=plan_data,
        retrieved_documents=retrieved_documents,
        client=client,
        model=model,
    )
    paths = scout_output.write_outputs(run_dir)
    return paths, len(scout_output.hypotheses), len(scout_output.eda_task_plan.eda_tasks)


async def _run_optional_eda(
    *,
    competition_id: str,
    competition_url: str,
    scout_output_paths: dict[str, Path],
    local_dataset_path: str | Path | None,
    eda_output_dir: str | Path,
    download_dataset: bool,
    force_download: bool,
    enable_p1_modules: bool,
    enable_baseline: bool,
    enable_baseline_ablations: bool,
    enable_interaction_diagnostics: bool,
    enable_source_claim_validation: bool,
    enable_visual_diagnostics: bool,
    enable_slice_diagnostics: bool,
):
    return await getattr(eda_orchestrator, "run" "_eda")(
        EdaRunConfig(
            competition_id=competition_id,
            competition_url=competition_url,
            hypotheses_path=scout_output_paths["research_hypotheses"],
            task_plan_path=scout_output_paths["eda_task_plan"],
            local_dataset_path=Path(local_dataset_path)
            if local_dataset_path is not None
            else None,
            output_dir=Path(eda_output_dir),
            download_dataset=download_dataset,
            force_download=force_download,
            enable_p1_modules=enable_p1_modules,
            enable_baseline=enable_baseline,
            enable_baseline_ablations=enable_baseline_ablations,
            enable_interaction_diagnostics=enable_interaction_diagnostics,
            enable_source_claim_validation=enable_source_claim_validation,
            enable_visual_diagnostics=enable_visual_diagnostics,
            enable_slice_diagnostics=enable_slice_diagnostics,
        )
    )


def _load_json_model(path: Path, model_type: Any) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return model_type.model_validate(payload)


def _load_optional_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def _load_domain_patterns(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "domain_patterns.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _write_final_strategy_outputs(
    output_dir: Path,
    final_strategy: FinalStrategyResult,
    *,
    eda_evidence_pack: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_strategy.json"
    markdown_path = output_dir / "final_strategy.md"
    summary_path = output_dir / "final_strategy_summary.md"
    markdown = render_final_strategy(final_strategy)
    summary = render_final_strategy_summary(final_strategy)
    for warning in validate_rendered_strategy_quality(
        final_strategy,
        markdown,
        summary,
        eda_evidence_pack=eda_evidence_pack,
    ):
        logger.warning("Final strategy quality check: %s", warning)
    _write_json_file(json_path, final_strategy)
    markdown_path.write_text(markdown, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    return json_path, summary_path


def _final_strategy_markdown(final_strategy: FinalStrategyResult) -> str:
    return render_final_strategy(final_strategy)


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
        "research_scout": reasoning_model,
        "validation_architect": reasoning_model,
        "leakage_risk_analyst": reasoning_model,
        "metric_specialist": reasoning_model,
        "experiment_planner": reasoning_model,
        "leaderboard_auditor": reasoning_model,
        "skeptical_reviewer": reasoning_model,
        "report_composer": reasoning_model,
        "embedder": settings.embed_model,
    }


def _add_missing_source_warnings(
    num_sources: dict[str, int],
    warnings: list[str],
    *,
    no_github: bool = False,
) -> None:
    if not no_github and num_sources.get("github", 0) == 0:
        warnings.append("GitHub source count is 0. Check GITHUB_TOKEN or query quality.")
    if num_sources.get("huggingface_papers", 0) == 0:
        warnings.append(
            "Hugging Face Papers source count is 0. Falling back to arXiv-only academic retrieval."
        )


def _build_research_run_summary(
    result: ResearchRunResult,
    plan_data: PlanData,
    retrieved_documents: list[RetrievedDocument],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        **result.model_dump(mode="json"),
        "plan_data": plan_data.model_dump(mode="json"),
        "retrieved_document_ids": [document.id for document in retrieved_documents],
        "reasoning_outputs_summary": _reasoning_outputs_summary(run_dir),
        "report_path": result.report_path,
    }


def _reasoning_outputs_summary(run_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    artifact_map = {
        "metric": "metric_result.json",
        "validation": "validation_result.json",
        "leakage": "leakage_result.json",
        "leaderboard": "leaderboard_audit.json",
        "experiments": "experiments.json",
        "review": "review_result.json",
    }
    for name, filename in artifact_map.items():
        path = run_dir / filename
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        summary[name] = _summarize_reasoning_output(value)
    return summary


def _load_reasoning_stage_for_synthesis(
    run_dir: Path, eda_pack: EdaEvidencePack
) -> ReasoningStageResult:
    """Load typed reasoning artifacts, with bounded in-memory standalone defaults.

    The canonical full-run path always has these artifacts. The defaults keep the
    older standalone `--final-synthesis` surface usable without persisting a
    second artifact family or passing an untyped summary map.
    """

    primary = (eda_pack.validation_evidence.get("primary_validation") or {}).get("method") or "custom_required"

    def load_or(path: Path, model: type[BaseModel], default: BaseModel):
        return _load_json_model(path, model) if path.is_file() else default

    metric = load_or(
        run_dir / "metric_result.json",
        MetricResult,
        MetricResult(
            confidence="low",
            metric_explanation="Metric reasoning was not run in standalone synthesis mode.",
            needs_calibration=False,
            rank_averaging_useful=False,
            threshold_search_needed=False,
            surrogate_loss_suggestion="Use the competition metric consistently.",
        ),
    )
    validation = load_or(
        run_dir / "validation_result.json",
        ValidationResult,
        ValidationResult(
            confidence="low",
            recommended_cv=str(primary),
            validation_risk="medium",
            likely_split="unknown",
            reasoning="Use the EDA-selected primary validation policy.",
            primary_validation={"method": str(primary)},
        ),
    )
    leakage = load_or(
        run_dir / "leakage_result.json",
        LeakageRiskResult,
        LeakageRiskResult(
            confidence="low", risk_level="medium", possible_issues=[], recommended_checks=[]
        ),
    )
    leaderboard = load_or(
        run_dir / "leaderboard_audit.json",
        LeaderboardAuditResult,
        LeaderboardAuditResult(
            confidence="low",
            shake_up_risk="medium",
            submission_selection_rule="Use the canonical validation policy.",
            public_lb_trust="low",
            warnings=[],
        ),
    )
    experiment_path = run_dir / "experiments.json"
    review_path = run_dir / "review_result.json"
    return ReasoningStageResult(
        metric=metric,
        validation=validation,
        leakage=leakage,
        leaderboard=leaderboard,
        experiments=load_experiment_plan(experiment_path) if experiment_path.is_file() else ExperimentPlan(),
        review=load_skeptical_review(review_path) if review_path.is_file() else None,
    )


def _summarize_reasoning_output(value: Any) -> Any:
    if isinstance(value, list):
        return {
            "count": len(value),
            "items": [
                {
                    key: item.get(key)
                    for key in ("priority", "experiment", "confidence", "evidence_ids")
                    if isinstance(item, dict) and key in item
                }
                for item in value[:10]
            ],
        }
    if isinstance(value, dict):
        keys = (
            "confidence",
            "evidence_ids",
            "risk_level",
            "validation_risk",
            "recommended_cv",
            "shake_up_risk",
            "public_lb_trust",
            "unsupported_claims",
            "too_generic",
            "unnecessary_experiments",
        )
        return {key: value[key] for key in keys if key in value}
    return value


def _fast_limit(configured: int, fast_limit: int, fast: bool) -> int:
    return min(configured, fast_limit) if fast else configured


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown-competition"


async def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "validate-contracts":
        from kaggle_researcher.contracts.artifacts import validate_contract_definitions

        print(json.dumps(validate_contract_definitions(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if argv and argv[0] in {"full-run", "run-all"}:
        from kaggle_researcher.orchestration.full_run import FullRunConfig, run_full_research

        args = build_full_run_parser().parse_args(argv[1:])
        config = FullRunConfig(
            competition_id=args.competition_id,
            competition_url=args.competition_url,
            competition_description=args.competition_description,
            local_dataset_path=args.local_dataset_path,
            download_dataset=args.download_dataset,
            output_root=args.output_root,
            profile=args.profile,
            enable_p1_modules=args.enable_p1_modules,
            enable_baseline=args.enable_baseline,
            enable_baseline_ablations=args.enable_baseline_ablations,
            enable_interaction_diagnostics=args.enable_interaction_diagnostics,
            enable_slice_diagnostics=args.enable_slice_diagnostics,
            enable_source_claim_validation=args.enable_source_claim_validation,
            enable_visual_diagnostics=args.enable_visual_diagnostics,
            fail_fast=args.fail_fast,
            require_valid_final_synthesis=args.require_valid_final_synthesis,
            resume_run_dir=args.resume_run_dir,
            force_rerun_stages=set(args.force_rerun_stage),
            disable_progress=args.no_progress,
        )
        try:
            result = await run_full_research(config)
        except FinalSynthesisDegradedError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Full run workflow status: {result.workflow_status}")
        if result.workflow_status == "completed_with_degradation":
            print(
                "WARNING: Final synthesis used a deterministic degraded fallback; "
                "artifacts were preserved."
            )
        print(f"Run directory: {result.run_dir}")
        print(f"Final strategy: {result.final_strategy_path}")
        print(f"Final report: {result.final_report_path}")
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source_registry_migrate:
        from kaggle_researcher.source_registry.migrations import _run_cli as run_migration_cli

        await run_migration_cli(["migrate-legacy-documents"])
        if not args.competition_url or not args.competition_desc:
            return 0
    if not args.competition_url or not args.competition_desc:
        print("Provide competition_url and competition_desc to run the minimal pipeline.")
        return 0

    mode = args.mode
    if mode == "full" and args.report_mode == "minimal":
        mode = "minimal"

    try:
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
            mode=mode,
            allow_minimal_fallback=args.allow_minimal_fallback,
            allow_partial_scout_output=args.allow_partial_scout_output,
            write_eda_plan=args.write_eda_plan,
            execute_eda=args.execute_eda,
            local_dataset_path=args.local_dataset_path,
            eda_output_dir=args.eda_output_dir,
            download_dataset=args.download_dataset,
            force_download=args.force_download,
            enable_p1_modules=args.enable_p1_modules,
            enable_baseline=args.enable_baseline,
            enable_baseline_ablations=args.enable_baseline_ablations,
            enable_interaction_diagnostics=args.enable_interaction_diagnostics,
            enable_source_claim_validation=args.enable_source_claim_validation,
            enable_visual_diagnostics=args.enable_visual_diagnostics,
            enable_slice_diagnostics=args.enable_slice_diagnostics,
            research_hypotheses_path=args.research_hypotheses_path,
            eda_task_plan_path=args.eda_task_plan_path,
            eda_evidence_pack_path=args.eda_evidence_pack_path,
            eda_summary_path=args.eda_summary_path,
            final_synthesis=args.final_synthesis,
            final_output_dir=args.final_output_dir,
            require_valid_final_synthesis=args.require_valid_final_synthesis,
            debug=args.debug,
            no_github=args.no_github,
            fast=args.fast,
            source_refresh=args.source_refresh,
            rebuild_source_artifacts=args.rebuild_source_artifacts,
            no_source_cache=args.no_source_cache,
            source_cache_report=args.source_cache_report,
        )
    except FinalSynthesisDegradedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if result.mode == "scout":
        print("Research Scout complete.")
        print(f"Hypotheses saved to: {result.research_hypotheses_path}")
        print(f"EDA task plan saved to: {result.eda_task_plan_path}")
        print(f"Summary saved to: {result.summary_path}")
        print(f"Run artifacts saved to: {result.run_artifacts_path}")
        print(f"Hypotheses: {result.num_hypotheses}")
        print(f"EDA tasks: {result.num_eda_tasks}")
        print(f"Warnings: {len(result.warnings)}")
        for warning in result.warnings:
            print(f"  - {warning}")
        print(result.model_dump_json(indent=2))
        return 0

    if result.workflow_status == "completed_with_degradation":
        print("Research run completed with degradation.")
        print(
            "WARNING: Final synthesis used a deterministic degraded fallback; "
            "artifacts were preserved."
        )
    else:
        print("Research run complete.")
    print(f"Workflow status: {result.workflow_status}")
    print(f"Report mode: {result.report_mode}")
    print(f"Report saved to: {result.report_path}")
    print(f"Run artifacts saved to: {result.run_artifacts_path}")
    if result.research_hypotheses_path:
        print(f"Research hypotheses saved to: {result.research_hypotheses_path}")
    if result.eda_task_plan_path:
        print(f"EDA task plan saved to: {result.eda_task_plan_path}")
    if result.eda_evidence_pack_path:
        print(f"EDA evidence pack saved to: {result.eda_evidence_pack_path}")
    if result.final_strategy_path:
        print(f"Final strategy saved to: {result.final_strategy_path}")
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


_LEGACY_SCOUT_MODULE_ALIASES = {
    "notebook_reverse_engineering": "notebook_static_analysis",
}

_DEFAULT_CONTRACT_CHECK_BY_CATEGORY = {
    "dataset_schema": "schema_inferer.roles",
    "schema": "schema_inferer.roles",
    "metric": "metric_analyzer.resolve_metric",
    "validation": "validation_analyzer.select_primary_validation",
    "leakage": "leakage_checker.basic",
    "relationships": "relationship_inferer.relationships",
    "relationship": "relationship_inferer.relationships",
    "drift": "drift_analyzer.generic",
    "feature_engineering": "feature_probe.feature_family_probe",
    "feature": "feature_probe.feature_family_probe",
    "baseline": "baseline_runner.honest_baseline",
    "notebook_reverse_engineering": "notebook_static_analysis.static_patterns",
    "notebook": "notebook_static_analysis.static_patterns",
    "leaderboard_risk": "drift_analyzer.train_test_shift",
}


def _legacy_scout_module_name(value: Any) -> str:
    name = str(value)
    return _LEGACY_SCOUT_MODULE_ALIASES.get(name, name)


def _adapt_legacy_scout_hypotheses(items: list[Any]) -> list[Any]:
    """Adapt the rich legacy Scout envelope before publishing strict artifacts."""

    result: list[Any] = []
    for raw in items:
        if not isinstance(raw, dict):
            result.append(raw)
            continue
        item = dict(raw)
        category = str(item.get("category") or "")
        checks: list[str] = []
        for step in item.get("verification_steps") or []:
            if not isinstance(step, dict):
                continue
            module = _legacy_scout_module_name(step.get("module"))
            operation = str(step.get("operation") or "").lower()
            if module == "file_inventory":
                check = "file_inventory.roles"
            elif module == "schema_inferer":
                check = (
                    "schema_inferer.detect_time_columns"
                    if "time" in operation or "date" in operation
                    else "schema_inferer.roles"
                )
            elif module == "table_profiler":
                check = "table_profiler.profile_tables"
            elif module == "metric_analyzer":
                check = "metric_analyzer.resolve_metric"
            elif module == "validation_analyzer":
                if any(token in operation for token in ("time", "oot", "rolling", "period")):
                    check = "validation_analyzer.temporal_cv_feasibility"
                elif "rank" in operation or "query" in operation:
                    check = "validation_analyzer.ranking_validation"
                elif "group" in operation:
                    check = "validation_analyzer.group_cv_feasibility"
                else:
                    check = "validation_analyzer.select_primary_validation"
            elif module == "leakage_checker":
                if "query" in operation or "rank" in operation:
                    check = "leakage_checker.ranking_query_overlap"
                elif "group" in operation:
                    check = "leakage_checker.group_overlap"
                elif "id" in operation or "overlap" in operation:
                    check = "leakage_checker.train_test_id_overlap"
                else:
                    check = "leakage_checker.target_proxy_scan"
            elif module == "relationship_inferer":
                check = "relationship_inferer.relationships"
            elif module == "drift_analyzer":
                check = "drift_analyzer.generic"
            elif module == "baseline_runner":
                check = "baseline_runner.honest_baseline"
            elif module == "feature_probe":
                check = "feature_probe.feature_family_probe"
            elif module == "notebook_static_analysis":
                check = "notebook_static_analysis.static_patterns"
            else:
                continue
            if check not in checks:
                checks.append(check)
        if not checks:
            checks = [_DEFAULT_CONTRACT_CHECK_BY_CATEGORY.get(
                category, "table_profiler.profile_tables"
            )]
        item["expected_eda_checks"] = checks
        if category in {"dataset_schema", "schema", "metric", "validation", "leakage"}:
            item["priority"] = "P0"
        result.append(item)
    return result


def _adapt_legacy_scout_tasks(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    for raw in items:
        if not isinstance(raw, dict):
            result.append(raw)
            continue
        item = dict(raw)
        item["module"] = _legacy_scout_module_name(item.get("module"))
        result.append(item)
    return result


def _derive_hypothesis_index(tasks: list[Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or task.get("id") or "")
        for hypothesis_id in task.get("related_hypothesis_ids") or []:
            values = result.setdefault(str(hypothesis_id), [])
            if task_id and task_id not in values:
                values.append(task_id)
    return result


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
