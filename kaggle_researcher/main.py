from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
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
from kaggle_researcher.config import load_config
from kaggle_researcher.eda import orchestrator as eda_orchestrator
from kaggle_researcher.eda.schemas import EdaEvidencePack, EdaRunConfig, ResearchHypotheses
from kaggle_researcher.embedder import embed_texts
from kaggle_researcher.planner import fallback_plan, plan
from kaggle_researcher.research_scout import (
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
)
from kaggle_researcher.store.domain_memory import DomainMemory
from kaggle_researcher.store.pg_store import PgStore


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
    debug: bool = False,
    no_github: bool = False,
    fast: bool = False,
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

    try:
        _stage("[1/8] Planning search queries...", show_progress)
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
        source_documents = await _collect_sources(
            plan_data=plan_data,
            competition_id=resolved_competition_id,
            settings=settings,
            warnings=warnings,
            show_progress=show_progress,
            no_github=no_github,
            fast=fast,
        )
        if not source_documents:
            raise RuntimeError("No source documents were collected")
        num_sources = _source_counts(source_documents)
        _add_missing_source_warnings(num_sources, warnings, no_github=no_github)
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
        if not embeddings:
            raise RuntimeError("No embeddings were generated")
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
            fast=fast,
        )
        _write_json_artifact(run_dir, "retrieved_documents.json", retrieved_documents)
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

            migration = migrate_research_hypotheses_payload(scout_payload)
            canonical_hypotheses = ResearchHypotheses.model_validate(migration.canonical_payload)
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
            task_plan_migration = migrate_eda_task_plan_payload(eda_task_plan)
            canonical_task_plan = EdaTaskPlan.model_validate(task_plan_migration.canonical_payload)
            validate_research_artifact_bundle(canonical_hypotheses, canonical_task_plan)
            scout_summary = build_research_scout_summary(scout_payload)
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
            final_strategy = await synthesize_final_strategy(
                competition_desc=competition_desc,
                plan_data=plan_data,
                retrieved_documents=retrieved_documents,
                domain_patterns=_load_domain_patterns(run_dir),
                research_hypotheses=_load_json_model(
                    research_hypotheses_for_final,
                    ResearchHypotheses,
                ),
                eda_evidence_pack=eda_pack_for_final,
                eda_summary_text=_load_optional_text(eda_summary_path),
                reasoning_outputs=_reasoning_outputs_summary(run_dir),
                client=client,
                model=settings.deepseek_v4_pro,
            )
            final_strategy_path, final_strategy_summary_path = _write_final_strategy_outputs(
                Path(final_output_dir) if final_output_dir is not None else run_dir,
                final_strategy,
                eda_evidence_pack=eda_pack_for_final.model_dump(mode="json"),
            )
            _stage(f"Final strategy saved to: {final_strategy_path}", show_progress)
        result = ResearchRunResult(
            competition_id=resolved_competition_id,
            mode=mode,
            report_path=str(actual_report_path),
            num_documents=len(summarized_documents),
            num_sources=num_sources,
            warnings=warnings,
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
        return result
    except Exception:
        _write_json_artifact(run_dir, "warnings.json", warnings)
        raise
    finally:
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
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    try:
        documents.extend(
            await asyncio.to_thread(
                _collect_kaggle_sources,
                plan_data,
                competition_id,
                settings,
                warnings,
                show_progress,
                fast,
            )
        )
    except Exception as exc:
        warnings.append(f"Kaggle source collection failed: {exc}")
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
            fast,
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
            )
        )
    return documents


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
) -> list[SourceDocument]:
    if not plan_data.github_queries:
        return []
    _stage("Collecting GitHub repos...", show_progress)
    try:
        raw_repos = await search_repos(
            plan_data.github_queries,
            token=settings.github_token,
            max_repos=_fast_limit(settings.max_repos, FAST_MAX_REPOS, fast),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


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
            resume_run_dir=args.resume_run_dir,
            force_rerun_stages=set(args.force_rerun_stage),
            disable_progress=args.no_progress,
        )
        result = await run_full_research(config)
        print(f"Full run status: {result.status}")
        print(f"Run directory: {result.run_dir}")
        print(f"Final strategy: {result.final_strategy_path}")
        print(f"Final report: {result.final_report_path}")
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.competition_url or not args.competition_desc:
        print("Provide competition_url and competition_desc to run the minimal pipeline.")
        return 0

    mode = args.mode
    if mode == "full" and args.report_mode == "minimal":
        mode = "minimal"

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
        debug=args.debug,
        no_github=args.no_github,
        fast=args.fast,
    )
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

    print("Research run complete.")
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


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
