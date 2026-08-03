from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_researcher.facts.competition import fetch_competition_metadata
from kaggle_researcher.facts.cv_lb import build_cv_lb_pairs, diagnose_cv_lb
from kaggle_researcher.facts.discussions import (
    discussion_auth_mode,
    fetch_competition_discussions,
    fetch_winner_writeups,
)
from kaggle_researcher.facts.files import fetch_file_manifest
from kaggle_researcher.facts.models import (
    CompetitionFacts,
    FileManifest,
    LeaderboardStability,
    NotebookFacts,
    UserConstraints,
)
from kaggle_researcher.facts.notebook_ast import (
    assign_lineage_clusters,
    ast_fingerprint,
    extract_observations,
)
from kaggle_researcher.facts.notebooks import list_competition_notebooks, pull_notebook

NOTEBOOK_CONCURRENCY = 4


def collect_facts(
    slug: str,
    max_notebooks: int,
    max_discussions: int,
    similar: list[str],
    user_constraints: UserConstraints,
    max_sample_sub_bytes: int,
    writeups_per_competition: int = 10,
) -> CompetitionFacts:
    if writeups_per_competition <= 0:
        raise ValueError("writeups_per_competition must be a positive integer")
    try:
        metadata = fetch_competition_metadata(slug)
    except Exception as exc:
        raise RuntimeError(
            f"competition metadata collection failed ({type(exc).__name__})"
        ) from exc

    collection_errors: list[str] = []
    limitations: list[str] = []
    try:
        files = fetch_file_manifest(slug, max_sample_sub_bytes)
    except Exception as exc:
        collection_errors.append(_stage_error("file manifest", exc))
        files = FileManifest(
            files=[],
            train_test_size_ratio=None,
            sample_submission_columns=[],
            sample_submission_source="unavailable",
            limitations=["File manifest collection failed."],
        )

    notebooks = _collect_notebooks(
        slug=slug,
        max_notebooks=max_notebooks,
        collection_errors=collection_errors,
    )
    try:
        notebooks = assign_lineage_clusters(notebooks)
    except Exception as exc:
        collection_errors.append(_stage_error("notebook lineage clustering", exc))

    try:
        cv_lb_pairs = build_cv_lb_pairs(notebooks, metadata.metric_name)
    except Exception as exc:
        collection_errors.append(_stage_error("CV/LB pairing", exc))
        cv_lb_pairs = []
    cv_lb_diagnostics = diagnose_cv_lb(notebooks, cv_lb_pairs)

    discussions = []
    discussion_status = "empty"
    discussion_error: str | None = None
    try:
        discussion_result = fetch_competition_discussions(slug, max_discussions)
        discussions.extend(discussion_result)
        discussion_status = getattr(
            discussion_result,
            "status",
            "collected" if discussion_result else "empty",
        )
        discussion_error = getattr(discussion_result, "error", None)
        limitation = getattr(discussion_result, "limitation", None)
        if limitation:
            limitations.append(limitation)
        if discussion_status in {"failed", "unavailable"} and discussion_error:
            collection_errors.append(discussion_error)
    except Exception as exc:
        collection_errors.append(_stage_error("competition discussions", exc))
        discussion_status = "failed"
        discussion_error = _stage_error("competition discussions", exc)
    try:
        discussions.extend(fetch_winner_writeups(similar, writeups_per_competition))
    except Exception as exc:
        collection_errors.append(_stage_error("winner writeups", exc))

    similar_competitions = [
        LeaderboardStability(
            competition_id=competition_slug,
            status="not_computable",
            matched_teams=0,
            source="unavailable",
            not_computable_reason="Meta Kaggle dumps not configured.",
        )
        for competition_slug in similar
    ]
    return CompetitionFacts(
        competition_id=metadata.competition_id,
        collected_at=datetime.now(timezone.utc),
        metadata=metadata,
        files=files,
        notebooks=notebooks,
        discussions=discussions,
        similar_competitions=similar_competitions,
        cv_lb_pairs=cv_lb_pairs,
        cv_lb_diagnostics=cv_lb_diagnostics,
        discussion_collection_status=discussion_status,
        discussion_collection_error=discussion_error,
        discussion_auth_mode=discussion_auth_mode(),
        limitations=limitations,
        user_constraints=user_constraints,
        collection_errors=collection_errors,
    )


def _collect_notebooks(
    *,
    slug: str,
    max_notebooks: int,
    collection_errors: list[str],
) -> list[NotebookFacts]:
    try:
        notebook_records = list_competition_notebooks(slug, max_notebooks)
    except Exception as exc:
        collection_errors.append(_stage_error("notebook listing", exc))
        return []
    if not notebook_records:
        return []

    with tempfile.TemporaryDirectory(prefix="wavebreaker_notebooks_") as temp_dir:
        return asyncio.run(
            _collect_notebooks_async(
                notebook_records,
                Path(temp_dir),
                collection_errors,
            )
        )


async def _collect_notebooks_async(
    notebook_records: list[dict[str, Any]],
    temp_dir: Path,
    collection_errors: list[str],
) -> list[NotebookFacts]:
    semaphore = asyncio.Semaphore(NOTEBOOK_CONCURRENCY)
    tasks = [
        _collect_one_notebook(
            record=record,
            destination=temp_dir / f"notebook_{index:04d}",
            semaphore=semaphore,
        )
        for index, record in enumerate(notebook_records)
    ]
    results = await asyncio.gather(*tasks)
    notebooks: list[NotebookFacts] = []
    for _record, result in zip(notebook_records, results, strict=True):
        if isinstance(result, str):
            collection_errors.append(result)
        else:
            notebooks.append(result)
    return notebooks


async def _collect_one_notebook(
    *,
    record: dict[str, Any],
    destination: Path,
    semaphore: asyncio.Semaphore,
) -> NotebookFacts | str:
    ref = str(record.get("ref") or "<unknown>")
    try:
        async with semaphore:
            notebook_path = await asyncio.to_thread(
                pull_notebook,
                ref,
                destination,
            )
        if notebook_path is None:
            return f"notebook {ref} pull failed"
        observations, fingerprint = await asyncio.to_thread(
            _analyze_downloaded_notebook,
            notebook_path,
        )
        return NotebookFacts(
            ref=ref,
            title=str(record.get("title") or ref),
            author=record.get("author"),
            votes=record.get("votes", 0),
            public_score=record.get("public_score"),
            last_run=record.get("last_run"),
            ast_fingerprint=fingerprint,
            lineage_cluster_id=f"unassigned_{fingerprint[:12]}",
            splitters=observations["splitters"],
            models=observations["models"],
            metrics=observations["metrics"],
            feature_ops=observations["feature_ops"],
            declared_cv=observations["declared_cv"],
            declared_cv_observations=observations.get(
                "declared_cv_observations",
                [],
            ),
            parse_status=observations["parse_status"],
        )
    except Exception as exc:
        return _stage_error(f"notebook {ref}", exc)


def _analyze_downloaded_notebook(
    notebook_path: Path,
) -> tuple[dict[str, Any], str]:
    observations = extract_observations(notebook_path)
    fingerprint = ast_fingerprint(notebook_path)
    return observations, fingerprint


def _stage_error(stage: str, exc: BaseException) -> str:
    return f"{stage} failed ({type(exc).__name__})"
