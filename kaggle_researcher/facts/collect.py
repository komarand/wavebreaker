from __future__ import annotations

import asyncio
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from kaggle_researcher.config import (
    get_max_similar_verifications,
    get_notebook_concurrency,
)
from kaggle_researcher.facts.code_aggregates import compute_code_aggregates
from kaggle_researcher.facts.competition import fetch_competition_metadata
from kaggle_researcher.facts.competition_leaderboard import (
    compute_leaderboard_shape,
    fetch_public_leaderboard,
)
from kaggle_researcher.facts.cv_lb import (
    build_cv_lb_pairs,
    build_leaderboard_cv_lb_pairs,
    diagnose_cv_lb,
    match_leaderboard_scores,
)
from kaggle_researcher.facts.discussions import (
    discussion_auth_mode,
    fetch_competition_discussions,
    fetch_winner_writeups,
)
from kaggle_researcher.facts.files import fetch_file_manifest
from kaggle_researcher.facts.models import (
    CompetitionFacts,
    DatasetReference,
    FileManifest,
    LeaderboardStability,
    NotebookFacts,
    PublicLeaderboard,
    ScoreObservation,
    SimilarSearchDiagnostics,
    UserConstraints,
)
from kaggle_researcher.facts.notebook_ast import (
    assign_lineage_clusters,
    ast_fingerprint,
    canonicalize_metric_label,
    diagnose_scores,
    extract_observations,
    extract_score_observations,
    metric_optimization_direction,
    recanonicalize_score_observations,
)
from kaggle_researcher.facts.notebooks import (
    NotebookPullResult,
    list_competition_notebooks,
)
from kaggle_researcher.facts.notebooks import (
    pull_notebook_with_diagnostics as pull_notebook,
)
from kaggle_researcher.facts.similar import build_similar_candidates


@dataclass(frozen=True, slots=True)
class _NotebookCollectionFailure:
    message: str
    http_status: int | None = None
    error_type: str | None = None


def collect_facts(
    slug: str,
    max_notebooks: int,
    max_discussions: int,
    similar: list[str],
    user_constraints: UserConstraints,
    max_sample_sub_bytes: int,
    writeups_per_competition: int = 10,
    notebook_concurrency: int | None = None,
    max_similar_verifications: int | None = None,
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

    concurrency = (
        get_notebook_concurrency() if notebook_concurrency is None else notebook_concurrency
    )
    if concurrency <= 0:
        raise ValueError("notebook_concurrency must be a positive integer")
    (
        notebooks,
        notebooks_failed_by_status,
        notebooks_failed_by_exception,
    ) = _collect_notebooks(
        slug=slug,
        max_notebooks=max_notebooks,
        collection_errors=collection_errors,
        competition_metric_name=metadata.metric_name,
        notebook_concurrency=concurrency,
    )
    notebooks = recanonicalize_score_observations(
        notebooks,
        competition_metric_name=metadata.metric_name,
    )
    try:
        notebooks = assign_lineage_clusters(notebooks)
    except Exception as exc:
        collection_errors.append(_stage_error("notebook lineage clustering", exc))
    code_aggregates = compute_code_aggregates(
        notebooks,
        optimization_direction=metric_optimization_direction(metadata.metric_name),
    )
    dataset_references = aggregate_dataset_references(notebooks)

    try:
        public_leaderboard = fetch_public_leaderboard(slug)
    except Exception as exc:
        public_leaderboard = PublicLeaderboard(
            status="unavailable",
            entries=[],
            entry_count=0,
            unavailable_reason=f"{type(exc).__name__}: {exc}",
        )
    if public_leaderboard.status == "unavailable":
        collection_errors.append(
            "public leaderboard unavailable"
            + (
                f" ({public_leaderboard.unavailable_reason})"
                if public_leaderboard.unavailable_reason
                else ""
            )
        )
    public_leaderboard = public_leaderboard.model_copy(
        update={
            "shape": compute_leaderboard_shape(
                public_leaderboard,
                canonicalize_metric_label(metadata.metric_name),
            )
        }
    )

    try:
        leaderboard_matches = match_leaderboard_scores(notebooks, public_leaderboard)
    except Exception as exc:
        collection_errors.append(_stage_error("leaderboard matching", exc))
        leaderboard_matches = {}

    try:
        observation_pairs = build_cv_lb_pairs(notebooks, metadata.metric_name)
        leaderboard_pairs = build_leaderboard_cv_lb_pairs(
            notebooks,
            leaderboard_matches,
            metadata.metric_name,
        )
        cv_lb_pairs = [*observation_pairs, *leaderboard_pairs]
        implausible_gap_pairs = [
            *getattr(observation_pairs, "implausible_gap_pairs", []),
            *getattr(leaderboard_pairs, "implausible_gap_pairs", []),
        ]
    except Exception as exc:
        collection_errors.append(_stage_error("CV/LB pairing", exc))
        cv_lb_pairs = []
        implausible_gap_pairs = []
    cv_lb_diagnostics = diagnose_cv_lb(
        notebooks,
        cv_lb_pairs,
        metadata.metric_name,
        leaderboard_matches,
    )
    score_diagnostics = diagnose_scores(notebooks).model_copy(
        update={
            "notebooks_failed_by_status": dict(sorted(notebooks_failed_by_status.items())),
            "notebooks_failed_by_exception": dict(sorted(notebooks_failed_by_exception.items())),
        }
    )

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
        if (
            discussion_status
            in {
                "partial",
                "rate_limited",
                "failed",
                "unavailable",
            }
            and discussion_error
        ):
            collection_errors.append(discussion_error)
    except Exception as exc:
        collection_errors.append(_stage_error("competition discussions", exc))
        discussion_status = "failed"
        discussion_error = _stage_error("competition discussions", exc)

    verification_limit = (
        get_max_similar_verifications()
        if max_similar_verifications is None
        else max_similar_verifications
    )
    similar_candidates = []
    similar_diagnostics: SimilarSearchDiagnostics | None = None
    try:
        similar_candidates, similar_diagnostics = build_similar_candidates(
            discussions=discussions,
            self_metadata=metadata,
            self_submission_columns=files.sample_submission_columns,
            manual=similar,
            max_verifications=verification_limit,
            metadata_fetcher=fetch_competition_metadata,
        )
    except Exception as exc:
        collection_errors.append(_stage_error("similar competition verification", exc))
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
        code_aggregates=code_aggregates,
        public_leaderboard=public_leaderboard,
        leaderboard_matches=list(leaderboard_matches.values()),
        dataset_references=dataset_references,
        discussions=discussions,
        similar_competitions=similar_competitions,
        similar_candidates=similar_candidates,
        similar_diagnostics=similar_diagnostics,
        cv_lb_pairs=cv_lb_pairs,
        implausible_gap_pairs=implausible_gap_pairs,
        cv_lb_diagnostics=cv_lb_diagnostics,
        score_diagnostics=score_diagnostics,
        discussion_collection_status=discussion_status,
        discussion_collection_error=discussion_error,
        discussion_auth_mode=discussion_auth_mode(),
        limitations=limitations,
        user_constraints=user_constraints,
        collection_errors=collection_errors,
    )


def aggregate_dataset_references(
    notebooks: list[NotebookFacts],
) -> list[DatasetReference]:
    grouped: dict[str, list[NotebookFacts]] = {}
    display_slugs: dict[str, str] = {}
    for notebook in notebooks:
        for slug in notebook.dataset_paths:
            key = slug.casefold()
            display_slugs.setdefault(key, slug)
            grouped.setdefault(key, []).append(notebook)

    references: list[DatasetReference] = []
    for key, members in grouped.items():
        slug = display_slugs[key]
        notebook_refs = sorted({notebook.ref for notebook in members})
        cluster_ids = sorted({notebook.lineage_cluster_id for notebook in members})
        references.append(
            DatasetReference(
                slug=slug,
                raw_path=f"/kaggle/input/{slug}/",
                notebook_refs=notebook_refs,
                lineage_cluster_ids=cluster_ids,
                reference_count=len(notebook_refs),
                cluster_count=len(cluster_ids),
            )
        )
    return sorted(references, key=lambda item: (-item.cluster_count, item.slug.casefold()))


def _collect_notebooks(
    *,
    slug: str,
    max_notebooks: int,
    collection_errors: list[str],
    notebook_concurrency: int,
    competition_metric_name: str | None = None,
) -> tuple[list[NotebookFacts], dict[int, int], dict[str, int]]:
    try:
        notebook_records = list_competition_notebooks(slug, max_notebooks)
    except Exception as exc:
        collection_errors.append(_stage_error("notebook listing", exc))
        return [], {}, {}
    if not notebook_records:
        return [], {}, {}

    with tempfile.TemporaryDirectory(prefix="wavebreaker_notebooks_") as temp_dir:
        return asyncio.run(
            _collect_notebooks_async(
                notebook_records,
                Path(temp_dir),
                collection_errors,
                notebook_concurrency,
                competition_metric_name,
                slug,
            )
        )


async def _collect_notebooks_async(
    notebook_records: list[dict[str, Any]],
    temp_dir: Path,
    collection_errors: list[str],
    notebook_concurrency: int,
    competition_metric_name: str | None = None,
    competition_id: str | None = None,
) -> tuple[list[NotebookFacts], dict[int, int], dict[str, int]]:
    semaphore = asyncio.Semaphore(notebook_concurrency)
    tasks = [
        asyncio.create_task(
            _collect_one_notebook_indexed(
                index=index,
                record=record,
                destination=temp_dir / f"notebook_{index:04d}",
                semaphore=semaphore,
                competition_metric_name=competition_metric_name,
                competition_id=competition_id,
            )
        )
        for index, record in enumerate(notebook_records)
    ]
    results: list[NotebookFacts | _NotebookCollectionFailure | None] = [None] * len(tasks)
    completed_ok = 0
    completed_failed = 0
    progress = tqdm(
        total=len(tasks),
        desc="Pulling notebooks",
        unit="notebook",
        dynamic_ncols=True,
        disable=None,
    )
    try:
        for completed in asyncio.as_completed(tasks):
            index, result = await completed
            results[index] = result
            if isinstance(result, _NotebookCollectionFailure):
                completed_failed += 1
            else:
                completed_ok += 1
            progress.set_postfix(ok=completed_ok, failed=completed_failed)
            progress.update(1)
    finally:
        progress.close()

    notebooks: list[NotebookFacts] = []
    failures_by_status: Counter[int] = Counter()
    failures_by_exception: Counter[str] = Counter()
    for _record, result in zip(notebook_records, results, strict=True):
        if isinstance(result, _NotebookCollectionFailure):
            collection_errors.append(result.message)
            if result.http_status is not None:
                failures_by_status[result.http_status] += 1
            if result.error_type is not None:
                failures_by_exception[result.error_type] += 1
        else:
            notebooks.append(result)
    return notebooks, dict(failures_by_status), dict(failures_by_exception)


async def _collect_one_notebook_indexed(
    *,
    index: int,
    record: dict[str, Any],
    destination: Path,
    semaphore: asyncio.Semaphore,
    competition_metric_name: str | None = None,
    competition_id: str | None = None,
) -> tuple[int, NotebookFacts | _NotebookCollectionFailure]:
    result = await _collect_one_notebook(
        record=record,
        destination=destination,
        semaphore=semaphore,
        competition_metric_name=competition_metric_name,
        competition_id=competition_id,
    )
    return index, result


async def _collect_one_notebook(
    *,
    record: dict[str, Any],
    destination: Path,
    semaphore: asyncio.Semaphore,
    competition_metric_name: str | None = None,
    competition_id: str | None = None,
) -> NotebookFacts | _NotebookCollectionFailure:
    ref = str(record.get("ref") or "<unknown>")
    try:
        async with semaphore:
            raw_pull_result = await asyncio.to_thread(
                pull_notebook,
                ref,
                destination,
            )
        pull_result = (
            raw_pull_result
            if isinstance(raw_pull_result, NotebookPullResult)
            else NotebookPullResult(path=raw_pull_result)
        )
        notebook_path = pull_result.path
        if notebook_path is None:
            detail = pull_result.failure_detail()
            suffix = f" ({detail})" if detail is not None else ""
            return _NotebookCollectionFailure(
                message=f"notebook {ref} pull failed{suffix}",
                http_status=pull_result.http_status,
                error_type=pull_result.error_type,
            )
        observations, fingerprint = await asyncio.to_thread(
            _analyze_downloaded_notebook,
            notebook_path,
            competition_metric_name,
            competition_id,
        )
        context_scores, context_candidates, context_excluded = _notebook_context_scores(
            record, competition_metric_name
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
            score_observations=[
                *observations.get("score_observations", []),
                *context_scores,
            ],
            score_candidates_seen=(
                observations.get("score_candidates_seen", 0) + context_candidates
            ),
            score_candidates_excluded=observations.get(
                "score_candidates_excluded",
                0,
            )
            + context_excluded,
            style_markup_stripped_cells=observations.get(
                "style_markup_stripped_cells",
                0,
            ),
            style_markup_stripped_markdown_cells=observations.get(
                "style_markup_stripped_markdown_cells",
                0,
            ),
            style_markup_stripped_code_strings=observations.get(
                "style_markup_stripped_code_strings",
                0,
            ),
            dataset_paths=observations.get("dataset_paths", []),
            parse_status=observations["parse_status"],
        )
    except Exception as exc:
        return _NotebookCollectionFailure(
            _stage_error(f"notebook {ref}", exc),
            error_type=type(exc).__name__,
        )


def _analyze_downloaded_notebook(
    notebook_path: Path,
    competition_metric_name: str | None = None,
    competition_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    metric_hints = (competition_metric_name,) if competition_metric_name is not None else ()
    observations = extract_observations(
        notebook_path,
        metric_hints=metric_hints,
    )
    if competition_id is not None:
        observations["dataset_paths"] = [
            slug
            for slug in observations.get("dataset_paths", [])
            if slug.casefold() != competition_id.casefold()
        ]
    fingerprint = ast_fingerprint(notebook_path)
    return observations, fingerprint


def _notebook_context_scores(
    record: dict[str, Any],
    competition_metric_name: str | None = None,
) -> tuple[list[ScoreObservation], int, int]:
    observations: list[ScoreObservation] = []
    candidates_seen = 0
    candidates_excluded = 0
    seen: set[tuple[float, str | None]] = set()
    ref = str(record.get("ref") or "")
    contexts = (
        (str(record.get("title") or ""), "title"),
        (ref.rsplit("/", 1)[-1], "ref"),
    )
    for text, source in contexts:
        added, source_candidates, source_excluded = extract_score_observations(
            text,
            locator=source,
            source=source,
            metric_hints=(
                (competition_metric_name,) if competition_metric_name is not None else ()
            ),
        )
        candidates_seen += source_candidates
        candidates_excluded += source_excluded
        for observation in added:
            key = (observation.value, observation.metric_raw)
            if key in seen:
                continue
            seen.add(key)
            observations.append(observation)
    return observations, candidates_seen, candidates_excluded


def _stage_error(stage: str, exc: BaseException) -> str:
    return f"{stage} failed ({type(exc).__name__})"
