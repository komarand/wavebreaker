from __future__ import annotations

import inspect
import math
import statistics
from typing import Any, Literal

from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    create_kaggle_api,
    unwrap_response,
)
from kaggle_researcher.facts.models import (
    LeaderboardEntry,
    LeaderboardShape,
    PublicLeaderboard,
)
from kaggle_researcher.facts.notebook_ast import metric_optimization_direction

_LEADERBOARD_COLLECTION_FIELDS = ("submissions", "entries", "leaderboard")
_SHAPE_RANKS = (1, 10, 25, 50, 100, 200)


def fetch_public_leaderboard(
    slug: str,
    api: Any | None = None,
    max_entries: int = 500,
) -> PublicLeaderboard:
    if max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")

    try:
        client = api or create_kaggle_api()
        response = GLOBAL_KAGGLE_POLICY.call(
            lambda: _leaderboard_view(client, slug, max_entries)
        )
        records = unwrap_response(response, *_LEADERBOARD_COLLECTION_FIELDS)
        entries = [
            _leaderboard_entry(record, fallback_rank=index + 1)
            for index, record in enumerate(records[:max_entries])
        ]
        return PublicLeaderboard(
            status="collected",
            entries=entries,
            entry_count=len(entries),
            unavailable_reason=None,
        )
    except Exception as exc:
        return PublicLeaderboard(
            status="unavailable",
            entries=[],
            entry_count=0,
            unavailable_reason=f"{type(exc).__name__}: {exc}",
        )


def compute_leaderboard_shape(
    leaderboard: PublicLeaderboard,
    metric_canonical: str | None,
) -> LeaderboardShape | None:
    if leaderboard.status != "collected":
        return None
    entries = sorted(
        (
            entry
            for entry in leaderboard.entries
            if entry.score is not None and entry.rank is not None
        ),
        key=lambda entry: entry.rank,
    )
    if len(entries) < 10:
        return None

    scores = [entry.score for entry in entries]
    adjacent_deltas = [
        abs(current - previous)
        for previous, current in zip(scores, scores[1:], strict=False)
    ]
    nonzero_deltas = [delta for delta in adjacent_deltas if delta != 0]
    tied_adjacent_pairs = len(adjacent_deltas) - len(nonzero_deltas)
    nonzero_adjacent_pairs = len(nonzero_deltas)
    tied_ratio = (
        round(tied_adjacent_pairs / len(adjacent_deltas), 4)
        if adjacent_deltas
        else None
    )
    median_adjacent_delta = (
        statistics.median(nonzero_deltas) if nonzero_deltas else None
    )
    median_score = statistics.median(scores)
    if median_adjacent_delta is None:
        teams_within_delta = None
        plateau_ratio = None
    else:
        teams_within_delta = sum(
            abs(score - median_score) <= median_adjacent_delta for score in scores
        )
        plateau_ratio = round(
            sum(
                abs(score - median_score) <= 5 * median_adjacent_delta
                for score in scores
            )
            / len(scores),
            4,
        )

    scores_by_rank = {entry.rank: entry.score for entry in entries}
    return LeaderboardShape(
        entry_count=len(entries),
        top_score=scores[0],
        score_at_rank={
            rank: scores_by_rank[rank]
            for rank in _SHAPE_RANKS
            if rank in scores_by_rank
        },
        median_adjacent_delta=median_adjacent_delta,
        tied_adjacent_pairs=tied_adjacent_pairs,
        nonzero_adjacent_pairs=nonzero_adjacent_pairs,
        tied_ratio=tied_ratio,
        teams_within_median_delta_of_median=teams_within_delta,
        plateau_ratio=plateau_ratio,
        span_top_to_last=abs(scores[0] - scores[-1]),
        direction=_leaderboard_direction(scores, metric_canonical),
    )


def _leaderboard_direction(
    scores: list[float],
    metric_canonical: str | None,
) -> Literal["higher_is_better", "lower_is_better", "unknown"]:
    metric_direction = metric_optimization_direction(metric_canonical)
    if metric_direction == "maximize":
        return "higher_is_better"
    if metric_direction == "minimize":
        return "lower_is_better"
    decreases = all(left >= right for left, right in zip(scores, scores[1:], strict=False))
    increases = all(left <= right for left, right in zip(scores, scores[1:], strict=False))
    if decreases and any(left > right for left, right in zip(scores, scores[1:], strict=False)):
        return "higher_is_better"
    if increases and any(left < right for left, right in zip(scores, scores[1:], strict=False)):
        return "lower_is_better"
    return "unknown"


def _leaderboard_entry(record: Any, *, fallback_rank: int) -> LeaderboardEntry:
    # Read submission_date as part of the version-tolerant SDK contract even though
    # the compact facts model intentionally does not persist it.
    _field(record, "submission_date", "submissionDate")
    return LeaderboardEntry(
        team_name=_optional_text(_field(record, "team_name", "teamName")),
        score=_finite_float(_field(record, "score", "publicScore")),
        rank=_optional_int(_field(record, "rank")) or fallback_rank,
    )


def _leaderboard_view(client: Any, slug: str, max_entries: int) -> Any:
    method = client.competition_leaderboard_view
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_page_size = any(
        parameter.name == "page_size" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_page_size:
        return method(slug, page_size=max_entries)
    return method(slug)


def _field(record: Any, *names: str) -> Any:
    if isinstance(record, dict):
        normalized = {_normalized_key(str(key)): value for key, value in record.items()}
        for name in names:
            key = _normalized_key(name)
            if key in normalized:
                return normalized[key]
        return None
    for name in names:
        try:
            return getattr(record, name)
        except (AttributeError, TypeError):
            continue
        except Exception:
            continue
    return None


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
