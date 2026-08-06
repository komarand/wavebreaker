from __future__ import annotations

import inspect
import math
from typing import Any

from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    create_kaggle_api,
    unwrap_response,
)
from kaggle_researcher.facts.models import LeaderboardEntry, PublicLeaderboard

_LEADERBOARD_COLLECTION_FIELDS = ("submissions", "entries", "leaderboard")


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
        entries = [_leaderboard_entry(record) for record in records[:max_entries]]
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


def _leaderboard_entry(record: Any) -> LeaderboardEntry:
    # Read submission_date as part of the version-tolerant SDK contract even though
    # the compact facts model intentionally does not persist it.
    _field(record, "submission_date", "submissionDate")
    return LeaderboardEntry(
        team_name=_optional_text(_field(record, "team_name", "teamName")),
        score=_finite_float(_field(record, "score", "publicScore")),
        rank=_optional_int(_field(record, "rank")),
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
