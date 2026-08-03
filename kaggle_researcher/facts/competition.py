from __future__ import annotations

from typing import Any

from kaggle_researcher.facts.kaggle_api import create_kaggle_api, unpack_list_response
from kaggle_researcher.facts.models import CompetitionMetadata

_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "title": ("title", "competitionTitle", "competition_title"),
    "metric_name": (
        "evaluation_metric",
        "evaluationMetric",
        "metric_name",
        "metricName",
    ),
    "is_code_competition": (
        "is_kernels_submissions_only",
        "isKernelsSubmissionsOnly",
        "isKernelsSubmission",
        "is_kernels_submission",
        "is_code_competition",
        "isCodeCompetition",
    ),
    "submissions_per_day": (
        "max_daily_submissions",
        "maxDailySubmissions",
        "submissions_per_day",
        "submissionsPerDay",
    ),
    "max_team_size": ("max_team_size", "maxTeamSize"),
    "deadline": ("deadline", "deadlineDate", "deadline_date"),
    "reward": ("reward", "rewardDisplay", "reward_display"),
    "category": ("category", "categoryName", "category_name"),
    "num_teams": (
        "team_count",
        "teamCount",
        "num_teams",
        "numTeams",
        "number_of_teams",
        "numberOfTeams",
    ),
}
_REF_CANDIDATES = ("ref", "competitionRef", "competition_ref", "slug", "id")


def fetch_competition_metadata(
    slug: str,
    api: Any | None = None,
) -> CompetitionMetadata:
    if api is None:
        api = create_kaggle_api()
    response = api.competitions_list(search=slug)
    competitions = unpack_list_response(response, "competitions").items
    competition = next(
        (
            candidate
            for candidate in competitions or []
            if _competition_ref_matches(
                _get_competition_value(candidate, *_REF_CANDIDATES),
                slug,
            )
        ),
        None,
    )

    values: dict[str, Any] = {}
    unavailable_fields: list[str] = []
    for field_name, candidate_names in _FIELD_CANDIDATES.items():
        value = (
            _get_competition_value(competition, *candidate_names)
            if competition is not None
            else None
        )
        if field_name == "metric_name":
            raw_metric = _normalize_raw_metric(value)
            values["evaluation_metric_raw"] = raw_metric
            values["metric_status"] = _metric_status(raw_metric)
            value = _normalize_metric_name(raw_metric)
        values[field_name] = value
        if value is None:
            unavailable_fields.append(field_name)

    return CompetitionMetadata(
        competition_id=slug,
        unavailable_fields=unavailable_fields,
        **values,
    )


def _get_competition_value(competition: Any, *names: str) -> Any:
    if competition is None:
        return None

    if isinstance(competition, dict):
        for name in names:
            if name in competition and _has_value(competition[name]):
                return competition[name]
        normalized_values = {
            _normalize_key(str(key)): value for key, value in competition.items()
        }
        for name in names:
            value = normalized_values.get(_normalize_key(name))
            if _has_value(value):
                return value
        return None

    for name in names:
        try:
            value = getattr(competition, name)
        except Exception:
            continue
        if _has_value(value):
            return value

    normalized_names = {_normalize_key(name) for name in names}
    for attribute_name in dir(competition):
        if _normalize_key(attribute_name) not in normalized_names:
            continue
        try:
            value = getattr(competition, attribute_name)
        except Exception:
            continue
        if _has_value(value):
            return value

    return None


def _competition_ref_matches(value: Any, slug: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().rstrip("/")
    return normalized == slug or normalized.rsplit("/", 1)[-1] == slug


def _normalize_metric_name(value: Any) -> str | None:
    metric_name = _normalize_raw_metric(value)
    if metric_name is None:
        return None
    normalized = " ".join(metric_name.lower().replace("_", " ").split())
    if normalized in {"", "metric", "metric template", "unknown"}:
        return None
    return metric_name


def _normalize_raw_metric(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    metric_name = value.strip()
    return metric_name or None


def _metric_status(
    raw_metric: str | None,
) -> str:
    if raw_metric is None:
        return "unavailable"
    return "available" if _normalize_metric_name(raw_metric) is not None else "placeholder"


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _normalize_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())
