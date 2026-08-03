from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kaggle_researcher.facts.competition as competition_module
from kaggle_researcher.facts.competition import fetch_competition_metadata

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "competition_object.json"
)
METADATA_FIELDS = [
    "title",
    "metric_name",
    "is_code_competition",
    "submissions_per_day",
    "max_team_size",
    "deadline",
    "reward",
    "category",
    "num_teams",
]


class FakeKaggleApi:
    competitions: list[Any] = []
    instances: list[FakeKaggleApi] = []

    def __init__(self) -> None:
        self.authenticated = False
        self.list_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def authenticate(self) -> None:
        self.authenticated = True

    def competitions_list(self, **kwargs: Any) -> list[Any]:
        self.list_calls.append(kwargs)
        return self.__class__.competitions


@pytest.fixture(autouse=True)
def fake_kaggle_api(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeKaggleApi.competitions = []
    FakeKaggleApi.instances = []

    def create_api() -> FakeKaggleApi:
        api = FakeKaggleApi()
        api.authenticate()
        return api

    monkeypatch.setattr(competition_module, "create_kaggle_api", create_api)


def _load_full_competition() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_full_fixture_maps_every_metadata_field() -> None:
    FakeKaggleApi.competitions = [_load_full_competition()]

    metadata = fetch_competition_metadata("example-competition")

    assert metadata.competition_id == "example-competition"
    assert metadata.title == "Example Competition"
    assert metadata.metric_name == "ROC AUC"
    assert metadata.is_code_competition is True
    assert metadata.submissions_per_day == 5
    assert metadata.max_team_size == 4
    assert metadata.deadline == datetime(2026, 8, 31, 23, 59, tzinfo=UTC)
    assert metadata.reward == "$50,000"
    assert metadata.category == "Featured"
    assert metadata.num_teams == 1234
    assert metadata.unavailable_fields == []

    api = FakeKaggleApi.instances[0]
    assert api.authenticated is True
    assert api.list_calls == [{"search": "example-competition"}]


def test_kaggle_2_response_object_is_unwrapped() -> None:
    competition = _load_full_competition()
    competition["ref"] = "https://www.kaggle.com/competitions/example-competition"
    FakeKaggleApi.competitions = SimpleNamespace(
        competitions=[competition],
        next_page_token=None,
    )

    metadata = fetch_competition_metadata("example-competition")

    assert metadata.title == "Example Competition"
    assert metadata.metric_name == "ROC AUC"
    assert metadata.evaluation_metric_raw == "ROC AUC"
    assert metadata.metric_status == "available"


def test_kaggle_2_metadata_fields_have_priority_and_preserve_false() -> None:
    FakeKaggleApi.competitions = [
        SimpleNamespace(
            ref="kaggle-2-competition",
            title="Kaggle 2 Competition",
            evaluation_metric="mAP",
            metric_name="Legacy Metric",
            metric_template="metric_template",
            is_kernels_submissions_only=False,
            is_code_competition=True,
            max_daily_submissions=7,
            submissions_per_day=3,
            max_team_size=2,
            deadline="2026-10-01T00:00:00Z",
            reward="Kudos",
            category="Community",
            team_count=348,
            num_teams=12,
        )
    ]

    metadata = fetch_competition_metadata("kaggle-2-competition")

    assert metadata.metric_name == "mAP"
    assert metadata.evaluation_metric_raw == "mAP"
    assert metadata.metric_status == "available"
    assert metadata.is_code_competition is False
    assert metadata.submissions_per_day == 7
    assert metadata.max_team_size == 2
    assert metadata.num_teams == 348
    assert metadata.unavailable_fields == []


@pytest.mark.parametrize(
    "placeholder",
    ["metric_template", " metric template ", "", "metric", "unknown", "UNKNOWN"],
)
def test_metric_placeholders_are_unavailable(placeholder: str) -> None:
    FakeKaggleApi.competitions = [
        {
            "ref": "placeholder-metric",
            "title": "Placeholder Metric",
            "evaluation_metric": placeholder,
        }
    ]

    metadata = fetch_competition_metadata("placeholder-metric")

    assert metadata.metric_name is None
    assert metadata.evaluation_metric_raw == (placeholder.strip() or None)
    assert metadata.metric_status == (
        "unavailable" if not placeholder.strip() else "placeholder"
    )
    assert "metric_name" in metadata.unavailable_fields


def test_kaggle_2_plural_code_competition_field_maps_true() -> None:
    FakeKaggleApi.competitions = [
        {
            "ref": "code-only",
            "title": "Code Only",
            "is_kernels_submissions_only": True,
        }
    ]

    metadata = fetch_competition_metadata("code-only")

    assert metadata.is_code_competition is True


def test_missing_fields_are_none_and_reported_without_description_inference() -> None:
    FakeKaggleApi.competitions = [
        {
            "ref": "example-competition",
            "title": "Sparse Competition",
            "description": "Submissions are ranked with macro F1.",
        }
    ]

    metadata = fetch_competition_metadata("example-competition")

    assert metadata.title == "Sparse Competition"
    assert metadata.metric_name is None
    assert metadata.evaluation_metric_raw is None
    assert metadata.metric_status == "unavailable"
    assert metadata.is_code_competition is None
    assert metadata.submissions_per_day is None
    assert metadata.max_team_size is None
    assert metadata.deadline is None
    assert metadata.reward is None
    assert metadata.category is None
    assert metadata.num_teams is None
    assert metadata.unavailable_fields == METADATA_FIELDS[1:]


def test_missing_competition_returns_all_fields_as_unavailable() -> None:
    FakeKaggleApi.competitions = [
        {
            **_load_full_competition(),
            "ref": "different-competition",
        }
    ]

    metadata = fetch_competition_metadata("missing-competition")

    assert metadata.competition_id == "missing-competition"
    assert metadata.unavailable_fields == METADATA_FIELDS
    for field_name in METADATA_FIELDS:
        assert getattr(metadata, field_name) is None


def test_tolerant_getter_supports_alternate_client_attribute_names() -> None:
    FakeKaggleApi.competitions = [
        SimpleNamespace(
            competition_ref="alternate-competition",
            competition_title="Alternate Competition",
            evaluation_metric="Log Loss",
            is_code_competition=False,
            submissions_per_day=3,
            max_team_size=2,
            deadline_date="2026-09-01T00:00:00Z",
            reward_display="Knowledge",
            category_name="Research",
            number_of_teams=250,
        )
    ]

    metadata = fetch_competition_metadata("alternate-competition")

    assert metadata.title == "Alternate Competition"
    assert metadata.metric_name == "Log Loss"
    assert metadata.is_code_competition is False
    assert metadata.submissions_per_day == 3
    assert metadata.max_team_size == 2
    assert metadata.deadline == datetime(2026, 9, 1, tzinfo=UTC)
    assert metadata.reward == "Knowledge"
    assert metadata.category == "Research"
    assert metadata.num_teams == 250
    assert metadata.unavailable_fields == []


def test_exact_ref_match_is_selected_from_search_results() -> None:
    wrong_match = {
        **_load_full_competition(),
        "ref": "example-competition-playground",
        "title": "Wrong Competition",
    }
    FakeKaggleApi.competitions = [wrong_match, _load_full_competition()]

    metadata = fetch_competition_metadata("example-competition")

    assert metadata.title == "Example Competition"
