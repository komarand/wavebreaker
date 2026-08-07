from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.facts import competition_leaderboard
from kaggle_researcher.facts.competition_leaderboard import (
    compute_leaderboard_shape,
    fetch_public_leaderboard,
)
from kaggle_researcher.facts.models import LeaderboardEntry, PublicLeaderboard

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "leaderboard_view.json"
)


class _ImmediatePolicy:
    def call(self, operation):
        return operation()


class _FakeApi:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[str] = []

    def competition_leaderboard_view(self, slug: str) -> Any:
        self.calls.append(slug)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def immediate_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        competition_leaderboard,
        "GLOBAL_KAGGLE_POLICY",
        _ImmediatePolicy(),
    )


@pytest.mark.parametrize("fixture_key", ["list", "wrapper"])
def test_fetch_public_leaderboard_accepts_raw_list_and_entries_wrapper(
    fixture_key: str,
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    api = _FakeApi(fixture[fixture_key])

    leaderboard = fetch_public_leaderboard("example", api=api)

    assert api.calls == ["example"]
    assert leaderboard.status == "collected"
    assert leaderboard.entry_count == 2
    assert leaderboard.entries[0].team_name == "Alice Smith"
    assert leaderboard.entries[0].score == pytest.approx(0.8123)
    assert leaderboard.entries[0].rank == 1
    if fixture_key == "wrapper":
        assert leaderboard.entries[1].score is None
        assert leaderboard.entries[1].rank == 2


def test_fetch_public_leaderboard_honors_entry_limit() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    leaderboard = fetch_public_leaderboard(
        "example",
        api=_FakeApi(fixture["list"]),
        max_entries=1,
    )

    assert leaderboard.entry_count == 1
    assert len(leaderboard.entries) == 1


def test_unavailable_leaderboard_is_nonfatal() -> None:
    leaderboard = fetch_public_leaderboard(
        "example",
        api=_FakeApi(error=RuntimeError("not permitted")),
    )

    assert leaderboard.status == "unavailable"
    assert leaderboard.entries == []
    assert leaderboard.entry_count == 0
    assert "RuntimeError" in str(leaderboard.unavailable_reason)


def test_entry_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fetch_public_leaderboard("example", api=_FakeApi([]), max_entries=0)


def test_compute_leaderboard_shape_for_known_200_entry_distribution() -> None:
    leaderboard = _leaderboard(
        [LeaderboardEntry(team_name=str(rank), score=1 - rank / 1_000, rank=rank)
         for rank in range(1, 201)]
    )

    shape = compute_leaderboard_shape(leaderboard, None)

    assert shape is not None
    assert shape.entry_count == 200
    assert shape.top_score == pytest.approx(0.999)
    assert shape.score_at_rank == pytest.approx(
        {1: 0.999, 10: 0.99, 25: 0.975, 50: 0.95, 100: 0.9, 200: 0.8}
    )
    assert shape.median_adjacent_delta == pytest.approx(0.001)
    assert shape.teams_within_median_delta_of_median == 2
    assert shape.plateau_ratio == pytest.approx(0.05)
    assert shape.span_top_to_last == pytest.approx(0.199)
    assert shape.direction == "higher_is_better"


def test_compute_leaderboard_shape_requires_ten_usable_entries() -> None:
    leaderboard = _leaderboard(
        [LeaderboardEntry(team_name=str(rank), score=float(rank), rank=rank)
         for rank in range(1, 6)]
    )

    assert compute_leaderboard_shape(leaderboard, None) is None


def test_compute_leaderboard_shape_handles_identical_scores() -> None:
    leaderboard = _leaderboard(
        [LeaderboardEntry(team_name=str(rank), score=0.5, rank=rank)
         for rank in range(1, 11)]
    )

    shape = compute_leaderboard_shape(leaderboard, None)

    assert shape is not None
    assert shape.median_adjacent_delta == 0
    assert shape.teams_within_median_delta_of_median is None
    assert shape.plateau_ratio is None


def test_compute_leaderboard_shape_infers_lower_is_better_from_order() -> None:
    leaderboard = _leaderboard(
        [LeaderboardEntry(team_name=str(rank), score=float(rank), rank=rank)
         for rank in range(1, 11)]
    )

    shape = compute_leaderboard_shape(leaderboard, None)

    assert shape is not None
    assert shape.direction == "lower_is_better"


def _leaderboard(entries: list[LeaderboardEntry]) -> PublicLeaderboard:
    return PublicLeaderboard(
        status="collected",
        entries=entries,
        entry_count=len(entries),
        unavailable_reason=None,
    )
