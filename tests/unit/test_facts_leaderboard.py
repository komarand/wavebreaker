from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from kaggle_researcher.facts import leaderboard
from kaggle_researcher.facts.leaderboard import compute_stability
from kaggle_researcher.facts.models import LeaderboardStability

FIXTURE = Path("tests/fixtures/facts/meta_kaggle_submissions.csv")


def test_missing_meta_kaggle_directory_is_expected_not_computable() -> None:
    result = compute_stability("synthetic-comp", None)

    assert result.status == "not_computable"
    assert result.source == "unavailable"
    assert result.matched_teams == 0
    assert result.match_fraction is None
    assert result.not_computable_reason == "Meta Kaggle dumps are not configured."
    assert _derived_metrics(result) == (None, None, None)


def test_missing_required_csvs_are_expected_not_computable(tmp_path: Path) -> None:
    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "not_computable"
    assert result.source == "unavailable"
    assert "Competitions.csv" in result.not_computable_reason
    assert "Teams.csv" in result.not_computable_reason
    assert "Submissions.csv" in result.not_computable_reason


def test_known_reverse_ordering_produces_expected_stability(tmp_path: Path) -> None:
    _write_fixture_meta_kaggle_files(tmp_path)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.source == "meta_kaggle"
    assert result.matched_teams == 60
    assert result.match_fraction == 1.0
    assert result.public_private_spearman == pytest.approx(-1.0)
    assert result.top10_retention == 0.0
    assert result.median_rank_change == 30.0
    assert result.not_computable_reason is None
    assert result.limitations == []


def test_actual_ranks_are_assigned_before_matching(tmp_path: Path) -> None:
    public_scores = {team_id: 101 - team_id for team_id in range(1, 101)}
    final_order = [*range(42, 101), 41]
    final_scores = {
        team_id: len(final_order) - index
        for index, team_id in enumerate(final_order)
    }
    _write_snapshots(tmp_path, public_scores, final_scores)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.match_fraction == 0.60
    assert result.matched_teams == 60
    assert result.median_rank_change == 41.0
    old_within_matched_change = pd.Series([59, *([1] * 59)]).median()
    assert old_within_matched_change == 1.0
    assert result.median_rank_change > old_within_matched_change


def test_missing_actual_public_top10_is_not_replaced_by_rank_11_plus(
    tmp_path: Path,
) -> None:
    public_scores = {team_id: 101 - team_id for team_id in range(1, 101)}
    final_scores = {team_id: 101 - team_id for team_id in range(2, 101)}
    _write_snapshots(tmp_path, public_scores, final_scores)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.match_fraction == 0.99
    assert result.top10_retention is None
    assert result.not_computable_reason is None
    assert result.limitations == [
        "Only 9 of 10 actual public top-10 teams have a matched final entry; "
        "top10_retention is unavailable."
    ]


def test_full_actual_top10_retention_ignores_reordering_below_top10(
    tmp_path: Path,
) -> None:
    public_scores = {team_id: 101 - team_id for team_id in range(1, 101)}
    final_scores = {
        team_id: (1000 - team_id if team_id <= 10 else team_id)
        for team_id in range(1, 101)
    }
    _write_snapshots(tmp_path, public_scores, final_scores)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.top10_retention == 1.0
    assert result.limitations == []


@pytest.mark.parametrize(
    ("final_count", "expected_status"),
    [(59, "not_computable"), (60, "computed"), (61, "computed")],
)
def test_match_fraction_threshold_is_inclusive(
    tmp_path: Path,
    final_count: int,
    expected_status: str,
) -> None:
    public_scores = {team_id: 101 - team_id for team_id in range(1, 101)}
    final_scores = {
        team_id: 101 - team_id for team_id in range(1, final_count + 1)
    }
    _write_snapshots(tmp_path, public_scores, final_scores)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.match_fraction == pytest.approx(final_count / 100)
    assert result.status == expected_status
    if expected_status == "not_computable":
        assert "at least 60.0%" in result.not_computable_reason
        assert _derived_metrics(result) == (None, None, None)
    else:
        assert result.not_computable_reason is None
        assert result.public_private_spearman is not None
        assert result.median_rank_change is not None


def test_match_fraction_uses_larger_snapshot_as_denominator(tmp_path: Path) -> None:
    public_scores = {team_id: 1201 - team_id for team_id in range(1, 1201)}
    final_scores = {team_id: 401 - team_id for team_id in range(1, 401)}
    _write_snapshots(tmp_path, public_scores, final_scores)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "not_computable"
    assert result.matched_teams == 400
    assert result.match_fraction == pytest.approx(400 / 1200)
    assert "33.3%" in result.not_computable_reason
    assert _derived_metrics(result) == (None, None, None)


def test_fewer_than_fifty_matched_teams_is_not_computable(tmp_path: Path) -> None:
    scores = {team_id: 50 - team_id for team_id in range(1, 50)}
    _write_snapshots(tmp_path, scores, scores)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "not_computable"
    assert result.match_fraction == 1.0
    assert result.matched_teams == 49
    assert "at least 50" in result.not_computable_reason
    assert _derived_metrics(result) == (None, None, None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
        ("true", True),
        ("True", True),
        (" false ", False),
        ("1", True),
        ("1.0", True),
        ("0", False),
        ("0.0", False),
        (None, False),
        (np.nan, False),
        ("unknown", False),
        (2, False),
    ],
)
def test_selected_mask_parses_only_supported_boolean_values(
    raw: object,
    expected: bool,
) -> None:
    assert bool(leaderboard._selected_mask(pd.Series([raw])).iloc[0]) is expected


def test_float_selected_column_keeps_rows_marked_one() -> None:
    values = pd.Series([1.0, 0.0, 1.0, np.nan], dtype="float64")

    assert leaderboard._selected_mask(values).tolist() == [True, False, True, False]


def test_model_rejects_computed_status_with_reason() -> None:
    with pytest.raises(ValidationError, match="cannot have a not_computable_reason"):
        _computed_model(not_computable_reason="low coverage")


def test_model_rejects_computed_status_below_coverage_threshold() -> None:
    with pytest.raises(ValidationError, match="at least 60.0%"):
        _computed_model(match_fraction=0.59)


def test_model_rejects_not_computable_status_with_metrics() -> None:
    with pytest.raises(ValidationError, match="cannot contain derived metrics"):
        LeaderboardStability(
            competition_id="synthetic-comp",
            status="not_computable",
            public_private_spearman=0.5,
            matched_teams=40,
            match_fraction=0.4,
            source="meta_kaggle",
            not_computable_reason="Coverage is too low.",
        )


def test_model_rejects_not_computable_status_without_reason() -> None:
    with pytest.raises(ValidationError, match="requires a reason"):
        LeaderboardStability(
            competition_id="synthetic-comp",
            status="not_computable",
            matched_teams=0,
            source="unavailable",
        )


def test_api_final_only_not_computable_has_no_derived_metrics() -> None:
    result = LeaderboardStability(
        competition_id="synthetic-comp",
        status="not_computable",
        matched_teams=0,
        source="api_final_only",
        not_computable_reason="A public leaderboard snapshot is unavailable.",
    )

    assert _derived_metrics(result) == (None, None, None)
    assert result.match_fraction is None


def test_csv_reads_select_columns_and_never_match_display_names() -> None:
    source = inspect.getsource(leaderboard)

    assert source.count("usecols=") == 3
    assert "TeamName" not in source
    assert 'snapshot["Id"] == snapshot["submission_team_id"]' in source
    assert 'merge(final_snapshot, on="team_id"' in source


def _computed_model(**updates: Any) -> LeaderboardStability:
    payload: dict[str, Any] = {
        "competition_id": "synthetic-comp",
        "status": "computed",
        "public_private_spearman": 0.9,
        "top10_retention": 0.8,
        "median_rank_change": 3.0,
        "matched_teams": 80,
        "match_fraction": 0.8,
        "source": "meta_kaggle",
    }
    payload.update(updates)
    return LeaderboardStability.model_validate(payload)


def _derived_metrics(
    result: LeaderboardStability,
) -> tuple[float | None, float | None, float | None]:
    return (
        result.public_private_spearman,
        result.top10_retention,
        result.median_rank_change,
    )


def _write_fixture_meta_kaggle_files(directory: Path) -> None:
    submissions = pd.read_csv(FIXTURE)
    team_ids = submissions["TeamId"].astype(int).tolist()
    _write_common_files(
        directory,
        teams=pd.DataFrame(
            [
                {
                    "Id": team_id,
                    "CompetitionId": 101,
                    "PublicLeaderboardSubmissionId": 1000 + team_id,
                    "PrivateLeaderboardSubmissionId": 1000 + team_id,
                }
                for team_id in team_ids
            ]
        ),
        submissions=submissions,
    )


def _write_snapshots(
    directory: Path,
    public_scores: dict[int, float],
    final_scores: dict[int, float],
) -> None:
    team_ids = sorted(set(public_scores) | set(final_scores))
    teams: list[dict[str, object]] = []
    submissions: list[dict[str, object]] = []
    for team_id in team_ids:
        public_submission_id = 100_000 + team_id
        final_submission_id = 200_000 + team_id
        teams.append(
            {
                "Id": team_id,
                "CompetitionId": 101,
                "PublicLeaderboardSubmissionId": (
                    public_submission_id if team_id in public_scores else None
                ),
                "PrivateLeaderboardSubmissionId": (
                    final_submission_id if team_id in final_scores else None
                ),
            }
        )
        if team_id in public_scores:
            submissions.append(
                {
                    "Id": public_submission_id,
                    "TeamId": team_id,
                    "IsSelected": 1.0,
                    "PublicScoreFullPrecision": public_scores[team_id],
                    "PrivateScoreFullPrecision": 0.0,
                }
            )
        if team_id in final_scores:
            submissions.append(
                {
                    "Id": final_submission_id,
                    "TeamId": team_id,
                    "IsSelected": 1.0,
                    "PublicScoreFullPrecision": 0.0,
                    "PrivateScoreFullPrecision": final_scores[team_id],
                }
            )
    _write_common_files(
        directory,
        teams=pd.DataFrame(teams),
        submissions=pd.DataFrame(submissions),
    )


def _write_common_files(
    directory: Path,
    *,
    teams: pd.DataFrame,
    submissions: pd.DataFrame,
) -> None:
    pd.DataFrame(
        [
            {
                "Id": 101,
                "Slug": "synthetic-comp",
                "EvaluationAlgorithmIsMax": True,
            }
        ]
    ).to_csv(directory / "Competitions.csv", index=False)
    teams.to_csv(directory / "Teams.csv", index=False)
    submissions.to_csv(directory / "Submissions.csv", index=False)
