from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from kaggle_researcher.facts import leaderboard
from kaggle_researcher.facts.leaderboard import compute_stability

FIXTURE = Path("tests/fixtures/facts/meta_kaggle_submissions.csv")


def test_missing_meta_kaggle_directory_is_expected_not_computable() -> None:
    result = compute_stability("synthetic-comp", None)

    assert result.status == "not_computable"
    assert result.source == "unavailable"
    assert result.matched_teams == 0
    assert result.not_computable_reason == "Meta Kaggle dumps are not configured."


def test_missing_required_csvs_are_expected_not_computable(tmp_path: Path) -> None:
    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "not_computable"
    assert result.source == "unavailable"
    assert "Competitions.csv" in result.not_computable_reason
    assert "Teams.csv" in result.not_computable_reason
    assert "Submissions.csv" in result.not_computable_reason


def test_known_reverse_ordering_produces_expected_stability(tmp_path: Path) -> None:
    _write_meta_kaggle_files(tmp_path)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.source == "meta_kaggle"
    assert result.matched_teams == 60
    assert result.public_private_spearman == pytest.approx(-1.0)
    assert result.top10_retention == 0.0
    assert result.median_rank_change == 30.0
    assert result.not_computable_reason is None


def test_fewer_than_fifty_matched_teams_is_not_computable(tmp_path: Path) -> None:
    submissions = pd.read_csv(FIXTURE).head(49)
    _write_meta_kaggle_files(tmp_path, submissions=submissions)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "not_computable"
    assert result.source == "meta_kaggle"
    assert result.matched_teams == 49
    assert "at least 50" in result.not_computable_reason


def test_excluded_majority_is_recorded_even_when_metrics_are_computed(
    tmp_path: Path,
) -> None:
    _write_meta_kaggle_files(tmp_path, total_teams=121)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.matched_teams == 60
    assert result.not_computable_reason == (
        "Excluded 61 of 121 teams without matched selected public/private submissions."
    )


def test_unselected_and_cross_team_submissions_are_excluded(tmp_path: Path) -> None:
    submissions = pd.read_csv(FIXTURE)
    submissions.loc[submissions["TeamId"] == 1, "IsSelected"] = False
    submissions.loc[submissions["TeamId"] == 2, "TeamId"] = 999
    _write_meta_kaggle_files(tmp_path, submissions=submissions)

    result = compute_stability("synthetic-comp", tmp_path)

    assert result.status == "computed"
    assert result.matched_teams == 58


def test_csv_reads_select_columns_and_never_match_display_names() -> None:
    source = inspect.getsource(leaderboard)

    assert source.count("usecols=") == 3
    assert "TeamName" not in source
    assert 'matched["Id"] == matched["public_team_id"]' in source
    assert 'matched["Id"] == matched["private_team_id"]' in source


def _write_meta_kaggle_files(
    directory: Path,
    *,
    submissions: pd.DataFrame | None = None,
    total_teams: int = 60,
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
    pd.DataFrame(
        [
            {
                "Id": team_id,
                "CompetitionId": 101,
                "PublicLeaderboardSubmissionId": 1000 + team_id,
                "PrivateLeaderboardSubmissionId": 1000 + team_id,
            }
            for team_id in range(1, total_teams + 1)
        ]
    ).to_csv(directory / "Teams.csv", index=False)
    fixture = submissions if submissions is not None else pd.read_csv(FIXTURE)
    fixture.to_csv(directory / "Submissions.csv", index=False)
