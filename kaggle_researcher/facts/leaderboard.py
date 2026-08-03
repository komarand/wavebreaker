from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from kaggle_researcher.facts.models import LeaderboardStability

_MIN_MATCHED_TEAMS = 50
_COMPETITIONS_FILE = "Competitions.csv"
_TEAMS_FILE = "Teams.csv"
_SUBMISSIONS_FILE = "Submissions.csv"

_COMPETITION_COLUMNS = ["Id", "Slug", "EvaluationAlgorithmIsMax"]
_TEAM_COLUMNS = [
    "Id",
    "CompetitionId",
    "PublicLeaderboardSubmissionId",
    "PrivateLeaderboardSubmissionId",
]
_SUBMISSION_COLUMNS = [
    "Id",
    "TeamId",
    "IsSelected",
    "PublicScoreFullPrecision",
    "PrivateScoreFullPrecision",
]


def compute_stability(
    competition_slug: str,
    meta_kaggle_dir: Path | None,
) -> LeaderboardStability:
    """Compute public/private leaderboard stability from selected submissions."""
    if meta_kaggle_dir is None:
        return _not_computable(
            competition_slug,
            source="unavailable",
            reason="Meta Kaggle dumps are not configured.",
        )

    required_paths = {
        _COMPETITIONS_FILE: meta_kaggle_dir / _COMPETITIONS_FILE,
        _TEAMS_FILE: meta_kaggle_dir / _TEAMS_FILE,
        _SUBMISSIONS_FILE: meta_kaggle_dir / _SUBMISSIONS_FILE,
    }
    missing_files = [name for name, path in required_paths.items() if not path.is_file()]
    if missing_files:
        return _not_computable(
            competition_slug,
            source="unavailable",
            reason=f"Required Meta Kaggle CSVs are missing: {', '.join(missing_files)}.",
        )

    try:
        competitions = pd.read_csv(
            required_paths[_COMPETITIONS_FILE],
            usecols=_COMPETITION_COLUMNS,
            low_memory=False,
        )
        teams = pd.read_csv(
            required_paths[_TEAMS_FILE],
            usecols=_TEAM_COLUMNS,
            low_memory=False,
        )
        submissions = pd.read_csv(
            required_paths[_SUBMISSIONS_FILE],
            usecols=_SUBMISSION_COLUMNS,
            low_memory=False,
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        return _not_computable(
            competition_slug,
            source="unavailable",
            reason=f"Meta Kaggle CSVs could not be read ({type(exc).__name__}).",
        )

    competition_rows = competitions.loc[competitions["Slug"] == competition_slug]
    if competition_rows.empty:
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="Competition slug is absent from Competitions.csv.",
        )

    competition = competition_rows.iloc[0]
    higher_is_better = _parse_bool(competition["EvaluationAlgorithmIsMax"])
    if higher_is_better is None:
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="Competition score direction is unavailable.",
        )

    competition_id = competition["Id"]
    competition_teams = teams.loc[teams["CompetitionId"] == competition_id].copy()
    competition_teams = competition_teams.drop_duplicates(subset=["Id"], keep="first")
    total_teams = len(competition_teams)
    if total_teams == 0:
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="No teams were found for the competition.",
        )

    selected = submissions.loc[_selected_mask(submissions["IsSelected"])].copy()
    selected = selected.drop_duplicates(subset=["Id"], keep="first")
    matched = _match_selected_scores(competition_teams, selected)
    matched_teams = len(matched)
    excluded_teams = total_teams - matched_teams
    excluded_reason = _excluded_reason(excluded_teams, total_teams)

    if matched_teams < _MIN_MATCHED_TEAMS:
        reason = (
            f"Only {matched_teams} matched teams have selected final submissions with "
            f"both public and private scores; at least {_MIN_MATCHED_TEAMS} are required."
        )
        if excluded_reason:
            reason = f"{reason} {excluded_reason}"
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason=reason,
            matched_teams=matched_teams,
        )

    public_ranks = matched["public_score"].rank(
        method="average",
        ascending=not higher_is_better,
    )
    private_ranks = matched["private_score"].rank(
        method="average",
        ascending=not higher_is_better,
    )
    spearman = public_ranks.corr(private_ranks, method="pearson")
    if pd.isna(spearman):
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="Public/private Spearman correlation is undefined for the matched scores.",
            matched_teams=matched_teams,
        )

    public_top10 = _top_team_ids(matched, "public_score", higher_is_better)
    private_top10 = _top_team_ids(matched, "private_score", higher_is_better)
    top10_retention = len(public_top10 & private_top10) / len(public_top10)
    median_rank_change = float((public_ranks - private_ranks).abs().median())

    return LeaderboardStability(
        competition_id=competition_slug,
        status="computed",
        public_private_spearman=float(spearman),
        top10_retention=float(top10_retention),
        median_rank_change=median_rank_change,
        matched_teams=matched_teams,
        source="meta_kaggle",
        not_computable_reason=excluded_reason,
    )


def _match_selected_scores(
    teams: pd.DataFrame,
    selected_submissions: pd.DataFrame,
) -> pd.DataFrame:
    public = selected_submissions[
        ["Id", "TeamId", "PublicScoreFullPrecision"]
    ].rename(
        columns={
            "Id": "PublicLeaderboardSubmissionId",
            "TeamId": "public_team_id",
            "PublicScoreFullPrecision": "public_score",
        }
    )
    private = selected_submissions[
        ["Id", "TeamId", "PrivateScoreFullPrecision"]
    ].rename(
        columns={
            "Id": "PrivateLeaderboardSubmissionId",
            "TeamId": "private_team_id",
            "PrivateScoreFullPrecision": "private_score",
        }
    )
    matched = teams.merge(public, on="PublicLeaderboardSubmissionId", how="inner")
    matched = matched.merge(private, on="PrivateLeaderboardSubmissionId", how="inner")
    matched = matched.loc[
        (matched["Id"] == matched["public_team_id"])
        & (matched["Id"] == matched["private_team_id"])
    ].copy()
    matched["public_score"] = pd.to_numeric(matched["public_score"], errors="coerce")
    matched["private_score"] = pd.to_numeric(matched["private_score"], errors="coerce")
    matched = matched.dropna(subset=["public_score", "private_score"])
    return matched.loc[
        np.isfinite(matched["public_score"])
        & np.isfinite(matched["private_score"])
    ].reset_index(drop=True)


def _top_team_ids(
    matched: pd.DataFrame,
    score_column: str,
    higher_is_better: bool,
) -> set[object]:
    ordered = matched.sort_values(
        by=[score_column, "Id"],
        ascending=[not higher_is_better, True],
        kind="stable",
    )
    return set(ordered.head(10)["Id"])


def _selected_mask(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin({"true", "1"})


def _parse_bool(value: object) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _excluded_reason(excluded_teams: int, total_teams: int) -> str | None:
    if excluded_teams > total_teams / 2:
        return (
            f"Excluded {excluded_teams} of {total_teams} teams without matched selected "
            "public/private submissions."
        )
    return None


def _not_computable(
    competition_slug: str,
    *,
    source: Literal["meta_kaggle", "unavailable"],
    reason: str,
    matched_teams: int = 0,
) -> LeaderboardStability:
    return LeaderboardStability(
        competition_id=competition_slug,
        status="not_computable",
        matched_teams=matched_teams,
        source=source,
        not_computable_reason=reason,
    )
