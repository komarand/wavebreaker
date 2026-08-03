from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from kaggle_researcher.facts.models import (
    MIN_LEADERBOARD_MATCH_FRACTION,
    LeaderboardStability,
)

_MIN_MATCHED_TEAMS = 50
MIN_MATCH_FRACTION = MIN_LEADERBOARD_MATCH_FRACTION
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
    if competition_teams.empty:
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="No teams were found for the competition.",
        )

    selected = submissions.loc[_selected_mask(submissions["IsSelected"])].copy()
    selected = selected.drop_duplicates(subset=["Id"], keep="first")
    public_snapshot = _build_snapshot(
        competition_teams,
        selected,
        selection_column="PublicLeaderboardSubmissionId",
        score_column="PublicScoreFullPrecision",
        score_name="public_score",
        rank_name="public_rank_actual",
        higher_is_better=higher_is_better,
    )
    final_snapshot = _build_snapshot(
        competition_teams,
        selected,
        selection_column="PrivateLeaderboardSubmissionId",
        score_column="PrivateScoreFullPrecision",
        score_name="final_score",
        rank_name="final_rank_actual",
        higher_is_better=higher_is_better,
    )
    valid_public_teams = len(public_snapshot)
    valid_final_teams = len(final_snapshot)
    coverage_denominator = max(valid_public_teams, valid_final_teams)
    if coverage_denominator == 0:
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="No valid selected public or final leaderboard rows were found.",
            match_fraction=0.0,
        )

    # Divide by the larger snapshot so a complete match to a much smaller final
    # leaderboard cannot be reported as 100% coverage.
    matched = public_snapshot.merge(final_snapshot, on="team_id", how="inner")
    matched_teams = len(matched)
    match_fraction = matched_teams / coverage_denominator

    gate_reasons: list[str] = []
    if match_fraction < MIN_MATCH_FRACTION:
        gate_reasons.append(
            f"Only {match_fraction:.1%} of valid leaderboard teams could be matched; "
            f"at least {MIN_MATCH_FRACTION:.1%} is required."
        )
    if matched_teams < _MIN_MATCHED_TEAMS:
        gate_reasons.append(
            f"Only {matched_teams} matched teams have selected final submissions with "
            f"both public and private scores; at least {_MIN_MATCHED_TEAMS} are required."
        )
    if gate_reasons:
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason=" ".join(gate_reasons),
            matched_teams=matched_teams,
            match_fraction=match_fraction,
        )

    spearman = matched["public_rank_actual"].corr(
        matched["final_rank_actual"],
        method="pearson",
    )
    if pd.isna(spearman):
        return _not_computable(
            competition_slug,
            source="meta_kaggle",
            reason="Public/private Spearman correlation is undefined for the matched scores.",
            matched_teams=matched_teams,
            match_fraction=match_fraction,
        )

    public_top10 = _top_team_ids(public_snapshot, "public_score", higher_is_better)
    final_top10 = _top_team_ids(final_snapshot, "final_score", higher_is_better)
    matched_team_ids = set(matched["team_id"])
    public_top10_matched_count = len(public_top10 & matched_team_ids)
    limitations: list[str] = []
    if public_top10_matched_count < len(public_top10):
        top10_retention = None
        limitations.append(
            f"Only {public_top10_matched_count} of {len(public_top10)} actual public "
            "top-10 teams have a matched final entry; top10_retention is unavailable."
        )
    else:
        top10_retention = len(public_top10 & final_top10) / len(public_top10)
    median_rank_change = float(
        (matched["public_rank_actual"] - matched["final_rank_actual"]).abs().median()
    )

    return LeaderboardStability(
        competition_id=competition_slug,
        status="computed",
        public_private_spearman=float(spearman),
        top10_retention=top10_retention,
        median_rank_change=median_rank_change,
        matched_teams=matched_teams,
        match_fraction=match_fraction,
        source="meta_kaggle",
        limitations=limitations,
    )


def _build_snapshot(
    teams: pd.DataFrame,
    selected_submissions: pd.DataFrame,
    *,
    selection_column: str,
    score_column: str,
    score_name: str,
    rank_name: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    selected_scores = selected_submissions[["Id", "TeamId", score_column]].rename(
        columns={
            "Id": selection_column,
            "TeamId": "submission_team_id",
            score_column: score_name,
        }
    )
    snapshot = teams[["Id", selection_column]].merge(
        selected_scores,
        on=selection_column,
        how="inner",
    )
    snapshot = snapshot.loc[snapshot["Id"] == snapshot["submission_team_id"]].copy()
    snapshot[score_name] = pd.to_numeric(snapshot[score_name], errors="coerce")
    snapshot = snapshot.dropna(subset=[score_name])
    snapshot = snapshot.loc[np.isfinite(snapshot[score_name])].copy()
    snapshot = snapshot.rename(columns={"Id": "team_id"})
    snapshot = snapshot.drop_duplicates(subset=["team_id"], keep="first")
    # Average ranks preserve standard tie semantics; the ranks are assigned on the
    # complete valid snapshot before public/final teams are matched.
    snapshot[rank_name] = snapshot[score_name].rank(
        method="average",
        ascending=not higher_is_better,
    )
    return snapshot[["team_id", score_name, rank_name]].reset_index(drop=True)


def _top_team_ids(
    snapshot: pd.DataFrame,
    score_column: str,
    higher_is_better: bool,
) -> set[object]:
    # A score tie at the top-k boundary is resolved by the stable team id.
    ordered = snapshot.sort_values(
        by=[score_column, "team_id"],
        ascending=[not higher_is_better, True],
        kind="stable",
    )
    return set(ordered.head(10)["team_id"])


def _selected_mask(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(normalized, errors="coerce")
    return (numeric.eq(1) | normalized.eq("true")).fillna(False).astype(bool)


def _parse_bool(value: object) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    return None


def _not_computable(
    competition_slug: str,
    *,
    source: Literal["meta_kaggle", "unavailable"],
    reason: str,
    matched_teams: int = 0,
    match_fraction: float | None = None,
) -> LeaderboardStability:
    return LeaderboardStability(
        competition_id=competition_slug,
        status="not_computable",
        matched_teams=matched_teams,
        match_fraction=match_fraction,
        source=source,
        not_computable_reason=reason,
    )
