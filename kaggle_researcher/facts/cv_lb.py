from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Literal

from kaggle_researcher.facts.models import (
    CvLbDiagnostics,
    CvLbPair,
    LeaderboardMatch,
    MetricCanonicalSource,
    NotebookFacts,
    OptimizationDirection,
    PublicLeaderboard,
    ScoreObservation,
)
from kaggle_researcher.facts.notebook_ast import (
    METRIC_VALUE_RANGE,
    canonicalize_metric_label,
    metric_optimization_direction,
)

MAX_PLAUSIBLE_BOUNDED_GAP = 0.05
_BOUNDED_METRICS = frozenset({"accuracy", "f1", "mAP", "rank-1", "roc_auc"})
_METRIC_RANGE_KEYS = {
    "mAP": "map",
    "rank-1": "top1",
    "roc_auc": "auc",
}
_METRIC_CONTEXT_TOKENS = frozenset(
    {
        "cv",
        "fold",
        "leaderboard",
        "lb",
        "local",
        "metric",
        "offline",
        "online",
        "oof",
        "private",
        "public",
        "score",
        "submission",
        "val",
        "valid",
        "validation",
    }
)


@dataclass(frozen=True)
class _Representative:
    value: float
    metric_canonical: str | None
    metric_canonical_source: MetricCanonicalSource
    metric_raw: str | None
    direction: OptimizationDirection | None
    observation_ids: tuple[str, ...]
    representative_observation_id: str | None
    source: str
    aggregation: str
    selection_reason: str
    values: tuple[float, ...]
    raw_values: tuple[str, ...]


@dataclass(frozen=True)
class _PairingResult:
    pairs: tuple[CvLbPair, ...]
    implausible_gap_pairs: tuple[CvLbPair, ...]
    rejections: frozenset[str]


class CvLbPairList(list[CvLbPair]):
    def __init__(
        self,
        pairs: list[CvLbPair],
        *,
        implausible_gap_pairs: list[CvLbPair],
    ) -> None:
        super().__init__(pairs)
        self.implausible_gap_pairs = implausible_gap_pairs


def build_cv_lb_pairs(
    notebooks: list[NotebookFacts],
    competition_metric_name: str | None = None,
) -> list[CvLbPair]:
    results = [
        _pair_notebook(notebook, competition_metric_name) for notebook in notebooks
    ]
    return CvLbPairList(
        _sorted_pairs([pair for result in results for pair in result.pairs]),
        implausible_gap_pairs=_sorted_pairs(
            [
                pair
                for result in results
                for pair in result.implausible_gap_pairs
            ]
        ),
    )


def match_leaderboard_scores(
    notebooks: list[NotebookFacts],
    leaderboard: PublicLeaderboard,
) -> dict[str, LeaderboardMatch]:
    if leaderboard.status != "collected":
        return {}

    author_notebooks: dict[str, list[NotebookFacts]] = {}
    for notebook in notebooks:
        author = _normalized_identity(notebook.author or notebook.ref.partition("/")[0])
        if author:
            author_notebooks.setdefault(author, []).append(notebook)

    teams: dict[str, tuple[str, float, int | None, int]] = {}
    for index, entry in enumerate(leaderboard.entries):
        team = _normalized_identity(entry.team_name)
        score = _finite_float(entry.score)
        if not team or entry.team_name is None or score is None:
            continue
        candidate = (entry.team_name, score, entry.rank, index)
        previous = teams.get(team)
        if previous is None or _team_entry_order(candidate) < _team_entry_order(previous):
            teams[team] = candidate

    author_candidates: dict[str, list[tuple[str, Literal["exact", "partial"]]]] = {}
    for author in author_notebooks:
        exact = [(team, "exact") for team in teams if team == author]
        candidates = exact or [
            (team, "partial")
            for team in teams
            if author in team or team in author
        ]
        if candidates:
            author_candidates[author] = candidates

    team_authors: dict[str, set[str]] = {}
    for author, candidates in author_candidates.items():
        for team, _confidence in candidates:
            team_authors.setdefault(team, set()).add(author)

    matches: dict[str, LeaderboardMatch] = {}
    for author, candidates in sorted(author_candidates.items()):
        unique_teams = {team for team, _confidence in candidates}
        if len(unique_teams) != 1:
            continue
        team, confidence = candidates[0]
        if len(team_authors.get(team, set())) != 1:
            continue
        team_name, score, _rank, _index = teams[team]
        for notebook in sorted(author_notebooks[author], key=lambda item: item.ref):
            matches[notebook.ref] = LeaderboardMatch(
                notebook_ref=notebook.ref,
                team_name=team_name,
                score=score,
                match_confidence=confidence,
            )
    return matches


def build_leaderboard_cv_lb_pairs(
    notebooks: list[NotebookFacts],
    matches: dict[str, LeaderboardMatch],
    competition_metric_name: str | None = None,
) -> list[CvLbPair]:
    pairs: list[CvLbPair] = []
    implausible_gap_pairs: list[CvLbPair] = []
    for notebook in notebooks:
        match = matches.get(notebook.ref)
        if match is None:
            continue
        matched_notebook = notebook.model_copy(update={"public_score": match.score})
        result = _pair_notebook(matched_notebook, competition_metric_name)
        pairs.extend(
            _leaderboard_pair(pair, match) for pair in result.pairs
        )
        implausible_gap_pairs.extend(
            _leaderboard_pair(pair, match)
            for pair in result.implausible_gap_pairs
        )
    return CvLbPairList(
        _sorted_pairs(pairs),
        implausible_gap_pairs=_sorted_pairs(implausible_gap_pairs),
    )


def _leaderboard_pair(pair: CvLbPair, match: LeaderboardMatch) -> CvLbPair:
    return pair.model_copy(
        update={
            "lb_source": "leaderboard_match",
            "lb_observation_ids": [],
            "lb_representative_observation_id": None,
            "lb_aggregation": "team_best_submission",
            "lb_selection_reason": (
                f"leaderboard_author_{match.match_confidence}_match"
            ),
        }
    )


def _sorted_pairs(pairs: list[CvLbPair]) -> list[CvLbPair]:
    return sorted(
        pairs,
        key=lambda pair: (
            pair.notebook_ref,
            pair.metric_canonical or pair.metric_raw or "",
            pair.cv_score if pair.cv_score is not None else pair.declared_cv,
            pair.lb_score if pair.lb_score is not None else pair.public_score,
        ),
    )


def diagnose_cv_lb(
    notebooks: list[NotebookFacts],
    pairs: list[CvLbPair],
    competition_metric_name: str | None = None,
    leaderboard_matches: dict[str, LeaderboardMatch] | None = None,
) -> CvLbDiagnostics:
    observation_pairs = [pair for pair in pairs if pair.lb_source == "observation"]
    leaderboard_pairs = [pair for pair in pairs if pair.lb_source == "leaderboard_match"]
    results = [_pair_notebook(notebook, competition_metric_name) for notebook in notebooks]
    with_public_score = sum(notebook.public_score is not None for notebook in notebooks)
    with_declared_cv = sum(bool(notebook.declared_cv) for notebook in notebooks)
    with_cv = sum(_has_cv_side(notebook) for notebook in notebooks)
    with_lb = sum(_has_lb_side(notebook) for notebook in notebooks)
    with_both = sum(_has_cv_side(notebook) and _has_lb_side(notebook) for notebook in notebooks)
    rejection_counts = {
        reason: sum(reason in result.rejections for result in results)
        for reason in (
            "missing_cv",
            "missing_lb",
            "metric_mismatch",
            "scale_mismatch",
            "implausible_gap",
            "ambiguous_metric",
            "ambiguous_split",
        )
    }
    rejection_counts["implausible_gap"] = sum(
        len(result.implausible_gap_pairs) for result in results
    )
    leaderboard_results = [
        _pair_notebook(
            notebook.model_copy(update={"public_score": match.score}),
            competition_metric_name,
        )
        for notebook in notebooks
        if (match := (leaderboard_matches or {}).get(notebook.ref)) is not None
    ]
    leaderboard_implausible_gap_rejections = sum(
        len(result.implausible_gap_pairs) for result in leaderboard_results
    )
    rejected_implausible_gap = (
        rejection_counts["implausible_gap"]
        + leaderboard_implausible_gap_rejections
    )
    zero_pairs_reason = (
        _zero_pairs_reason(notebooks, rejection_counts) if not observation_pairs else None
    )
    return CvLbDiagnostics(
        notebooks_total=len(notebooks),
        notebooks_with_public_score=with_public_score,
        notebooks_with_declared_cv=with_declared_cv,
        notebooks_with_both=sum(
            notebook.public_score is not None and bool(notebook.declared_cv)
            for notebook in notebooks
        ),
        comparable_pairs=len(observation_pairs),
        rejected_non_comparable_pairs=(
            rejection_counts["metric_mismatch"]
            + rejection_counts["scale_mismatch"]
            + rejection_counts["implausible_gap"]
            + rejection_counts["ambiguous_metric"]
        ),
        zero_pairs_reason=zero_pairs_reason,
        notebooks_with_cv_scores=with_cv,
        notebooks_with_lb_scores=with_lb,
        notebooks_with_both_sides=with_both,
        pairs_created=len(observation_pairs),
        pairs_created_from_api_lb=sum(
            pair.lb_source == "observation"
            and pair.lb_selection_reason == "api_public_score_priority"
            for pair in observation_pairs
        ),
        pairs_created_from_observation_lb=sum(
            pair.lb_source == "observation"
            and pair.lb_selection_reason != "api_public_score_priority"
            for pair in observation_pairs
        ),
        pairs_created_from_leaderboard_match=len(leaderboard_pairs),
        pairs_rejected_missing_cv=rejection_counts["missing_cv"],
        pairs_rejected_missing_lb=rejection_counts["missing_lb"],
        pairs_rejected_metric_mismatch=rejection_counts["metric_mismatch"],
        pairs_rejected_scale_mismatch=rejection_counts["scale_mismatch"],
        rejected_implausible_gap=rejected_implausible_gap,
        pairs_rejected_implausible_gap=rejection_counts["implausible_gap"],
        leaderboard_pairs_rejected_implausible_gap=(
            leaderboard_implausible_gap_rejections
        ),
        pairs_rejected_ambiguous_metric=rejection_counts["ambiguous_metric"],
        pairs_rejected_ambiguous_split=rejection_counts["ambiguous_split"],
        fold_series_aggregated=sum(
            pair.cv_aggregation == "median_fold_series" for pair in observation_pairs
        ),
        unknown_direction_fallbacks=sum(
            reason == "unknown_direction_max_fallback"
            for pair in observation_pairs
            for reason in (pair.cv_selection_reason, pair.lb_selection_reason)
        ),
    )


def summarize_cv_lb(
    pairs: list[CvLbPair],
) -> dict[str, int | float | str | None]:
    observation_pairs = [pair for pair in pairs if pair.lb_source == "observation"]
    leaderboard_pairs = [pair for pair in pairs if pair.lb_source == "leaderboard_match"]
    count = len(observation_pairs)
    gaps = [pair.declared_cv - pair.public_score for pair in observation_pairs]
    leaderboard_gaps = [
        pair.declared_cv - pair.public_score for pair in leaderboard_pairs
    ]
    reliability, note = _cv_lb_reliability(count)
    return {
        "count": count,
        "mean_gap": statistics.fmean(gaps) if gaps else None,
        "median_gap": statistics.median(gaps) if gaps else None,
        "spearman": (
            _spearman_correlation(
                [pair.declared_cv for pair in observation_pairs],
                [pair.public_score for pair in observation_pairs],
            )
            if count >= 3
            else None
        ),
        "distinct_lineage_clusters": len(
            {pair.lineage_cluster_id for pair in observation_pairs}
        ),
        "reliability": reliability,
        "note": note,
        "leaderboard_pair_count": len(leaderboard_pairs),
        "leaderboard_median_gap": (
            statistics.median(leaderboard_gaps) if leaderboard_gaps else None
        ),
        "leaderboard_gap_note": (
            "Leaderboard score is the team's best submission, not the score of the "
            "matched notebook."
            if leaderboard_pairs
            else None
        ),
    }


def _cv_lb_reliability(count: int) -> tuple[str, str | None]:
    if count == 0:
        return "insufficient", "no pairs; gap cannot be estimated"
    if count == 1:
        return (
            "insufficient",
            "single pair; gap is not evidence of a systematic pattern",
        )
    if count < 5:
        return (
            "insufficient",
            "fewer than 5 pairs; gap is not evidence of a systematic pattern",
        )
    if count < 15:
        return "weak", "fewer than 15 pairs; aggregate gap evidence is weak"
    return "sufficient", None


def _pair_notebook(
    notebook: NotebookFacts,
    competition_metric_name: str | None,
) -> _PairingResult:
    plausible_observations = _plausible(notebook.score_observations)
    cv_observations = _deduplicate_observations(
        observation
        for observation in plausible_observations
        if observation.split == "cv"
        and _finite_float(observation.value) is not None
    )
    lb_observations = _deduplicate_observations(
        observation
        for observation in plausible_observations
        if observation.split == "lb"
        and _finite_float(observation.value) is not None
    )
    unknown_observations = [
        observation
        for observation in plausible_observations
        if observation.split == "unknown"
    ]
    cv_groups = _observation_groups(
        cv_observations,
        side="cv",
        competition_metric_name=competition_metric_name,
    )
    if not cv_groups:
        legacy_cv = _legacy_cv_representative(notebook, competition_metric_name)
        if legacy_cv is not None:
            cv_groups = {_representative_key(legacy_cv): legacy_cv}

    public_score = _finite_float(notebook.public_score)
    if public_score is not None:
        lb_groups = {
            _metric_key(competition_metric_name, competition_metric_name): (
                _api_lb_representative(public_score, competition_metric_name)
            )
        }
    else:
        lb_groups = _observation_groups(
            lb_observations,
            side="lb",
            competition_metric_name=competition_metric_name,
        )

    rejections: set[str] = set()
    if not cv_groups:
        rejections.add("missing_cv")
    if not lb_groups:
        rejections.add("missing_lb")
    if unknown_observations and (not cv_groups or not lb_groups):
        rejections.add("ambiguous_split")
    if not cv_groups or not lb_groups:
        return _PairingResult((), (), frozenset(rejections))

    matches, match_rejection = _match_groups(
        cv_groups,
        lb_groups,
    )
    if match_rejection is not None:
        rejections.add(match_rejection)

    pairs: list[CvLbPair] = []
    implausible_gap_pairs: list[CvLbPair] = []
    for cv_side, lb_side, metric_match in matches:
        metric_canonical = (
            cv_side.metric_canonical
            or lb_side.metric_canonical
            or canonicalize_metric_label(competition_metric_name)
        )
        if not _scales_are_compatible(cv_side, lb_side, metric_canonical):
            rejections.add("scale_mismatch")
            continue
        direction = _resolve_direction(
            cv_side.direction,
            lb_side.direction,
            metric_canonical or competition_metric_name,
        )
        gap = cv_side.value - lb_side.value
        pair = CvLbPair(
            notebook_ref=notebook.ref,
            declared_cv=cv_side.value,
            public_score=lb_side.value,
            lineage_cluster_id=notebook.lineage_cluster_id,
            metric_name=metric_canonical or cv_side.metric_raw or lb_side.metric_raw,
            metric_raw=cv_side.metric_raw or lb_side.metric_raw,
            metric_canonical=metric_canonical,
            metric_match=metric_match,
            optimization_direction=_cv_lb_optimization_direction(direction),
            cv_score=cv_side.value,
            lb_score=lb_side.value,
            cv_observation_ids=list(cv_side.observation_ids),
            lb_observation_ids=list(lb_side.observation_ids),
            cv_representative_observation_id=cv_side.representative_observation_id,
            lb_representative_observation_id=lb_side.representative_observation_id,
            cv_source=cv_side.source,
            lb_source="observation",
            cv_aggregation=cv_side.aggregation,
            lb_aggregation=lb_side.aggregation,
            cv_selection_reason=cv_side.selection_reason,
            lb_selection_reason=lb_side.selection_reason,
            gap=gap,
            absolute_gap=abs(gap),
        )
        if not _gap_is_plausible(gap, metric_canonical):
            rejections.add("implausible_gap")
            implausible_gap_pairs.append(
                pair.model_copy(
                    update={
                        "comparability_status": "implausible_gap",
                        "comparability_reason": _implausible_gap_reason(
                            gap,
                            metric_canonical,
                        ),
                    }
                )
            )
            continue
        pairs.append(pair)
    return _PairingResult(
        tuple(pairs),
        tuple(implausible_gap_pairs),
        frozenset(rejections),
    )


def _observation_groups(
    observations: list[ScoreObservation],
    *,
    side: Literal["cv", "lb"],
    competition_metric_name: str | None,
) -> dict[tuple[str, str] | None, _Representative]:
    grouped: dict[tuple[str, str] | None, list[ScoreObservation]] = {}
    for observation in observations:
        grouped.setdefault(_observation_metric_key(observation), []).append(observation)
    return {
        key: _represent_observations(
            group,
            side=side,
            fallback_metric_name=(competition_metric_name if key is None else None),
        )
        for key, group in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def _represent_observations(
    observations: list[ScoreObservation],
    *,
    side: Literal["cv", "lb"],
    fallback_metric_name: str | None,
) -> _Representative:
    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.observation_id or "",
            observation.value,
            observation.value_raw,
            observation.raw_text,
        ),
    )
    values = [float(observation.value) for observation in ordered]
    directions = {
        observation.optimization_direction
        for observation in ordered
        if observation.optimization_direction is not None
    }
    metric_canonical = next(
        iter(
            sorted(
                {
                    observation.metric_canonical
                    for observation in ordered
                    if observation.metric_canonical
                }
            )
        ),
        None,
    )
    canonical_sources = {
        observation.metric_canonical_source
        for observation in ordered
        if observation.metric_canonical is not None
    }
    if "alias" in canonical_sources:
        metric_canonical_source: MetricCanonicalSource = "alias"
    elif canonical_sources == {"competition_hint"}:
        metric_canonical_source = "competition_hint"
    else:
        metric_canonical_source = "none"
    metric_raw = next(
        iter(sorted({observation.metric_raw for observation in ordered if observation.metric_raw})),
        None,
    )
    direction = (
        next(iter(directions))
        if len(directions) == 1
        else metric_optimization_direction(metric_canonical or metric_raw or fallback_metric_name)
    )
    fold_observations = [
        observation for observation in ordered if "fold" in observation.split_signals
    ]
    if side == "cv" and len(fold_observations) >= 2 and len(fold_observations) == len(ordered):
        selected_value = statistics.median(values)
        aggregation = "median_fold_series"
        selection_reason = "fold_series_median"
        representative_id = None
    else:
        selector = min if direction == "minimize" else max
        selected_value = selector(values)
        aggregation = "min" if direction == "minimize" else "max"
        selection_reason = (
            f"best_by_{direction}" if direction is not None else "unknown_direction_max_fallback"
        )
        selected = [observation for observation in ordered if observation.value == selected_value][
            0
        ]
        representative_id = selected.observation_id
    representative = next(
        (observation for observation in ordered if observation.observation_id == representative_id),
        ordered[0],
    )
    source = "score_observation" if side == "cv" else _lb_observation_source(representative)
    return _Representative(
        value=selected_value,
        metric_canonical=metric_canonical,
        metric_canonical_source=metric_canonical_source,
        metric_raw=metric_raw,
        direction=direction,
        observation_ids=tuple(
            sorted(
                observation.observation_id
                for observation in ordered
                if observation.observation_id is not None
            )
        ),
        representative_observation_id=representative_id,
        source=source,
        aggregation=aggregation,
        selection_reason=selection_reason,
        values=tuple(values),
        raw_values=tuple(observation.value_raw for observation in ordered),
    )


def _legacy_cv_representative(
    notebook: NotebookFacts,
    competition_metric_name: str | None,
) -> _Representative | None:
    value = _select_declared_cv(notebook.declared_cv)
    if value is None:
        return None
    metric = _notebook_cv_metric(notebook)
    numeric_values = tuple(
        number
        for raw_value in notebook.declared_cv
        if (number := _finite_float(raw_value)) is not None
    )
    return _Representative(
        value=value,
        metric_canonical=canonicalize_metric_label(metric),
        metric_canonical_source="none",
        metric_raw=metric,
        direction=metric_optimization_direction(metric or competition_metric_name),
        observation_ids=(),
        representative_observation_id=None,
        source="declared_cv_legacy",
        aggregation="median_legacy" if len(numeric_values) > 3 else "first_legacy",
        selection_reason="legacy_declared_cv_fallback",
        values=numeric_values,
        raw_values=tuple(str(value) for value in notebook.declared_cv),
    )


def _api_lb_representative(
    public_score: float,
    competition_metric_name: str | None,
) -> _Representative:
    return _Representative(
        value=public_score,
        metric_canonical=canonicalize_metric_label(competition_metric_name),
        metric_canonical_source="none",
        metric_raw=competition_metric_name,
        direction=metric_optimization_direction(competition_metric_name),
        observation_ids=(),
        representative_observation_id=None,
        source="public_score_api",
        aggregation="single",
        selection_reason="api_public_score_priority",
        values=(public_score,),
        raw_values=(str(public_score),),
    )


def _match_groups(
    cv_groups: dict[tuple[str, str] | None, _Representative],
    lb_groups: dict[tuple[str, str] | None, _Representative],
) -> tuple[
    list[tuple[_Representative, _Representative, Literal["exact", "assumed"]]],
    str | None,
]:
    exact_keys = sorted(
        (set(cv_groups) & set(lb_groups)) - {None},
        key=str,
    )
    if exact_keys:
        return [
            (
                cv_groups[key],
                lb_groups[key],
                _metric_match_status(cv_groups[key], lb_groups[key]),
            )
            for key in exact_keys
        ], None

    if len(cv_groups) == 1 and len(lb_groups) == 1:
        cv_side = next(iter(cv_groups.values()))
        lb_side = next(iter(lb_groups.values()))
        if cv_side.metric_canonical is None or lb_side.metric_canonical is None:
            return [(cv_side, lb_side, "assumed")], None
        if cv_side.metric_canonical == lb_side.metric_canonical:
            return [(cv_side, lb_side, _metric_match_status(cv_side, lb_side))], None
        return [], "metric_mismatch"
    if len(cv_groups) > 1 or len(lb_groups) > 1:
        return [], "ambiguous_metric"
    return [], "metric_mismatch"


def _metric_match_status(
    cv_side: _Representative,
    lb_side: _Representative,
) -> Literal["exact", "assumed"]:
    if (
        cv_side.metric_canonical_source == "competition_hint"
        and lb_side.metric_canonical_source == "competition_hint"
    ):
        return "assumed"
    return "exact"


def _scales_are_compatible(
    cv_side: _Representative,
    lb_side: _Representative,
    metric_canonical: str | None,
) -> bool:
    if metric_canonical not in _BOUNDED_METRICS:
        return True
    cv_large = any(abs(value) > 1 for value in cv_side.values)
    lb_large = any(abs(value) > 1 for value in lb_side.values)
    return cv_large == lb_large


def _gap_is_plausible(gap: float, metric_canonical: str | None) -> bool:
    value_range = _bounded_metric_range(metric_canonical)
    return value_range is None or abs(gap) <= MAX_PLAUSIBLE_BOUNDED_GAP


def _bounded_metric_range(
    metric_canonical: str | None,
) -> tuple[float, float] | None:
    range_key = _METRIC_RANGE_KEYS.get(metric_canonical or "", metric_canonical)
    value_range = METRIC_VALUE_RANGE.get(range_key or "")
    if value_range is None:
        return None
    lower, upper = value_range
    if not math.isfinite(lower) or not math.isfinite(upper) or upper - lower > 2.0:
        return None
    return value_range


def _implausible_gap_reason(
    gap: float,
    metric_canonical: str | None,
) -> str:
    return (
        f"absolute gap {abs(gap):.6g} exceeds "
        f"{MAX_PLAUSIBLE_BOUNDED_GAP:.6g} for bounded metric "
        f"{metric_canonical or 'unknown'}"
    )


def _resolve_direction(
    cv_direction: OptimizationDirection | None,
    lb_direction: OptimizationDirection | None,
    metric_name: str | None,
) -> OptimizationDirection | None:
    directions = {direction for direction in (cv_direction, lb_direction) if direction is not None}
    if len(directions) == 1:
        return next(iter(directions))
    if len(directions) > 1:
        return None
    return metric_optimization_direction(metric_name)


def _cv_lb_optimization_direction(
    direction: OptimizationDirection | None,
) -> Literal["higher_is_better", "lower_is_better"] | None:
    if direction == "maximize":
        return "higher_is_better"
    if direction == "minimize":
        return "lower_is_better"
    return None


def _observation_metric_key(
    observation: ScoreObservation,
) -> tuple[str, str] | None:
    return _metric_key(observation.metric_canonical, observation.metric_raw)


def _representative_key(
    representative: _Representative,
) -> tuple[str, str] | None:
    return _metric_key(representative.metric_canonical, representative.metric_raw)


def _metric_key(
    metric_canonical: str | None,
    metric_raw: str | None,
) -> tuple[str, str] | None:
    if metric_canonical:
        return "canonical", metric_canonical.lower()
    normalized_raw = _normalized_metric_raw(metric_raw)
    return ("raw", normalized_raw) if normalized_raw else None


def _normalized_metric_raw(metric_raw: str | None) -> str | None:
    if not metric_raw:
        return None
    tokens = [
        token
        for token in re.findall(r"[a-z]+\d*|\d+", metric_raw.lower())
        if token not in _METRIC_CONTEXT_TOKENS and not token.isdigit()
    ]
    return "".join(tokens) or None


def _deduplicate_observations(
    observations: Any,
) -> list[ScoreObservation]:
    unique: dict[tuple[Any, ...], ScoreObservation] = {}
    for observation in observations:
        key = (
            observation.observation_id,
            observation.value,
            observation.metric_canonical,
            observation.metric_raw,
            observation.source,
            observation.locator,
            observation.raw_text,
        )
        unique.setdefault(key, observation)
    return list(unique.values())


def _lb_observation_source(observation: ScoreObservation) -> str:
    source = observation.source_kind or observation.source
    if source in {"title", "notebook_title"}:
        return "score_observation_title"
    if source in {"ref", "notebook_ref"}:
        return "score_observation_ref"
    return "score_observation_text"


def _has_cv_side(notebook: NotebookFacts) -> bool:
    return any(
        observation.split == "cv"
        for observation in _plausible(notebook.score_observations)
    ) or bool(notebook.declared_cv)


def _has_lb_side(notebook: NotebookFacts) -> bool:
    return notebook.public_score is not None or any(
        observation.split == "lb"
        for observation in _plausible(notebook.score_observations)
    )


def _plausible(observations: list[ScoreObservation]) -> list[ScoreObservation]:
    return [
        observation
        for observation in observations
        if getattr(observation, "plausible", True)
    ]


def _zero_pairs_reason(
    notebooks: list[NotebookFacts],
    rejection_counts: dict[str, int],
) -> str:
    if not notebooks:
        return "No notebooks were collected."
    reasons = [
        f"missing CV side: {rejection_counts['missing_cv']}",
        f"missing leaderboard side: {rejection_counts['missing_lb']}",
        f"metric mismatch: {rejection_counts['metric_mismatch']}",
        f"ambiguous metric: {rejection_counts['ambiguous_metric']}",
        f"scale mismatch: {rejection_counts['scale_mismatch']}",
        f"implausible gap: {rejection_counts['implausible_gap']}",
        f"unknown/conflicting split: {rejection_counts['ambiguous_split']}",
    ]
    return "No comparable CV/LB pairs (" + "; ".join(reasons) + ")."


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized_identity(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(part for part in re.split(r"[\s-]+", value.casefold()) if part)


def _team_entry_order(
    entry: tuple[str, float, int | None, int],
) -> tuple[int, int]:
    rank = entry[2]
    return (rank if rank is not None else 2**31 - 1, entry[3])


def _select_declared_cv(values: list[Any]) -> float | None:
    numeric_values = [number for value in values if (number := _finite_float(value)) is not None]
    if not numeric_values:
        return None
    if len(numeric_values) <= 3:
        return numeric_values[0]
    return statistics.median(numeric_values)


def _notebook_cv_metric(notebook: NotebookFacts) -> str | None:
    observation_metrics = {
        metric
        for observation in notebook.declared_cv_observations
        if (metric := _canonical_metric_name(observation.metric_name)) is not None
    }
    if len(observation_metrics) == 1:
        return next(iter(observation_metrics))

    code_metrics = {
        metric
        for observation in notebook.metrics
        if (metric := _canonical_metric_name(observation.name)) is not None
    }
    return next(iter(code_metrics)) if len(code_metrics) == 1 else None


def _canonical_metric_name(value: str | None) -> str | None:
    return canonicalize_metric_label(value) or (
        value.strip().lower() if value and value.strip() else None
    )


def _spearman_correlation(
    left: list[float],
    right: list[float],
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if len(left) < 2:
        return None
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Pearson inputs must have equal lengths")
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_offsets = [value - left_mean for value in left]
    right_offsets = [value - right_mean for value in right]
    left_squared = sum(value * value for value in left_offsets)
    right_squared = sum(value * value for value in right_offsets)
    if left_squared == 0 or right_squared == 0:
        return None
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_offsets, right_offsets, strict=False)
    )
    correlation = numerator / math.sqrt(left_squared * right_squared)
    return max(-1.0, min(1.0, correlation))
