from __future__ import annotations

import builtins
import importlib
from typing import Any

import pytest

from kaggle_researcher.facts.cv_lb import (
    _select_declared_cv,
    _spearman_correlation,
    build_cv_lb_pairs,
    build_leaderboard_cv_lb_pairs,
    diagnose_cv_lb,
    match_leaderboard_scores,
    summarize_cv_lb,
)
from kaggle_researcher.facts.models import (
    CvLbPair,
    DeclaredCvObservation,
    LeaderboardEntry,
    LeaderboardMatch,
    NotebookFacts,
    PublicLeaderboard,
    ScoreObservation,
)


def test_build_pairs_uses_first_declared_cv_and_preserves_notebook_order() -> None:
    notebooks = [
        _notebook("author/first", [" 0.8123 ", "0.9999"], 0.8, "lc_a"),
        _notebook("author/second", ["0.7450"], 0.75, "lc_b"),
    ]

    pairs = build_cv_lb_pairs(notebooks)

    assert [
        (pair.notebook_ref, pair.declared_cv, pair.public_score, pair.cv_source) for pair in pairs
    ] == [
        ("author/first", 0.8123, 0.8, "declared_cv_legacy"),
        ("author/second", 0.745, 0.75, "declared_cv_legacy"),
    ]


def test_build_pairs_filters_invalid_values_before_selecting_cv() -> None:
    notebooks = [
        _notebook("author/no-cv", [], 0.7, "lc_a"),
        _notebook("author/no-lb", ["0.7"], None, "lc_b"),
        _notebook("author/bad-first", ["not-a-score", "0.8"], 0.75, "lc_c"),
        _notebook("author/nan-cv", ["nan"], 0.75, "lc_d"),
        _notebook("author/infinite-lb", ["0.8"], float("inf"), "lc_e"),
    ]

    pairs = build_cv_lb_pairs(notebooks)

    assert len(pairs) == 1
    assert pairs[0].notebook_ref == "author/bad-first"
    assert pairs[0].declared_cv == pytest.approx(0.8)
    assert pairs[0].public_score == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], None),
        (["bad", "nan", float("inf")], None),
        (["0.7"], 0.7),
        (["bad", "0.7", "0.8", "0.9"], 0.7),
        ([0.1, 0.2, 0.3, 0.4], 0.25),
        ([0.1, 0.2, 0.3, 0.4, 0.5], 0.3),
        ([0.7, 0.71, 0.72, 99.0], 0.715),
    ],
)
def test_select_declared_cv(values: list[object], expected: float | None) -> None:
    result = _select_declared_cv(values)

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_build_pairs_uses_median_for_long_fold_like_list() -> None:
    notebook = _notebook("author/folds", ["0.70", "bad", "0.72", "0.74", "9.0"], 0.71, "lc_a")

    assert build_cv_lb_pairs([notebook])[0].declared_cv == pytest.approx(0.73)


def test_incompatible_declared_and_competition_metrics_are_not_paired() -> None:
    notebook = _notebook("author/accuracy", ["0.8123"], 0.8, "lc_a")
    notebook.declared_cv_observations = [
        DeclaredCvObservation(
            value=0.8123,
            metric_name="accuracy",
            locator="cell_1",
            raw_text="validation accuracy 0.8123",
        )
    ]

    assert build_cv_lb_pairs([notebook], "mAP") == []
    compatible = build_cv_lb_pairs([notebook], "accuracy")
    assert len(compatible) == 1
    assert compatible[0].metric_name == "accuracy"


def test_diagnostics_count_public_cv_both_and_rejected_separately() -> None:
    public_only = _notebook("author/public", [], 0.8, "lc_a")
    cv_only = _notebook("author/cv", ["0.79"], None, "lc_b")
    incompatible = _notebook("author/both", ["0.78"], 0.77, "lc_c")
    incompatible.declared_cv_observations = [
        DeclaredCvObservation(
            value=0.78,
            metric_name="accuracy",
            locator="cell_0",
            raw_text="local accuracy 0.78",
        )
    ]
    notebooks = [public_only, cv_only, incompatible]
    pairs = build_cv_lb_pairs(notebooks, "mAP")

    diagnostics = diagnose_cv_lb(notebooks, pairs, "mAP")

    assert diagnostics.notebooks_total == 3
    assert diagnostics.notebooks_with_public_score == 2
    assert diagnostics.notebooks_with_declared_cv == 2
    assert diagnostics.notebooks_with_both == 1
    assert diagnostics.comparable_pairs == 0
    assert diagnostics.rejected_non_comparable_pairs == 1
    assert diagnostics.zero_pairs_reason is not None
    assert "metric mismatch: 1" in diagnostics.zero_pairs_reason
    assert "missing CV side: 1" in diagnostics.zero_pairs_reason


def test_summary_returns_none_statistics_for_empty_pairs() -> None:
    assert summarize_cv_lb([]) == {
        "count": 0,
        "mean_gap": None,
        "median_gap": None,
        "spearman": None,
        "distinct_lineage_clusters": 0,
        "reliability": "insufficient",
        "note": "no pairs; gap cannot be estimated",
        "leaderboard_pair_count": 0,
        "leaderboard_median_gap": None,
        "leaderboard_gap_note": None,
    }


def test_summary_computes_gap_statistics_and_distinct_lineages() -> None:
    pairs = [
        _pair("a", declared_cv=0.9, public_score=0.8, lineage="lc_shared"),
        _pair("b", declared_cv=0.7, public_score=0.8, lineage="lc_shared"),
    ]

    summary = summarize_cv_lb(pairs)

    assert summary["count"] == 2
    assert summary["mean_gap"] == pytest.approx(0.0)
    assert summary["median_gap"] == pytest.approx(0.0)
    assert summary["spearman"] is None
    assert summary["distinct_lineage_clusters"] == 1
    assert summary["reliability"] == "insufficient"


@pytest.mark.parametrize(
    ("count", "expected_reliability", "expected_note"),
    [
        (1, "insufficient", "single pair"),
        (4, "insufficient", "fewer than 5 pairs"),
        (5, "weak", "fewer than 15 pairs"),
        (14, "weak", "fewer than 15 pairs"),
        (15, "sufficient", None),
    ],
)
def test_summary_reliability_thresholds(
    count: int,
    expected_reliability: str,
    expected_note: str | None,
) -> None:
    pairs = [
        _pair(
            f"notebook-{index}",
            declared_cv=0.9 + index / 100,
            public_score=0.8 + index / 100,
            lineage=f"lc_{index}",
        )
        for index in range(count)
    ]

    summary = summarize_cv_lb(pairs)

    assert summary["reliability"] == expected_reliability
    if expected_note is None:
        assert summary["note"] is None
    else:
        assert expected_note in str(summary["note"])


def test_summary_returns_spearman_for_three_or_more_pairs() -> None:
    pairs = [
        _pair("a", 0.6, 0.3, "lc_a"),
        _pair("b", 0.7, 0.2, "lc_b"),
        _pair("c", 0.8, 0.1, "lc_c"),
    ]

    summary = summarize_cv_lb(pairs)

    assert summary["spearman"] == pytest.approx(-1.0)
    assert summary["distinct_lineage_clusters"] == 3


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 1.0),
        ([1.0, 2.0, 3.0], [30.0, 20.0, 10.0], -1.0),
        ([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], 0.9486832980505138),
    ],
)
def test_local_spearman(left: list[float], right: list[float], expected: float) -> None:
    assert _spearman_correlation(left, right) == pytest.approx(expected)


def test_local_spearman_handles_short_constant_and_unequal_inputs() -> None:
    assert _spearman_correlation([], []) is None
    assert _spearman_correlation([1.0], [2.0]) is None
    assert _spearman_correlation([1.0, 1.0], [2.0, 3.0]) is None
    with pytest.raises(ValueError, match="equal lengths"):
        _spearman_correlation([1.0], [1.0, 2.0])


def test_module_imports_when_scipy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    real_import = builtins.__import__

    def import_without_scipy(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("scipy"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_scipy)
    module = importlib.import_module("kaggle_researcher.facts.cv_lb")

    importlib.reload(module)
    module_path = Path(__file__).resolve().parents[2] / "kaggle_researcher" / "facts" / "cv_lb.py"

    assert "scipy" not in module_path.read_text(encoding="utf-8")


def test_summary_constant_values_produce_no_spearman_number() -> None:
    pairs = [
        _pair("a", 0.7, 0.1, "lc_a"),
        _pair("b", 0.7, 0.2, "lc_b"),
        _pair("c", 0.7, 0.3, "lc_c"),
    ]

    assert summarize_cv_lb(pairs)["spearman"] is None


def test_module_has_no_text_output_or_pipeline_dependencies() -> None:
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[2] / "kaggle_researcher" / "facts" / "cv_lb.py"
    source = module_path.read_text(encoding="utf-8")

    assert "print(" not in source
    assert "logging" not in source
    for forbidden in ("deepseek_client", "retriever", "embedder", "store"):
        assert forbidden not in source


def test_observation_pair_requires_same_notebook_and_compatible_metric() -> None:
    first = _notebook("author/first", [], None, "lc_a")
    first.score_observations = [
        _score("cv-1", 0.80, "cv", "mAP"),
        _score("lb-1", 0.82, "lb", "mAP", source="title"),
    ]
    second = _notebook("author/second", [], None, "lc_b")
    second.score_observations = [_score("cv-2", 0.7, "cv", "accuracy")]

    pairs = build_cv_lb_pairs([second, first], "mAP")

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.notebook_ref == "author/first"
    assert pair.cv_source == "score_observation"
    assert pair.lb_source == "observation"
    assert pair.cv_observation_ids == ["cv-1"]
    assert pair.lb_observation_ids == ["lb-1"]
    assert pair.metric_canonical == "mAP"
    assert pair.metric_match == "exact"


def test_different_metrics_and_unknown_splits_do_not_pair() -> None:
    mismatch = _notebook("author/mismatch", [], None, "lc_a")
    mismatch.score_observations = [
        _score("cv-map", 0.8, "cv", "mAP"),
        _score("lb-acc", 0.8, "lb", "accuracy"),
    ]
    unknown = _notebook("author/unknown", [], None, "lc_b")
    unknown.score_observations = [
        _score("unknown", 0.8, "unknown", "mAP"),
        _score("lb", 0.81, "lb", "mAP"),
    ]

    assert build_cv_lb_pairs([mismatch, unknown], "mAP") == []
    diagnostics = diagnose_cv_lb([mismatch, unknown], [], "mAP")
    assert diagnostics.pairs_rejected_metric_mismatch == 1
    assert diagnostics.pairs_rejected_missing_cv == 1
    assert diagnostics.pairs_rejected_ambiguous_split == 1


def test_uncanonicalized_metric_within_notebook_is_assumed_match() -> None:
    notebook = _notebook("author/assumed", [], None, "lc")
    notebook.score_observations = [
        _score("cv-map", 0.8382, "cv", "mAP", metric_raw="full_mAP"),
        _score("lb", 0.797, "lb", None, metric_raw="LB"),
    ]

    pair = build_cv_lb_pairs([notebook])[0]

    assert pair.cv_score == pytest.approx(0.8382)
    assert pair.lb_score == pytest.approx(0.797)
    assert pair.metric_canonical == "mAP"
    assert pair.metric_match == "assumed"


def test_matching_normalized_raw_metric_creates_pair() -> None:
    notebook = _notebook("author/raw", [], None, "lc")
    notebook.score_observations = [
        _score("cv", 2.0, "cv", None, metric_raw="Custom Quality"),
        _score("lb", 3.0, "lb", None, metric_raw="custom-quality"),
    ]

    pair = build_cv_lb_pairs([notebook])[0]

    assert pair.metric_raw in {"Custom Quality", "custom-quality"}
    assert pair.cv_score == pytest.approx(2.0)
    assert pair.lb_score == pytest.approx(3.0)
    assert pair.cv_selection_reason == "unknown_direction_max_fallback"


def test_api_public_score_has_priority_over_author_reported_lb() -> None:
    notebook = _notebook("author/api", ["0.70"], 0.81, "lc")
    notebook.score_observations = [
        _score("cv", 0.80, "cv", "mAP"),
        _score("title", 0.99, "lb", "mAP", source="title"),
    ]

    pair = build_cv_lb_pairs([notebook], "mAP")[0]

    assert pair.declared_cv == pytest.approx(0.80)
    assert pair.public_score == pytest.approx(0.81)
    assert pair.cv_source == "score_observation"
    assert pair.lb_source == "observation"
    assert pair.lb_observation_ids == []


def test_legacy_declared_cv_and_public_score_remain_fallbacks() -> None:
    notebook = _notebook("author/legacy", ["0.72"], 0.70, "lc")

    pair = build_cv_lb_pairs([notebook])[0]

    assert pair.cv_source == "declared_cv_legacy"
    assert pair.lb_source == "observation"
    assert pair.declared_cv == pytest.approx(0.72)
    assert pair.public_score == pytest.approx(0.70)


@pytest.mark.parametrize(
    ("metric", "cv_values", "lb_values", "expected_cv", "expected_lb"),
    [
        ("roc_auc", [0.80, 0.82], [0.81, 0.84], 0.82, 0.84),
        ("rmse", [1.4, 1.2], [1.3, 1.1], 1.2, 1.1),
    ],
)
def test_representative_score_respects_optimization_direction(
    metric: str,
    cv_values: list[float],
    lb_values: list[float],
    expected_cv: float,
    expected_lb: float,
) -> None:
    notebook = _notebook(f"author/{metric}", [], None, "lc")
    notebook.score_observations = [
        *[_score(f"cv-{index}", value, "cv", metric) for index, value in enumerate(cv_values)],
        *[_score(f"lb-{index}", value, "lb", metric) for index, value in enumerate(lb_values)],
    ]

    pair = build_cv_lb_pairs([notebook], metric)[0]

    assert pair.cv_score == pytest.approx(expected_cv)
    assert pair.lb_score == pytest.approx(expected_lb)
    expected_aggregation = "min" if metric == "rmse" else "max"
    assert pair.cv_aggregation == expected_aggregation
    assert pair.lb_aggregation == expected_aggregation


def test_fold_series_uses_median_instead_of_best_fold() -> None:
    notebook = _notebook("author/folds-observed", [], None, "lc")
    notebook.score_observations = [
        _score("fold-1", 0.80, "cv", "mAP", signals=["fold"]),
        _score("fold-2", 0.82, "cv", "mAP", signals=["fold"]),
        _score("fold-3", 0.78, "cv", "mAP", signals=["fold"]),
        _score("lb", 0.81, "lb", "mAP"),
    ]

    pair = build_cv_lb_pairs([notebook], "mAP")[0]

    assert pair.cv_score == pytest.approx(0.80)
    assert pair.cv_aggregation == "median_fold_series"
    assert pair.cv_representative_observation_id is None
    assert pair.lb_score == pytest.approx(0.81)


def test_unknown_direction_fallback_and_diagnostics_are_explicit() -> None:
    notebook = _notebook("author/custom", [], None, "lc")
    notebook.score_observations = [
        _score("cv-1", -2.0, "cv", None, metric_raw="custom"),
        _score("cv-2", -1.0, "cv", None, metric_raw="custom"),
        _score("lb-1", -3.0, "lb", None, metric_raw="custom"),
        _score("lb-2", -0.5, "lb", None, metric_raw="custom"),
    ]

    pairs = build_cv_lb_pairs([notebook])
    diagnostics = diagnose_cv_lb([notebook], pairs)

    assert pairs[0].cv_score == pytest.approx(-1.0)
    assert pairs[0].lb_score == pytest.approx(-0.5)
    assert pairs[0].optimization_direction is None
    assert diagnostics.unknown_direction_fallbacks == 2


def test_percent_normalization_is_compatible_but_unclear_bounded_scale_is_not() -> None:
    normalized = _notebook("author/normalized", [], None, "lc_a")
    normalized.score_observations = [
        _score("cv-percent", 0.945, "cv", "accuracy", value_raw="94.5%"),
        _score("lb-fraction", 0.94, "lb", "accuracy", value_raw="0.94"),
    ]
    unclear = _notebook("author/unclear", [], None, "lc_b")
    unclear.score_observations = [
        _score("cv-large", 94.5, "cv", "accuracy", value_raw="94.5"),
        _score("lb-small", 0.94, "lb", "accuracy", value_raw="0.94"),
    ]

    pairs = build_cv_lb_pairs([normalized, unclear], "accuracy")
    diagnostics = diagnose_cv_lb([normalized, unclear], pairs, "accuracy")

    assert [pair.notebook_ref for pair in pairs] == ["author/normalized"]
    assert diagnostics.pairs_rejected_scale_mismatch == 1


def test_pairing_is_deterministic_under_notebook_and_observation_permutation() -> None:
    first = _notebook("author/z", [], None, "lc_z")
    first.score_observations = [
        _score("lb-z", 0.8, "lb", "mAP"),
        _score("cv-z", 0.78, "cv", "mAP"),
    ]
    second = _notebook("author/a", [], None, "lc_a")
    second.score_observations = [
        _score("cv-a", 0.6, "cv", "mAP"),
        _score("lb-a", 0.62, "lb", "mAP"),
    ]

    forward = build_cv_lb_pairs([first, second], "mAP")
    first.score_observations.reverse()
    second.score_observations.reverse()
    reverse = build_cv_lb_pairs([second, first], "mAP")

    assert forward == reverse
    assert [pair.notebook_ref for pair in forward] == ["author/a", "author/z"]


def test_leaderboard_matching_prefers_exact_then_partial_identity_matches() -> None:
    exact = _notebook("alice-smith/notebook", [], None, "lc_exact")
    exact.author = "Alice-Smith"
    partial = _notebook("bob/notebook", [], None, "lc_partial")
    partial.author = "Bob"
    leaderboard = _public_leaderboard(
        ("Alice Smith", 0.81, 1),
        ("Team Bob Squad", 0.79, 2),
    )

    matches = match_leaderboard_scores([partial, exact], leaderboard)

    assert matches[exact.ref].match_confidence == "exact"
    assert matches[exact.ref].team_name == "Alice Smith"
    assert matches[partial.ref].match_confidence == "partial"
    assert matches[partial.ref].team_name == "Team Bob Squad"


def test_ambiguous_leaderboard_matches_are_rejected() -> None:
    one_team_many_authors = [
        _notebook("ann/one", [], None, "lc_ann"),
        _notebook("anna/two", [], None, "lc_anna"),
    ]
    for notebook in one_team_many_authors:
        notebook.author = notebook.ref.partition("/")[0]
    one_author_many_teams = _notebook("alice/three", [], None, "lc_alice")
    one_author_many_teams.author = "alice"
    leaderboard = _public_leaderboard(
        ("anna", 0.8, 1),
        ("alice alpha", 0.7, 2),
        ("alice beta", 0.6, 3),
    )

    matches = match_leaderboard_scores(
        [*one_team_many_authors, one_author_many_teams],
        leaderboard,
    )

    assert matches == {}


def test_leaderboard_pairs_are_built_and_summarized_separately() -> None:
    observed = _pair("observed", 0.82, 0.8, "lc_observed")
    notebook = _notebook("alice/notebook", [], None, "lc_leaderboard")
    notebook.score_observations = [_score("cv", 0.79, "cv", "mAP")]
    match = LeaderboardMatch(
        notebook_ref=notebook.ref,
        team_name="Alice",
        score=0.77,
        match_confidence="exact",
    )

    leaderboard_pairs = build_leaderboard_cv_lb_pairs(
        [notebook],
        {notebook.ref: match},
        "mAP",
    )
    summary = summarize_cv_lb([observed, *leaderboard_pairs])
    diagnostics = diagnose_cv_lb(
        [notebook],
        [observed, *leaderboard_pairs],
        "mAP",
    )

    assert len(leaderboard_pairs) == 1
    assert leaderboard_pairs[0].lb_source == "leaderboard_match"
    assert summary["count"] == 1
    assert summary["mean_gap"] == pytest.approx(0.02)
    assert summary["median_gap"] == pytest.approx(0.02)
    assert summary["leaderboard_pair_count"] == 1
    assert summary["leaderboard_median_gap"] == pytest.approx(0.02)
    assert "best submission" in str(summary["leaderboard_gap_note"])
    assert diagnostics.pairs_created == 1
    assert diagnostics.pairs_created_from_leaderboard_match == 1


def test_unavailable_leaderboard_has_no_matches() -> None:
    leaderboard = PublicLeaderboard(
        status="unavailable",
        entries=[],
        entry_count=0,
        unavailable_reason="forbidden",
    )

    assert match_leaderboard_scores([_notebook("alice/n", [], None, "lc")], leaderboard) == {}


def test_implausible_bounded_metric_gap_is_rejected_and_counted() -> None:
    notebook = _notebook("author/tolerance", [], 0.97085, "lc")
    notebook.score_observations = [
        _score("tol", 0.00001, "cv", None, metric_raw="tol")
    ]

    pairs = build_cv_lb_pairs([notebook], "Roc Auc Score")
    diagnostics = diagnose_cv_lb([notebook], pairs, "Roc Auc Score")

    assert pairs == []
    assert len(pairs.implausible_gap_pairs) == 1
    assert pairs.implausible_gap_pairs[0].comparability_status == "implausible_gap"
    assert diagnostics.pairs_rejected_implausible_gap == 1
    assert diagnostics.rejected_implausible_gap == 1
    assert diagnostics.leaderboard_pairs_rejected_implausible_gap == 0
    assert diagnostics.zero_pairs_reason is not None
    assert "implausible gap: 1" in diagnostics.zero_pairs_reason
    assert summarize_cv_lb(pairs)["median_gap"] is None


def test_implausible_leaderboard_match_gap_is_rejected_and_counted() -> None:
    notebook = _notebook("author/tolerance", [], None, "lc")
    notebook.score_observations = [
        _score("tol", 0.00001, "cv", None, metric_raw="tol")
    ]
    match = LeaderboardMatch(
        notebook_ref=notebook.ref,
        team_name="author",
        score=0.97085,
        match_confidence="exact",
    )
    matches = {notebook.ref: match}

    pairs = build_leaderboard_cv_lb_pairs(
        [notebook],
        matches,
        "Roc Auc Score",
    )
    diagnostics = diagnose_cv_lb(
        [notebook],
        pairs,
        "Roc Auc Score",
        matches,
    )

    assert pairs == []
    assert len(pairs.implausible_gap_pairs) == 1
    assert pairs.implausible_gap_pairs[0].lb_source == "leaderboard_match"
    assert diagnostics.pairs_rejected_implausible_gap == 0
    assert diagnostics.leaderboard_pairs_rejected_implausible_gap == 1
    assert diagnostics.rejected_implausible_gap == 1
    summary = summarize_cv_lb(pairs)
    assert summary["leaderboard_pair_count"] == 0
    assert summary["leaderboard_median_gap"] is None


def test_gap_threshold_applies_only_to_bounded_metrics() -> None:
    plausible_auc = _notebook("author/auc", [], 0.95, "lc_auc")
    plausible_auc.score_observations = [
        _score("cv", 0.91, "cv", None, metric_raw="score")
    ]
    unbounded = _notebook("author/rmse", [], 0.95, "lc_rmse")
    unbounded.score_observations = [
        _score("cv", 2.45, "cv", None, metric_raw="score")
    ]

    auc_pairs = build_cv_lb_pairs([plausible_auc], "roc_auc")
    rmse_pairs = build_cv_lb_pairs([unbounded], "rmse")

    assert len(auc_pairs) == 1
    assert auc_pairs[0].gap == pytest.approx(-0.04)
    assert len(rmse_pairs) == 1
    assert rmse_pairs[0].gap == pytest.approx(1.5)


def test_auc_gap_above_threshold_is_preserved_but_excluded_from_statistics() -> None:
    notebook = _notebook("author/auc-gap", [], 0.95, "lc_auc")
    notebook.score_observations = [
        _score("cv", 0.82, "cv", "roc_auc")
    ]

    pairs = build_cv_lb_pairs([notebook], "roc_auc")
    diagnostics = diagnose_cv_lb([notebook], pairs, "roc_auc")
    summary = summarize_cv_lb(pairs)

    assert pairs == []
    assert len(pairs.implausible_gap_pairs) == 1
    assert pairs.implausible_gap_pairs[0].absolute_gap == pytest.approx(0.13)
    assert summary["count"] == 0
    assert summary["mean_gap"] is None
    assert diagnostics.rejected_implausible_gap == 1


def test_implausible_score_observation_does_not_build_cv_lb_pair() -> None:
    notebook = _notebook("author/implausible", [], None, "lc_bad")
    notebook.score_observations = [
        _score(
            "bad-cv",
            1e-5,
            "cv",
            "roc_auc",
            plausible=False,
            implausible_reason="excluded_label",
        ),
        _score("good-lb", 0.97, "lb", "roc_auc"),
    ]

    assert build_cv_lb_pairs([notebook], "roc_auc") == []


def _score(
    observation_id: str,
    value: float,
    split: str,
    metric_canonical: str | None,
    *,
    metric_raw: str | None = None,
    source: str = "markdown",
    signals: list[str] | None = None,
    value_raw: str | None = None,
    plausible: bool = True,
    implausible_reason: str | None = None,
) -> ScoreObservation:
    return ScoreObservation(
        value=value,
        value_raw=value_raw or str(value),
        metric_raw=metric_raw or metric_canonical,
        metric_canonical=metric_canonical,
        locator=source,
        raw_text=f"{metric_raw or metric_canonical or 'score'}: {value}",
        source=source,
        source_kind=source,
        split=split,
        split_signals=signals or [],
        observation_id=observation_id,
        plausible=plausible,
        implausible_reason=implausible_reason,
    )


def _notebook(
    ref: str,
    declared_cv: list[str],
    public_score: float | None,
    lineage: str,
) -> NotebookFacts:
    return NotebookFacts(
        ref=ref,
        title=ref,
        public_score=public_score,
        ast_fingerprint=f"fp-{ref}",
        lineage_cluster_id=lineage,
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=declared_cv,
        parse_status="ok",
    )


def _pair(
    ref: str,
    declared_cv: float,
    public_score: float,
    lineage: str,
) -> CvLbPair:
    return CvLbPair(
        notebook_ref=ref,
        declared_cv=declared_cv,
        public_score=public_score,
        lineage_cluster_id=lineage,
    )


def _public_leaderboard(
    *entries: tuple[str, float, int],
) -> PublicLeaderboard:
    return PublicLeaderboard(
        status="collected",
        entries=[
            LeaderboardEntry(team_name=name, score=score, rank=rank)
            for name, score, rank in entries
        ],
        entry_count=len(entries),
        unavailable_reason=None,
    )
