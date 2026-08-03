from __future__ import annotations

import builtins
import importlib
from typing import Any

import pytest

from kaggle_researcher.facts.cv_lb import (
    _select_declared_cv,
    _spearman_correlation,
    build_cv_lb_pairs,
    diagnose_cv_lb,
    summarize_cv_lb,
)
from kaggle_researcher.facts.models import (
    CvLbPair,
    DeclaredCvObservation,
    NotebookFacts,
)


def test_build_pairs_uses_first_declared_cv_and_preserves_notebook_order() -> None:
    notebooks = [
        _notebook("author/first", [" 0.8123 ", "0.9999"], 0.8, "lc_a"),
        _notebook("author/second", ["0.7450"], 0.75, "lc_b"),
    ]

    pairs = build_cv_lb_pairs(notebooks)

    assert pairs == [
        CvLbPair(
            notebook_ref="author/first",
            declared_cv=0.8123,
            public_score=0.8,
            lineage_cluster_id="lc_a",
        ),
        CvLbPair(
            notebook_ref="author/second",
            declared_cv=0.745,
            public_score=0.75,
            lineage_cluster_id="lc_b",
        ),
    ]


def test_build_pairs_filters_invalid_values_before_selecting_cv() -> None:
    notebooks = [
        _notebook("author/no-cv", [], 0.7, "lc_a"),
        _notebook("author/no-lb", ["0.7"], None, "lc_b"),
        _notebook("author/bad-first", ["not-a-score", "0.8"], 0.75, "lc_c"),
        _notebook("author/nan-cv", ["nan"], 0.75, "lc_d"),
        _notebook("author/infinite-lb", ["0.8"], float("inf"), "lc_e"),
    ]

    assert build_cv_lb_pairs(notebooks) == [
        CvLbPair(
            notebook_ref="author/bad-first",
            declared_cv=0.8,
            public_score=0.75,
            lineage_cluster_id="lc_c",
        )
    ]


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
    notebook = _notebook(
        "author/folds", ["0.70", "bad", "0.72", "0.74", "9.0"], 0.71, "lc_a"
    )

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

    diagnostics = diagnose_cv_lb(notebooks, pairs)

    assert diagnostics.notebooks_total == 3
    assert diagnostics.notebooks_with_public_score == 2
    assert diagnostics.notebooks_with_declared_cv == 2
    assert diagnostics.notebooks_with_both == 1
    assert diagnostics.comparable_pairs == 0
    assert diagnostics.rejected_non_comparable_pairs == 1
    assert diagnostics.zero_pairs_reason == (
        "CV and leaderboard metrics are not comparable."
    )


def test_summary_returns_none_statistics_for_empty_pairs() -> None:
    assert summarize_cv_lb([]) == {
        "count": 0,
        "mean_gap": None,
        "median_gap": None,
        "spearman": None,
        "distinct_lineage_clusters": 0,
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
    module_path = (
        Path(__file__).resolve().parents[2]
        / "kaggle_researcher"
        / "facts"
        / "cv_lb.py"
    )

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

    module_path = (
        Path(__file__).resolve().parents[2]
        / "kaggle_researcher"
        / "facts"
        / "cv_lb.py"
    )
    source = module_path.read_text(encoding="utf-8")

    assert "print(" not in source
    assert "logging" not in source
    for forbidden in ("deepseek_client", "retriever", "embedder", "store"):
        assert forbidden not in source


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
