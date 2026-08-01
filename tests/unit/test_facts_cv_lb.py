from __future__ import annotations

import builtins
from typing import Any

import pytest

from kaggle_researcher.facts.cv_lb import build_cv_lb_pairs, summarize_cv_lb
from kaggle_researcher.facts.models import CvLbPair, NotebookFacts


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


def test_build_pairs_skips_missing_or_unparseable_first_value() -> None:
    notebooks = [
        _notebook("author/no-cv", [], 0.7, "lc_a"),
        _notebook("author/no-lb", ["0.7"], None, "lc_b"),
        _notebook("author/bad-first", ["not-a-score", "0.8"], 0.75, "lc_c"),
        _notebook("author/nan-cv", ["nan"], 0.75, "lc_d"),
        _notebook("author/infinite-lb", ["0.8"], float("inf"), "lc_e"),
    ]

    assert build_cv_lb_pairs(notebooks) == []


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


def test_summary_returns_none_when_scipy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing_scipy(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scipy.stats":
            raise ModuleNotFoundError("scipy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_scipy)
    pairs = [
        _pair("a", 0.6, 0.3, "lc_a"),
        _pair("b", 0.7, 0.2, "lc_b"),
        _pair("c", 0.8, 0.1, "lc_c"),
    ]

    assert summarize_cv_lb(pairs)["spearman"] is None


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
