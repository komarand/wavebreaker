from __future__ import annotations

from typing import Literal

import pytest

from kaggle_researcher.facts.models import (
    DeclaredCvObservation,
    NotebookFacts,
    ScoreObservation,
)
from kaggle_researcher.facts.notebook_ast import (
    classify_score_split,
    diagnose_scores,
    extract_score_observations,
    recanonicalize_score_observations,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CV: 0.81", "cv"),
        ("OOF RMSE 1.23", "cv"),
        ("validation mAP 0.94", "cv"),
        ("Fold 3 logloss: 0.042", "cv"),
        ("local score 0.82", "cv"),
        ("holdout MAE 12.5", "cv"),
        ("offline metric 0.73", "cv"),
        ("public LB 0.88", "lb"),
        ("private leaderboard: 0.91", "lb"),
        ("submission score 0.75", "lb"),
        ("online score 0.82", "lb"),
        ("score: 0.75", "unknown"),
        ("test score: 0.75", "unknown"),
        ("final score: 0.75", "unknown"),
        ("epoch 7 score 0.75", "unknown"),
    ],
)
def test_score_split_classification_from_observed_context(
    text: str,
    expected: str,
) -> None:
    split, _ = classify_score_split(
        source_kind="markdown",
        context_text=text,
        context_signals=[],
        metric_raw=None,
        locator="cell_1",
    )

    assert split == expected


@pytest.mark.parametrize(
    ("source_kind", "text", "expected"),
    [
        ("title", "0-95-jaguar-re-id", "lb"),
        ("ref", "author/0-95-jaguar-re-id", "lb"),
        ("title", "CV-0.95-baseline", "cv"),
        ("ref", "oof-0-91-model", "cv"),
        ("title", "CV-vs-LB-0.95", "unknown"),
    ],
)
def test_title_and_ref_split_contract(
    source_kind: str,
    text: str,
    expected: str,
) -> None:
    split, signals = classify_score_split(
        source_kind=source_kind,
        context_text=text,
        context_signals=[],
        metric_raw=None,
        locator=source_kind,
    )

    assert split == expected
    assert len(signals) == len(set(signals))


def test_conflicting_and_duplicate_signals_are_unknown_and_deterministic() -> None:
    first = classify_score_split(
        source_kind="title",
        context_text="CV versus public LB score 0.95",
        context_signals=["cv", "public", "cv"],
        metric_raw="score",
        locator="title",
    )
    second = classify_score_split(
        source_kind="title",
        context_text="CV versus public LB score 0.95",
        context_signals=["cv", "public", "cv"],
        metric_raw="score",
        locator="title",
    )

    assert first == second
    assert first[0] == "unknown"
    assert first[1] == ["cv", "lb", "public"]


def test_recognized_metric_next_to_explicit_lb_is_classified_as_cv() -> None:
    split, signals = classify_score_split(
        source_kind="code_string",
        context_text="Best backbone (full_mAP=0.8382, LB=0.797)",
        context_signals=[],
        metric_raw="full_mAP",
        locator="cell_0",
    )

    assert split == "cv"
    assert signals == ["paired-with-lb"]


def test_recanonicalization_replaces_a_stale_derived_split_signal() -> None:
    observation = ScoreObservation(
        value=0.8382,
        value_raw="0.8382",
        metric_raw="full_mAP",
        locator="cell_0",
        raw_text="full_mAP=0.8382",
        source="code_string",
        source_kind="code_string",
        context_text="Best backbone (full_mAP=0.8382, LB=0.797)",
        context_signals=["lb"],
        split="lb",
        split_signals=["lb"],
    )

    classified = recanonicalize_score_observations(
        [_notebook([observation])],
        competition_metric_name=None,
    )[0].score_observations[0]

    assert classified.split == "cv"
    assert classified.split_signals == ["paired-with-lb"]


def test_validation_fraction_is_not_a_cv_score() -> None:
    split, signals = classify_score_split(
        source_kind="code",
        context_text="val_frac_per_id = 0.2",
        context_signals=[],
        metric_raw="val_frac_per_id",
        locator="cell_4",
    )

    assert split == "unknown"
    assert signals == []


def test_recanonicalization_clears_stale_structured_legacy_cv() -> None:
    score = ScoreObservation(
        value=0.2,
        value_raw="0.2",
        metric_raw="val_frac_per_id",
        locator="cell_4",
        raw_text="val_frac_per_id = 0.2",
        source="code",
        source_kind="code",
        context_text="val_frac_per_id = 0.2",
        split="cv",
        split_signals=["val"],
    )
    notebook = _notebook([score])
    notebook.declared_cv = ["0.2"]
    notebook.declared_cv_observations = [
        DeclaredCvObservation(
            value=0.2,
            locator="cell_4",
            raw_text="val_frac_per_id = 0.2",
        )
    ]

    classified = recanonicalize_score_observations(
        [notebook],
        competition_metric_name=None,
    )[0]

    assert classified.score_observations[0].split == "unknown"
    assert classified.declared_cv == []
    assert classified.declared_cv_observations == []


def test_recanonicalization_preserves_legacy_cv_without_structured_evidence() -> None:
    notebook = _notebook([])
    notebook.declared_cv = ["0.72"]
    notebook.declared_cv_observations = [
        DeclaredCvObservation(
            value=0.72,
            locator="legacy",
            raw_text="CV 0.72",
        )
    ]

    classified = recanonicalize_score_observations(
        [notebook],
        competition_metric_name=None,
    )[0]

    assert classified.declared_cv == ["0.72"]
    assert classified.declared_cv_observations[0].value == pytest.approx(0.72)


def test_old_score_observation_defaults_to_unknown() -> None:
    observation = ScoreObservation(
        value=0.75,
        value_raw="0.75",
        locator="cell_0",
        raw_text="score: 0.75",
        source="markdown",
    )

    assert observation.split == "unknown"
    assert observation.split_signals == []


@pytest.mark.parametrize(
    ("source", "text", "expected"),
    [
        ("title", "0-95-jaguar-re-id-frozen-dinov2", "lb"),
        ("title", "CV-0.95-baseline", "cv"),
        ("ref", "oof-0-91-model", "cv"),
        ("title", "CV-vs-LB-0.95", "unknown"),
    ],
)
def test_title_ref_extraction_preserves_provenance_and_assigns_split(
    source: Literal["title", "ref"],
    text: str,
    expected: str,
) -> None:
    observations, _, _ = extract_score_observations(
        text,
        locator=source,
        source=source,
    )
    notebook = _notebook(observations)

    classified = recanonicalize_score_observations(
        [notebook],
        competition_metric_name=None,
    )[0]

    assert len(classified.score_observations) == 1
    observation = classified.score_observations[0]
    assert observation.split == expected
    assert observation.source == source
    assert observation.source_kind == source
    assert observation.observation_id is not None
    assert classified.public_score is None
    assert (observation.value_raw in classified.declared_cv) is (expected == "cv")


def test_recanonicalization_is_stable_and_does_not_leak_context_between_lines() -> None:
    observations, _, _ = extract_score_observations(
        "Validation mAP: 0.81\nscore: 0.75",
        locator="cell_0",
        source="markdown",
    )
    notebook = _notebook(observations)

    first = recanonicalize_score_observations([notebook], competition_metric_name="mAP")[0]
    second = recanonicalize_score_observations([notebook], competition_metric_name="mAP")[0]

    assert [observation.split for observation in first.score_observations] == [
        "cv",
        "unknown",
    ]
    assert [observation.observation_id for observation in first.score_observations] == [
        observation.observation_id for observation in second.score_observations
    ]


def test_score_split_diagnostics_count_sides_and_notebooks() -> None:
    both = _notebook(
        [
            _raw_observation("CV mAP: 0.8"),
            _raw_observation("public LB: 0.81"),
            _raw_observation("score: 0.79"),
        ]
    )
    cv_only = _notebook([_raw_observation("OOF RMSE: 1.2")])
    cv_only.ref = "author/cv-only"
    classified = recanonicalize_score_observations([both, cv_only], competition_metric_name=None)

    diagnostics = diagnose_scores(classified)

    assert (diagnostics.split_cv, diagnostics.split_lb, diagnostics.split_unknown) == (
        2,
        1,
        1,
    )
    assert diagnostics.notebooks_with_cv_scores == 2
    assert diagnostics.notebooks_with_lb_scores == 1
    assert diagnostics.notebooks_with_both_sides == 1


def _raw_observation(text: str) -> ScoreObservation:
    observations, _, _ = extract_score_observations(
        text,
        locator="cell_0",
        source="markdown",
    )
    assert len(observations) == 1
    return observations[0]


def _notebook(observations: list[ScoreObservation]) -> NotebookFacts:
    return NotebookFacts(
        ref="author/notebook",
        title="Notebook",
        ast_fingerprint="fingerprint",
        lineage_cluster_id="lineage",
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        score_observations=observations,
        parse_status="ok",
    )
