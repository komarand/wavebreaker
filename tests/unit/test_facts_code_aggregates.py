from __future__ import annotations

import pytest

from kaggle_researcher.facts.code_aggregates import compute_code_aggregates
from kaggle_researcher.facts.models import CodeObservation, NotebookFacts


def test_twenty_forks_count_as_one_cluster_and_twenty_notebooks() -> None:
    notebooks = [
        _notebook(
            f"author/fork-{index}",
            "shared-lineage",
            [("LGBMClassifier", {})],
        )
        for index in range(20)
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    assert aggregates.total_clusters == 1
    assert aggregates.total_notebooks == 20
    assert aggregates.models[0].name == "LGBMClassifier"
    assert aggregates.models[0].cluster_count == 1
    assert aggregates.models[0].notebook_count == 20
    assert aggregates.models[0].cluster_share == pytest.approx(1.0)


def test_families_sort_by_cluster_count_then_name() -> None:
    notebooks = [
        _notebook("a/one", "a", [("LGBMClassifier", {}), ("XGBClassifier", {})]),
        _notebook("b/two", "b", [("LGBMClassifier", {}), ("XGBClassifier", {})]),
        _notebook("c/three", "c", [("LGBMClassifier", {}), ("CatBoostClassifier", {})]),
        _notebook("d/four", "d", [("AdaBoostClassifier", {})]),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    assert [item.name for item in aggregates.models] == [
        "LGBMClassifier",
        "XGBClassifier",
        "AdaBoostClassifier",
        "CatBoostClassifier",
    ]
    assert [item.cluster_count for item in aggregates.models] == [3, 2, 1, 1]


def test_typical_kwargs_use_cluster_medians_and_ignore_one_cluster_keys() -> None:
    notebooks = [
        _notebook(
            "a/one",
            "a",
            [
                (
                    "LGBMClassifier",
                    {
                        "n_estimators": "100",
                        "learning_rate": "0.1",
                        "one_cluster_only": "7",
                        "boosting_type": "gbdt",
                    },
                )
            ],
        ),
        _notebook(
            "b/two",
            "b",
            [
                (
                    "LGBMClassifier",
                    {"n_estimators": "300", "learning_rate": "0.2"},
                )
            ],
        ),
        _notebook(
            "c/three",
            "c",
            [("LGBMClassifier", {"n_estimators": "500"})],
        ),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    assert aggregates.models[0].typical_kwargs == {
        "n_estimators": "300",
        "learning_rate": "0.15",
    }


def test_model_combinations_require_two_models_and_two_clusters() -> None:
    notebooks = [
        _notebook("a/one", "a", [("LGBM", {}), ("XGB", {})]),
        _notebook("b/two", "b", [("XGB", {}), ("LGBM", {})]),
        _notebook("c/three", "c", [("LGBM", {})]),
        _notebook("d/four", "d", [("CatBoost", {}), ("LogReg", {})]),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    assert [item.model_dump() for item in aggregates.model_combinations] == [
        {"names": ["LGBM", "XGB"], "cluster_count": 2}
    ]


def test_best_public_score_respects_known_optimization_direction() -> None:
    notebooks = [
        _notebook("a/one", "a", [("LGBM", {})], public_score=0.3),
        _notebook("b/two", "b", [("LGBM", {})], public_score=0.1),
    ]

    lower = compute_code_aggregates(notebooks, optimization_direction="minimize")
    fallback = compute_code_aggregates(notebooks)

    assert lower is not None
    assert fallback is not None
    assert lower.models[0].best_public_score == pytest.approx(0.1)
    assert fallback.models[0].best_public_score == pytest.approx(0.3)


def test_empty_notebook_list_has_no_code_aggregates() -> None:
    assert compute_code_aggregates([]) is None


def _notebook(
    ref: str,
    cluster_id: str,
    models: list[tuple[str, dict[str, str]]],
    *,
    public_score: float | None = None,
) -> NotebookFacts:
    return NotebookFacts(
        ref=ref,
        title=ref,
        public_score=public_score,
        ast_fingerprint=f"fp-{ref}",
        lineage_cluster_id=cluster_id,
        splitters=[],
        models=[
            CodeObservation(name=name, kwargs=kwargs, locator=f"cell_{index}")
            for index, (name, kwargs) in enumerate(models)
        ],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        parse_status="ok",
    )
