from __future__ import annotations

import pytest

from kaggle_researcher import wave
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


def test_kwargs_distribution_summarizes_numeric_values_by_cluster() -> None:
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
    assert [item.model_dump() for item in aggregates.models[0].kwargs_distribution] == [
        {
            "key": "n_estimators",
            "cluster_count": 3,
            "median": "300",
            "minimum": "100",
            "maximum": "500",
            "distinct_values": 3,
            "is_integer": True,
        },
        {
            "key": "learning_rate",
            "cluster_count": 2,
            "median": "0.15",
            "minimum": "0.1",
            "maximum": "0.2",
            "distinct_values": 2,
            "is_integer": False,
        },
    ]


def test_integer_kwargs_use_lower_median_instead_of_fractional_value() -> None:
    notebooks = [
        _notebook("a/one", "a", [("LGBMClassifier", {"max_depth": "4"})]),
        _notebook("b/two", "b", [("LGBMClassifier", {"max_depth": "5"})]),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    distribution = aggregates.models[0].kwargs_distribution[0]
    assert distribution.median == "4"
    assert distribution.minimum == "4"
    assert distribution.maximum == "5"
    assert distribution.is_integer is True


def test_fractional_kwargs_keep_median_and_range_without_exponents() -> None:
    notebooks = [
        _notebook("a/one", "a", [("LGBM", {"learning_rate": "0.01"})]),
        _notebook("b/two", "b", [("LGBM", {"learning_rate": "0.05"})]),
        _notebook("c/three", "c", [("LGBM", {"learning_rate": "0.1"})]),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    distribution = aggregates.models[0].kwargs_distribution[0]
    assert distribution.median == "0.05"
    assert distribution.minimum == "0.01"
    assert distribution.maximum == "0.1"
    assert distribution.is_integer is False


def test_integral_float_is_serialized_as_an_integer() -> None:
    notebooks = [
        _notebook("a/one", "a", [("LGBM", {"n_estimators": "1000.0"})]),
        _notebook("b/two", "b", [("LGBM", {"n_estimators": "1000"})]),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    distribution = aggregates.models[0].kwargs_distribution[0]
    assert distribution.median == "1000"
    assert distribution.minimum == "1000"
    assert distribution.maximum == "1000"
    assert distribution.distinct_values == 1
    assert distribution.is_integer is True


def test_nonnumeric_kwargs_use_mode_without_numeric_range() -> None:
    notebooks = [
        _notebook("a/one", "a", [("Model", {"solver": "auto"})]),
        _notebook("b/two", "b", [("Model", {"solver": "auto"})]),
        _notebook("c/three", "c", [("Model", {"solver": "manual"})]),
    ]

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    distribution = aggregates.models[0].kwargs_distribution[0]
    assert distribution.median == "auto"
    assert distribution.minimum is None
    assert distribution.maximum is None
    assert distribution.distinct_values == 2
    assert distribution.is_integer is False


def test_forks_contribute_one_cluster_representative_value() -> None:
    notebooks = [
        _notebook(
            f"author/fork-{index}",
            "shared",
            [("Model", {"depth": str(index)})],
        )
        for index in range(1, 21)
    ]
    notebooks.append(
        _notebook("author/independent", "independent", [("Model", {"depth": "30"})])
    )

    aggregates = compute_code_aggregates(notebooks)

    assert aggregates is not None
    distribution = aggregates.models[0].kwargs_distribution[0]
    assert distribution.cluster_count == 2
    assert distribution.median == "10"
    assert distribution.minimum == "10"
    assert distribution.maximum == "30"
    assert distribution.distinct_values == 2


def test_kwargs_basis_is_explicit_and_typical_kwargs_is_absent() -> None:
    aggregates = compute_code_aggregates(
        [_notebook("a/one", "a", [("LGBM", {})])]
    )

    assert aggregates is not None
    payload = aggregates.models[0].model_dump(mode="json")
    assert payload["kwargs_basis"] == "cluster_median_of_public_notebooks"
    assert payload["kwargs_distribution"] == []
    assert "typical_kwargs" not in payload


def test_wave_prints_numeric_ranges_and_nonnumeric_cluster_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    aggregates = compute_code_aggregates(
        [
            _notebook(
                "a/one",
                "a",
                [("LGBMClassifier", {"max_depth": "4", "boosting": "auto"})],
            ),
            _notebook(
                "b/two",
                "b",
                [("LGBMClassifier", {"max_depth": "5", "boosting": "auto"})],
            ),
        ]
    )
    assert aggregates is not None

    wave._print_model_kwargs_distributions(aggregates)

    output = capsys.readouterr().out
    assert "max_depth 4 (4–5, 2 clusters)" in output
    assert "boosting auto (2 clusters)" in output
    assert "auto–" not in output


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
