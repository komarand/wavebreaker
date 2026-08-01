from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher.facts.models import CodeObservation
from kaggle_researcher.facts.notebook_ast import extract_observations


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "facts"
GROUP_FIXTURE = FIXTURE_DIR / "notebook_groupkfold.ipynb"
TIMESERIES_FIXTURE = FIXTURE_DIR / "notebook_timeseries.ipynb"
BROKEN_FIXTURE = FIXTURE_DIR / "notebook_broken_cell.ipynb"


def test_groupkfold_fixture_extracts_splitter_model_metric_features_and_cv() -> None:
    result = extract_observations(GROUP_FIXTURE)

    assert result["parse_status"] == "ok"
    assert result["declared_cv"] == ["0.7841"]
    assert result["splitters"] == [
        CodeObservation(
            name="StratifiedGroupKFold",
            kwargs={
                "n_splits": "5",
                "shuffle": "True",
                "groups": "customer_id",
            },
            locator="cell_0",
        )
    ]
    assert result["models"] == [
        CodeObservation(
            name="LGBMClassifier",
            kwargs={
                "n_estimators": "500",
                "learning_rate": "0.05",
                "num_leaves": "31",
            },
            locator="cell_0",
        )
    ]
    assert [item.name for item in result["metrics"]] == ["roc_auc_score"]
    assert {item.name for item in result["feature_ops"]} == {"groupby", "agg"}
    assert all(item.locator == "cell_1" for item in result["feature_ops"])


def test_timeseries_fixture_extracts_splitter_and_shift() -> None:
    result = extract_observations(TIMESERIES_FIXTURE)

    assert result["parse_status"] == "ok"
    assert result["splitters"] == [
        CodeObservation(
            name="TimeSeriesSplit",
            kwargs={"n_splits": "6"},
            locator="cell_0",
        )
    ]
    assert result["feature_ops"] == [
        CodeObservation(name="shift", kwargs={}, locator="cell_0")
    ]


def test_broken_cell_fixture_is_partial_and_keeps_valid_observations() -> None:
    result = extract_observations(BROKEN_FIXTURE)

    assert result["parse_status"] == "partial"
    assert result["models"] == [
        CodeObservation(
            name="XGBClassifier",
            kwargs={"n_estimators": "200", "max_depth": "7"},
            locator="cell_1",
        )
    ]
    assert result["declared_cv"] == ["0.7315"]


def test_notebook_with_no_parseable_code_returns_empty_failed_result(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell("CV 0.9999"),
            _code_cell("if True print('broken')"),
        ],
    )

    result = extract_observations(notebook_path)

    assert result == {
        "splitters": [],
        "models": [],
        "metrics": [],
        "feature_ops": [],
        "declared_cv": [],
        "parse_status": "failed",
    }


def test_invalid_notebook_file_returns_failed_result_without_raising(tmp_path: Path) -> None:
    notebook_path = tmp_path / "invalid.ipynb"
    notebook_path.write_text("not JSON", encoding="utf-8")

    result = extract_observations(notebook_path)

    assert result["parse_status"] == "failed"
    assert all(result[key] == [] for key in _list_result_keys())


def test_keyword_values_capture_names_literals_short_source_and_long_marker(
    tmp_path: Path,
) -> None:
    long_expression = "build_configuration(" + ", ".join(f"value_{i}" for i in range(20)) + ")"
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell(
                "splitter = KFold(\n"
                "    n_splits=settings.n_splits,\n"
                "    shuffle=use_shuffle,\n"
                "    random_state=seed_factory(42),\n"
                f"    groups={long_expression},\n"
                ")\n"
                "model = Ridge(alpha=1.0, max_depth=None)\n"
            )
        ],
    )

    result = extract_observations(notebook_path)

    assert result["splitters"][0].kwargs == {
        "n_splits": "settings.n_splits",
        "shuffle": "use_shuffle",
        "random_state": "seed_factory(42)",
        "groups": "<expr>",
    }
    assert result["models"][0].kwargs == {"max_depth": "None"}


def test_target_names_match_exactly_and_attribute_calls_are_supported(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell(
                "splitter = custom.CustomKFold(n_splits=5)\n"
                "score = sklearn.metrics.f1_score(y, pred, average='macro')\n"
                "encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')\n"
            )
        ],
    )

    result = extract_observations(notebook_path)

    assert result["splitters"] == []
    assert result["metrics"] == [
        CodeObservation(
            name="f1_score",
            kwargs={"average": "macro"},
            locator="cell_0",
        )
    ]
    assert result["feature_ops"] == [
        CodeObservation(
            name="OneHotEncoder",
            kwargs={"handle_unknown": "ignore"},
            locator="cell_0",
        )
    ]


def test_declared_cv_from_code_strings_is_deduplicated_in_source_order(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell("Local score 0.7001"),
            _code_cell(
                "first = 'OOF metric 0.7112'\n"
                "duplicate = 'fold result 0.7001'\n"
                "second = 'CV=0.7223'\n"
            ),
        ],
    )

    result = extract_observations(notebook_path)

    assert result["declared_cv"] == ["0.7001", "0.7112", "0.7223"]


@pytest.mark.parametrize("fixture", [GROUP_FIXTURE, TIMESERIES_FIXTURE, BROKEN_FIXTURE])
def test_notebook_fixtures_have_valid_v4_structure(fixture: Path) -> None:
    notebook = json.loads(fixture.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["metadata"], dict)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_module_contains_no_notebook_execution_primitives() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "kaggle_researcher"
        / "facts"
        / "notebook_ast.py"
    )
    source = module_path.read_text(encoding="utf-8")
    forbidden = ("ex" + "ec(", "ev" + "al(", "run" + "py")

    assert all(token not in source for token in forbidden)


def _write_notebook(tmp_path: Path, cells: list[dict[str, object]]) -> Path:
    notebook_path = tmp_path / "fixture.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return notebook_path


def _code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _list_result_keys() -> tuple[str, ...]:
    return ("splitters", "models", "metrics", "feature_ops", "declared_cv")
