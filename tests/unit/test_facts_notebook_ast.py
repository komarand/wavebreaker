from __future__ import annotations

import builtins
import importlib
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.facts.models import CodeObservation, NotebookFacts
from kaggle_researcher.facts.notebook_ast import (
    _cell_source,
    _strip_magics,
    diagnose_scores,
    extract_observations,
    extract_score_observations,
)

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


def test_metric_function_mention_without_cv_context_is_not_declared_cv(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell(
                "from sklearn.metrics import accuracy_score\n"
                "score = accuracy_score(y_true, y_pred)\n"
            )
        ],
    )

    result = extract_observations(notebook_path)

    assert [metric.name for metric in result["metrics"]] == ["accuracy_score"]
    assert result["declared_cv"] == []
    assert result["declared_cv_observations"] == []


def test_validation_map_text_creates_grounded_declared_cv_observation(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell("Validation mAP: 0.8123"),
            _code_cell("value = 1"),
        ],
    )

    result = extract_observations(notebook_path)

    assert result["declared_cv"] == ["0.8123"]
    observation = result["declared_cv_observations"][0]
    assert observation.value == pytest.approx(0.8123)
    assert observation.metric_name == "mAP"
    assert observation.locator == "cell_0"
    assert observation.raw_text == "Validation mAP: 0.8123"


def test_score_extraction_is_open_to_unknown_metric_labels(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell(
                "Validation custom wildlife retrieval quality: 0.8123\n"
                "score: 0.799"
            ),
            _code_cell("value = 1"),
        ],
    )

    result = extract_observations(notebook_path)

    assert [item.value for item in result["score_observations"]] == pytest.approx(
        [0.8123, 0.799]
    )
    assert result["score_observations"][0].metric_raw == (
        "Validation custom wildlife retrieval quality"
    )
    assert result["score_observations"][0].metric_canonical is None
    assert result["score_observations"][1].metric_raw == "score"
    assert result["declared_cv"] == ["0.8123"]


def test_known_score_labels_are_canonicalized_separately(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell(
                "val mAP 0.8123\n"
                "Top-1: 94.2%\n"
                "val_acc=0.91\n"
                "score: 0.88"
            ),
            _code_cell("value = 1"),
        ],
    )

    result = extract_observations(notebook_path)

    observations = result["score_observations"]
    assert [item.metric_raw for item in observations] == [
        "Top-1",
        "val_acc",
        "score",
        "val mAP",
    ]
    assert [item.metric_canonical for item in observations] == [
        "rank-1",
        "accuracy",
        None,
        "mAP",
    ]
    assert observations[0].value == pytest.approx(0.942)


def test_hyperparameter_labels_are_the_only_score_candidates_excluded() -> None:
    observations, candidates_seen, candidates_excluded = extract_score_observations(
        "learning_rate: 0.001\nbatch_size=32\nseed=42\n"
        "optimizer weight_decay: 0.0001\n"
        "custom_distance: 12.4",
        locator="cell_0",
        source="markdown",
    )

    assert candidates_seen == 5
    assert candidates_excluded == 4
    assert len(observations) == 1
    assert observations[0].metric_raw == "custom_distance"
    assert observations[0].value == pytest.approx(12.4)


def test_code_assignments_collect_score_positions_without_treating_all_numbers_as_scores(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell(
                "val_acc = 0.91\n"
                "custom_metric = 12.4\n"
                "score = 0.88\n"
                "learning_rate = 0.001\n"
                "x = 1\n"
            )
        ],
    )

    result = extract_observations(notebook_path)

    observations = result["score_observations"]
    assert [item.metric_raw for item in observations] == [
        "val_acc",
        "custom_metric",
        "score",
    ]
    assert [item.value for item in observations] == pytest.approx([0.91, 12.4, 0.88])
    assert result["score_candidates_seen"] == 5
    assert result["score_candidates_excluded"] == 2
    assert result["declared_cv"] == []


def test_ordinary_prose_numbers_are_not_score_observations() -> None:
    observations, candidates_seen, candidates_excluded = extract_score_observations(
        "Part 1.0\nStep 1.0\nThis creates validation sets because: 1\n"
        "unknown distance: 12.4\nscore 0.88",
        locator="cell_0",
        source="markdown",
    )

    assert candidates_seen == 5
    assert candidates_excluded == 3
    assert [item.metric_raw for item in observations] == [
        "unknown distance",
        "score",
    ]


@pytest.mark.parametrize(
    ("text", "source"),
    [
        ("0.95 Jaguar Re-ID frozen DINOv2", "title"),
        ("0-95-jaguar-re-id-frozen-dinov2", "ref"),
    ],
)
def test_leading_title_or_ref_score_is_collected(
    text: str,
    source: str,
) -> None:
    observations, candidates_seen, candidates_excluded = extract_score_observations(
        text,
        locator=source,
        source=source,  # type: ignore[arg-type]
    )

    assert candidates_seen == 1
    assert candidates_excluded == 0
    assert observations[0].value == pytest.approx(0.95)
    assert observations[0].metric_raw is None
    assert observations[0].source == source


def test_score_diagnostics_keep_raw_and_canonical_counts() -> None:
    observations, candidates_seen, candidates_excluded = extract_score_observations(
        "val mAP: 0.81\ncustom_score: 2.4\nseed: 42",
        locator="cell_0",
        source="markdown",
    )
    notebook = NotebookFacts(
        ref="author/notebook",
        title="Notebook",
        ast_fingerprint="a" * 64,
        lineage_cluster_id="lc_a",
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        score_observations=observations,
        score_candidates_seen=candidates_seen,
        score_candidates_excluded=candidates_excluded,
        parse_status="ok",
    )

    diagnostics = diagnose_scores([notebook])

    assert diagnostics.notebooks_with_score_observations == 1
    assert diagnostics.observations_total == 2
    assert diagnostics.observations_with_canonical_metric == 1
    assert diagnostics.observations_without_canonical_metric == 1
    assert diagnostics.candidates_seen == 3
    assert diagnostics.candidates_excluded == 1


def test_ast_parse_suppresses_only_syntax_warning(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [_code_cell("bad_escape = '\\d'\nvalue = 1")],
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = extract_observations(notebook_path)

    assert result["parse_status"] == "ok"
    assert not any(item.category is SyntaxWarning for item in captured)


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
        "declared_cv_observations": [],
        "score_observations": [],
        "score_candidates_seen": 0,
        "score_candidates_excluded": 0,
        "parse_status": "failed",
    }


def test_invalid_notebook_file_returns_failed_result_without_raising(tmp_path: Path) -> None:
    notebook_path = tmp_path / "invalid.ipynb"
    notebook_path.write_text("not JSON", encoding="utf-8")

    result = extract_observations(notebook_path)

    assert result["parse_status"] == "failed"
    assert all(result[key] == [] for key in _list_result_keys())


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ({"source": "value = 1"}, "value = 1"),
        ({"source": ["value", " = ", 1]}, "value = 1"),
        ({}, ""),
        ({"source": {"unexpected": True}}, "{'unexpected': True}"),
    ],
)
def test_cell_source_handles_json_source_shapes(
    cell: dict[str, object], expected: str
) -> None:
    assert _cell_source(cell) == expected


def test_parser_strips_multiline_magics_and_shell_commands(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [_code_cell("%matplotlib inline\n  !pip install package\nmodel = Ridge()\n")],
    )

    result = extract_observations(notebook_path)

    assert result["parse_status"] == "ok"
    assert [model.name for model in result["models"]] == ["Ridge"]


def test_magic_only_cell_is_parseable(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [_code_cell("%%time\n  %load_ext autoreload\n!pwd\n")],
    )

    assert extract_observations(notebook_path)["parse_status"] == "ok"


def test_strip_magics_preserves_lines_and_string_literal_contents() -> None:
    source = '!pwd\ntext = """first\n%literal\n!literal\n"""\n%time value\n'

    stripped = _strip_magics(source)

    assert len(stripped.splitlines()) == len(source.splitlines())
    assert "%literal" in stripped
    assert "!literal" in stripped
    assert stripped.splitlines()[0] == ""
    assert stripped.splitlines()[-1] == ""


def test_module_imports_when_nbformat_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_nbformat(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("nbformat"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_nbformat)
    module = importlib.import_module("kaggle_researcher.facts.notebook_ast")

    importlib.reload(module)
    module_path = (
        Path(__file__).resolve().parents[2]
        / "kaggle_researcher"
        / "facts"
        / "notebook_ast.py"
    )

    assert "nbformat" not in module_path.read_text(encoding="utf-8")


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
