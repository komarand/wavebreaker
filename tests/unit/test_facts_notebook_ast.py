from __future__ import annotations

import builtins
import importlib
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.facts.models import CodeObservation, NotebookFacts, ScoreObservation
from kaggle_researcher.facts.collect import aggregate_dataset_references
from kaggle_researcher.facts.notebook_ast import (
    _cell_source,
    _strip_magics,
    ast_fingerprint,
    diagnose_scores,
    extract_observations,
    extract_score_observations,
    recanonicalize_score_observations,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "facts"
GROUP_FIXTURE = FIXTURE_DIR / "notebook_groupkfold.ipynb"
TIMESERIES_FIXTURE = FIXTURE_DIR / "notebook_timeseries.ipynb"
BROKEN_FIXTURE = FIXTURE_DIR / "notebook_broken_cell.ipynb"
STAR_PARAMS_FIXTURE = FIXTURE_DIR / "notebook_star_params.ipynb"


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
            kwargs_resolved_from={
                "n_splits": "direct",
                "shuffle": "direct",
                "groups": "direct",
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
            kwargs_resolved_from={
                "n_estimators": "direct",
                "learning_rate": "direct",
                "num_leaves": "direct",
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
            kwargs_resolved_from={"n_splits": "direct"},
            locator="cell_0",
        )
    ]
    assert result["feature_ops"] == [
        CodeObservation(name="shift", kwargs={}, locator="cell_0")
    ]


def test_star_kwargs_fixture_resolves_dict_literal_and_tracks_sources() -> None:
    result = extract_observations(STAR_PARAMS_FIXTURE)

    assert result["models"] == [
        CodeObservation(
            name="LGBMClassifier",
            kwargs={
                "n_estimators": "3000",
                "learning_rate": "0.02",
                "num_leaves": "64",
            },
            kwargs_resolved_from={
                "n_estimators": "dict_literal",
                "learning_rate": "dict_literal",
                "num_leaves": "dict_literal",
            },
            locator="cell_1",
        ),
        CodeObservation(
            name="XGBClassifier",
            kwargs={
                "n_estimators": "500",
                "learning_rate": "0.02",
                "num_leaves": "64",
            },
            kwargs_resolved_from={
                "n_estimators": "direct",
                "learning_rate": "dict_literal",
                "num_leaves": "dict_literal",
            },
            locator="cell_2",
        ),
        CodeObservation(
            name="CatBoostClassifier",
            kwargs={},
            kwargs_resolved_from={},
            locator="cell_3",
        ),
    ]


def test_star_kwargs_uses_last_dict_assignment_across_notebook(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell('params = {"n_estimators": 100}\n'),
            _code_cell('params = {"n_estimators": 500}\n'),
            _code_cell("model = LGBMClassifier(**params)\n"),
        ],
    )

    result = extract_observations(notebook_path)

    assert result["models"][0].kwargs == {"n_estimators": "500"}
    assert result["models"][0].kwargs_resolved_from == {
        "n_estimators": "dict_literal"
    }


def test_star_kwargs_allows_one_level_merge_but_skips_second_level(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell(
                'base = {"n_estimators": 100, "learning_rate": 0.1}\n'
                'params = {**base, "learning_rate": 0.02}\n'
                'reverse = {"learning_rate": 0.03, **base}\n'
                'nested = {**params, "max_depth": 7}\n'
                "good = LGBMClassifier(**params)\n"
                "reverse_order = LGBMClassifier(**reverse)\n"
                "skipped = XGBClassifier(**nested)\n"
            )
        ],
    )

    result = extract_observations(notebook_path)

    assert result["models"][0].kwargs == {
        "n_estimators": "100",
        "learning_rate": "0.02",
    }
    assert result["models"][1].kwargs == {
        "n_estimators": "100",
        "learning_rate": "0.03",
    }
    assert result["models"][2].kwargs == {}


def test_star_kwargs_skips_non_string_keys_and_preserves_expressions(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _code_cell(
                'bad = {1: "value"}\n'
                'params = {"n_estimators": N * 2}\n'
                "bad_model = LGBMClassifier(**bad)\n"
                "good_model = LGBMClassifier(**params)\n"
            )
        ],
    )

    result = extract_observations(notebook_path)

    assert result["models"][0].kwargs == {}
    assert result["models"][1].kwargs == {"n_estimators": "N * 2"}


def test_star_kwargs_fixture_fingerprint_is_stable() -> None:
    assert ast_fingerprint(STAR_PARAMS_FIXTURE) == (
        "37f73c205ebdb73c2df6577220b4437e172d1441a22ff7acc453d6f52ff55f15"
    )


def test_broken_cell_fixture_is_partial_and_keeps_valid_observations() -> None:
    result = extract_observations(BROKEN_FIXTURE)

    assert result["parse_status"] == "partial"
    assert result["models"] == [
        CodeObservation(
            name="XGBClassifier",
            kwargs={"n_estimators": "200", "max_depth": "7"},
            kwargs_resolved_from={
                "n_estimators": "direct",
                "max_depth": "direct",
            },
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
    assert result["score_observations"][0].plausible is False
    assert result["score_observations"][0].implausible_reason == "label_too_long"
    assert result["declared_cv"] == []


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
    assert len(observations) == 5
    assert sum(not observation.plausible for observation in observations) == 4
    assert {observation.implausible_reason for observation in observations[:-1]} == {
        "excluded_label"
    }
    assert observations[-1].metric_raw == "custom_distance"
    assert observations[-1].value == pytest.approx(12.4)


def test_optimizer_tolerance_is_retained_as_implausible_observation(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [_code_cell("tol=1e-05")],
    )
    result = extract_observations(notebook_path, metric_hints=["Roc Auc Score"])
    observations = result["score_observations"]

    assert len(observations) == 1
    assert observations[0].plausible is False
    assert observations[0].implausible_reason == "excluded_label"


def test_integer_bounded_metric_value_uses_integer_reason() -> None:
    observations, _, _ = extract_score_observations(
        "ROC AUC: 100.0",
        locator="cell_0",
        source="markdown",
    )

    assert len(observations) == 1
    assert observations[0].plausible is False
    assert observations[0].implausible_reason == "value_is_integer"
    assert observations[0].metric_canonical == "roc_auc"


def test_bounded_metric_percent_is_converted_and_remains_plausible() -> None:
    observations, _, _ = extract_score_observations(
        "ROC AUC: 95.2%",
        locator="cell_0",
        source="markdown",
    )

    assert len(observations) == 1
    assert observations[0].value == pytest.approx(0.952)
    assert observations[0].metric_canonical == "roc_auc"
    assert observations[0].plausible is True


def test_long_metric_label_is_retained_as_implausible_observation() -> None:
    observations, _, _ = extract_score_observations(
        "Fold-safe target encoding of exact values: 0.001",
        locator="cell_0",
        source="markdown",
    )

    assert len(observations) == 1
    assert observations[0].plausible is False
    assert observations[0].implausible_reason == "label_too_long"


def test_unbounded_rmse_value_remains_plausible() -> None:
    observations, _, _ = extract_score_observations(
        "RMSE: 12.5",
        locator="cell_0",
        source="markdown",
    )

    assert len(observations) == 1
    assert observations[0].metric_canonical == "rmse"
    assert observations[0].plausible is True


@pytest.mark.parametrize(
    ("text", "expected_value"),
    [
        ("VALIDATION_SPLITS: 5", 5.0),
        ("MSE: 2", 2.0),
        ("MSE: 2021", 2021.0),
    ],
)
def test_integer_score_candidates_at_least_one_are_implausible(
    text: str,
    expected_value: float,
) -> None:
    observations, _, _ = extract_score_observations(
        text,
        locator="cell_0",
        source="markdown",
    )

    assert len(observations) == 1
    assert observations[0].value == expected_value
    assert observations[0].plausible is False
    assert observations[0].implausible_reason == "value_is_integer"


@pytest.mark.parametrize("value", [0.0, 0.15479, 0.999])
def test_values_below_one_are_not_rejected_as_integers(value: float) -> None:
    observations, _, _ = extract_score_observations(
        f"MSE: {value}",
        locator="cell_0",
        source="markdown",
    )

    assert len(observations) == 1
    assert observations[0].plausible is True
    assert observations[0].implausible_reason is None


@pytest.mark.parametrize(
    ("text", "metric_raw", "metric_canonical", "value"),
    [
        ("OOF RMSE 1.234", "OOF RMSE", "rmse", 1.234),
        ("Validation MAE 12.5", "Validation MAE", "mae", 12.5),
        ("Fold 3 logloss: 0.042", "Fold 3 logloss", "log_loss", 0.042),
        ("Epoch 7 val_mAP: 0.941", "Epoch 7 val_mAP", "mAP", 0.941),
        (
            "laplace_log_likelihood 2.718",
            "laplace_log_likelihood",
            "Laplace Log Likelihood",
            2.718,
        ),
        ("custom_metric -0.31", "custom_metric", None, -0.31),
    ],
)
def test_domain_agnostic_score_positions_support_unbounded_signed_values(
    text: str,
    metric_raw: str,
    metric_canonical: str | None,
    value: float,
) -> None:
    observations, candidates_seen, candidates_excluded = extract_score_observations(
        text,
        locator="cell_0",
        source="markdown",
        metric_hints=("Laplace Log Likelihood",),
    )

    assert candidates_seen == 1
    assert candidates_excluded == 0
    assert len(observations) == 1
    assert observations[0].metric_raw == metric_raw
    assert observations[0].metric_canonical == metric_canonical
    assert observations[0].value == pytest.approx(value)


def test_fold_and_epoch_context_do_not_block_the_following_metric() -> None:
    observations, candidates_seen, candidates_excluded = extract_score_observations(
        "Fold 3 logloss: 0.042\nEpoch 7 val_mAP: 0.941",
        locator="cell_0",
        source="markdown",
    )

    assert candidates_seen == 2
    assert candidates_excluded == 0
    assert [item.metric_canonical for item in observations] == ["log_loss", "mAP"]


def test_score_canonicalization_uses_metrics_from_the_whole_notebook_corpus() -> None:
    observations, _, _ = extract_score_observations(
        "custom_quality 12.5",
        locator="cell_0",
        source="markdown",
    )
    source_notebook = _notebook_facts(
        ref="author/source",
        metrics=[],
        score_observations=observations,
    )
    metric_notebook = _notebook_facts(
        ref="author/metric",
        metrics=[CodeObservation(name="custom_quality", kwargs={}, locator="cell_1")],
        score_observations=[],
    )

    canonicalized = recanonicalize_score_observations(
        [source_notebook, metric_notebook],
        competition_metric_name=None,
    )

    assert observations[0].metric_canonical is None
    assert canonicalized[0].score_observations[0].metric_canonical == "custom_quality"


def test_competition_metric_canonicalizes_only_generic_score_labels() -> None:
    observations = [
        ScoreObservation(
            value=value,
            value_raw=str(value),
            metric_raw=metric_raw,
            metric_canonical=None,
            locator=source,
            raw_text=f"{metric_raw or 'encoded title'} {value}",
            source=source,
            source_kind=source,
            split="unknown",
        )
        for value, metric_raw, source in (
            (0.91, "score", "markdown"),
            (0.90, "Public LB", "markdown"),
            (0.89, None, "title"),
            (0.001, "tol", "markdown"),
        )
    ]
    notebook = _notebook_facts(
        ref="author/generic-scores",
        metrics=[],
        score_observations=observations,
    )

    canonicalized = recanonicalize_score_observations(
        [notebook],
        competition_metric_name="Roc Auc Score",
    )

    assert [
        observation.metric_canonical
        for observation in canonicalized[0].score_observations
    ] == ["roc_auc", "roc_auc", "roc_auc", None]


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
        "learning_rate",
    ]
    assert observations[-1].plausible is False
    assert observations[-1].implausible_reason == "excluded_label"
    assert [item.value for item in observations] == pytest.approx(
        [0.91, 12.4, 0.88, 0.001]
    )
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
    assert diagnostics.implausible_observations == {"excluded_label": 1}


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
        "dataset_paths": [],
        "parse_status": "failed",
    }


def test_dataset_paths_are_extracted_from_markdown_and_code_strings(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell(
                "External: /kaggle/input/some-lib/oof/train.csv\n"
                "Current: /kaggle/input/current-comp/train.csv"
            ),
            _code_cell(
                'first = "/kaggle/input/external-features/data.csv"\n'
                'duplicate = "/kaggle/input/some-lib/test.csv"\n'
            ),
        ],
    )

    result = extract_observations(notebook_path, competition_id="CURRENT-COMP")

    assert result["dataset_paths"] == ["some-lib", "external-features"]


def test_dataset_paths_reject_path_segments_numeric_and_short_slugs(
    tmp_path: Path,
) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [
            _markdown_cell(
                "/kaggle/input/competitions/foo/train.csv\n"
                "/kaggle/input/notebooks/example/data.csv\n"
                "/kaggle/input/12345/data.csv\n"
                "/kaggle/input/ab/data.csv\n"
                "/kaggle/input/0-926-base-model/weights.pt\n"
                "/kaggle/input/current-comp/train.csv"
            ),
            _code_cell("value = 1\n"),
        ],
    )

    result = extract_observations(notebook_path, competition_id="current-comp")

    assert result["dataset_paths"] == ["0-926-base-model"]


def test_dataset_reference_aggregation_counts_notebooks_and_lineages() -> None:
    first = _notebook_fact("author/first", "lc_shared", ["common-lib"])
    second = _notebook_fact("author/second", "lc_shared", ["common-lib", "broad-lib"])
    third = _notebook_fact("author/third", "lc_other", ["broad-lib"])

    references = aggregate_dataset_references([first, second, third])

    assert [item.slug for item in references] == ["broad-lib", "common-lib"]
    assert references[0].cluster_count == 2
    assert references[0].reference_count == 2
    assert references[1].cluster_count == 1
    assert references[1].reference_count == 2
    assert references[1].raw_path == "/kaggle/input/common-lib/"


def test_dataset_extraction_does_not_change_ast_fingerprint(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path,
        [_code_cell('path = "/kaggle/input/some-lib/data.csv"\n')],
    )
    before = ast_fingerprint(notebook_path)

    extract_observations(notebook_path, competition_id="current-comp")

    assert ast_fingerprint(notebook_path) == before


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
            kwargs_resolved_from={"average": "direct"},
            locator="cell_0",
        )
    ]
    assert result["feature_ops"] == [
        CodeObservation(
            name="OneHotEncoder",
            kwargs={"handle_unknown": "ignore"},
            kwargs_resolved_from={"handle_unknown": "direct"},
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


@pytest.mark.parametrize(
    "fixture",
    [GROUP_FIXTURE, TIMESERIES_FIXTURE, BROKEN_FIXTURE, STAR_PARAMS_FIXTURE],
)
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


def _notebook_facts(
    *,
    ref: str,
    metrics: list[CodeObservation],
    score_observations: list[Any],
) -> NotebookFacts:
    return NotebookFacts(
        ref=ref,
        title=ref,
        ast_fingerprint="a" * 64,
        lineage_cluster_id="lc_a",
        splitters=[],
        models=[],
        metrics=metrics,
        feature_ops=[],
        declared_cv=[],
        score_observations=score_observations,
        parse_status="ok",
    )


def _notebook_fact(
    ref: str,
    lineage_cluster_id: str,
    dataset_paths: list[str],
) -> NotebookFacts:
    return NotebookFacts(
        ref=ref,
        title=ref,
        ast_fingerprint="a" * 64,
        lineage_cluster_id=lineage_cluster_id,
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        dataset_paths=dataset_paths,
        parse_status="ok",
    )


def _list_result_keys() -> tuple[str, ...]:
    return ("splitters", "models", "metrics", "feature_ops", "declared_cv")
