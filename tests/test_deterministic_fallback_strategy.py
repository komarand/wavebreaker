from __future__ import annotations

from copy import deepcopy

from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.reasoning.deterministic_strategy import (
    StrategyContext,
    compile_competition_strategy,
)
from kaggle_researcher.reasoning.final_synthesizer import (
    build_fallback_final_strategy,
    postprocess_final_strategy_result,
    render_final_strategy,
)


def _hypotheses() -> list[dict[str, str]]:
    return [
        {"hypothesis_id": "schema_001", "category": "schema"},
        {"hypothesis_id": "metric_001", "category": "metric"},
        {"hypothesis_id": "val_001", "category": "validation"},
        {"hypothesis_id": "feature_001", "category": "feature"},
        {"hypothesis_id": "baseline_001", "category": "baseline"},
        {"hypothesis_id": "model_001", "category": "model"},
    ]


def _column(name: str, dtype: str) -> dict[str, object]:
    return {"name": name, "dtype": dtype}


def _titanic_evidence() -> dict[str, object]:
    columns = [
        _column("PassengerId", "Int64"), _column("Survived", "Int64"),
        _column("Pclass", "Int64"), _column("Name", "String"),
        _column("Sex", "String"), _column("Age", "Float64"),
        _column("SibSp", "Int64"), _column("Parch", "Int64"),
        _column("Ticket", "String"), _column("Fare", "Float64"),
        _column("Cabin", "String"), _column("Embarked", "String"),
    ]
    return {
        "metric_evidence": {
            "metric_name": "accuracy", "task_type": "binary_classification",
            "greater_is_better": True, "requires_threshold": True,
            "requires_probabilities": False,
        },
        "validation_evidence": {
            "primary_validation": {"method": "stratified_kfold"},
        },
        "inferred_schema": {
            "target_column": "Survived", "primary_id_column": "PassengerId",
            "train_base_table": "train.csv",
            "tables": [{"path": "train.csv", "columns": columns}],
        },
        "table_profiles": [{
            "path": "train.csv", "n_rows": 891, "columns": columns,
        }],
        "feature_diagnostics": {
            "status": "completed",
            "safe_feature_columns": [
                "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket",
                "Fare", "Cabin", "Embarked",
            ],
            "categorical_feature_diagnostics": {"columns": [
                {"column": "Sex", "feature_value_type": "low_cardinality_categorical"},
                {"column": "Embarked", "feature_value_type": "low_cardinality_categorical"},
                {"column": "Name", "feature_value_type": "mixed_text_code"},
                {"column": "Ticket", "feature_value_type": "high_cardinality_categorical"},
                {"column": "Cabin", "feature_value_type": "mixed_text_code"},
            ]},
            "numeric_feature_diagnostics": {"columns": [
                {"column": "Pclass", "feature_value_type": "ordinal_low_cardinality"},
                {"column": "Fare", "feature_value_type": "continuous", "skew_proxy": 2.1},
            ]},
            "missingness_diagnostics": {"columns": [
                {"column": "Age", "missing_pct": 0.2},
                {"column": "Cabin", "missing_pct": 0.77},
            ]},
        },
        "feature_probe_evidence": [{
            "feature_family": "text_features", "status": "high_potential",
            "leakage_risk": "medium", "evidence": {"columns": ["Name", "Ticket"]},
            "recommendation": "Test structural text summaries.",
        }],
        "baseline_evidence": {
            "status": "completed", "model_type": "LogisticRegression",
            "metric_name": "accuracy", "metric_value": 0.79,
        },
        "baseline_ablation_evidence": {
            "status": "completed",
            "feature_block_findings": [
                {
                    "feature_block": "low_cardinality_categorical",
                    "status": "helped", "delta_metric": 0.012,
                    "stability": "stable", "confidence": "high",
                },
                {
                    "feature_block": "text_code_simple", "status": "unstable",
                    "delta_metric": 0.003, "stability": "unstable",
                    "confidence": "low",
                },
            ],
        },
    }


def _fallback(evidence: dict[str, object]) -> FinalStrategyResult:
    return FinalStrategyResult.model_validate(build_fallback_final_strategy(
        competition_id="titanic-style",
        research_hypotheses=_hypotheses(),
        eda_evidence_pack=evidence,
    ))


def test_titanic_style_fallback_compiles_actual_feature_experiments() -> None:
    result = _fallback(_titanic_evidence())
    metadata = {
        tuple(action.feature_metadata.input_columns): action.feature_metadata
        for action in result.actions if action.feature_metadata
    }
    experiment_ids = {experiment.experiment_id for experiment in result.experiments}

    assert ("Parch", "SibSp") in metadata
    assert ("Name",) in metadata
    assert ("Ticket",) in metadata
    assert ("Cabin",) in metadata
    assert ("Age",) in metadata
    assert ("Fare", "Pclass") in metadata
    age_metadata = next(
        action.feature_metadata for action in result.actions
        if action.feature_metadata and action.action.startswith("Impute `Age`")
    )
    assert age_metadata.fit_scope == "within_fold"
    assert metadata[("Ticket",)].fit_scope == "within_fold"
    assert "fallback_exp_family_size" in experiment_ids
    assert "fallback_exp_is_alone" in experiment_ids
    assert "fallback_exp_title" in experiment_ids
    assert "fallback_exp_ticket_group_size" in experiment_ids
    assert "fallback_exp_cabin_known_and_deck" in experiment_ids
    assert "fallback_exp_age_imputation" in experiment_ids
    assert "fallback_exp_fare_transforms" in experiment_ids
    assert all(experiment.acceptance_rule for experiment in result.experiments)
    assert all(experiment.evidence_refs for experiment in result.experiments)


def test_compiler_never_emits_titanic_features_when_columns_are_absent() -> None:
    evidence = _titanic_evidence()
    evidence["inferred_schema"] = {
        "target_column": "target",
        "tables": [{"path": "train.csv", "columns": [
            _column("target", "Int64"), _column("x1", "Float64"),
            _column("segment", "String"),
        ]}],
    }
    evidence["feature_diagnostics"] = {
        "safe_feature_columns": ["x1", "segment"],
        "categorical_feature_diagnostics": {"columns": [{"column": "segment"}]},
        "missingness_diagnostics": {"columns": []},
    }
    evidence["feature_probe_evidence"] = []
    evidence["baseline_ablation_evidence"] = {}
    evidence["table_profiles"] = [{
        "path": "train.csv", "n_rows": 1000,
        "columns": [_column("target", "Int64"), _column("x1", "Float64"),
                    _column("segment", "String")],
    }]

    result = _fallback(evidence)
    text = " ".join(action.action for action in result.actions).casefold()

    for invented in ("family_size", "is_alone", "title", "ticket_group", "cabin", "age"):
        assert invented not in text
    assert all(
        column in {"x1", "segment"}
        for action in result.actions if action.feature_metadata
        for column in action.feature_metadata.input_columns
    )


def test_regression_fallback_uses_regression_family_without_threshold_experiment() -> None:
    evidence = _titanic_evidence()
    evidence["metric_evidence"] = {
        "metric_name": "rmse", "task_type": "regression",
        "greater_is_better": False, "requires_threshold": False,
    }
    evidence["baseline_evidence"] = {"status": "skipped", "reason": "disabled"}

    result = _fallback(evidence)

    assert all("threshold" not in item.experiment_id for item in result.experiments)
    assert all(
        "Regressor" in item.model_family or "Regression" in item.model_family
        for item in result.experiments
    )
    assert all(
        "decrease" in item.acceptance_rule
        for item in result.experiments if item.feature_inputs
    )


def test_rmsle_target_transform_requires_nonnegative_target_evidence() -> None:
    evidence = _titanic_evidence()
    evidence["metric_evidence"] = {
        "metric_name": "rmsle", "task_type": "regression",
        "greater_is_better": False, "requires_threshold": False,
    }
    evidence["target_diagnostics"] = {
        "target_column": "Survived", "distribution": {"min": 0.0},
    }

    supported = _fallback(evidence)
    evidence["target_diagnostics"]["distribution"]["min"] = -1.0
    unsupported = _fallback(evidence)

    assert any(
        item.experiment_id == "fallback_exp_target_log1p"
        for item in supported.experiments
    )
    assert not any(
        item.experiment_id == "fallback_exp_target_log1p"
        for item in unsupported.experiments
    )


def test_calibration_experiment_is_oof_only_when_metric_requires_it() -> None:
    evidence = _titanic_evidence()
    evidence["metric_evidence"]["requires_calibration"] = True

    result = _fallback(evidence)
    calibration = next(
        item for item in result.experiments
        if item.experiment_id == "fallback_exp_calibration"
    )

    assert calibration.fit_scope == "oof_only"
    assert "metric_evidence.requires_calibration" in calibration.evidence_refs
    assert "fold" in calibration.change.casefold()


def test_small_dataset_multiple_seed_advice_requires_policy_support() -> None:
    evidence = _titanic_evidence()
    default_result = _fallback(evidence)
    evidence["validation_evidence"]["multiple_seed_diagnostics_supported"] = True
    approved_result = _fallback(evidence)

    default_text = " ".join(action.action for action in default_result.actions)
    approved_text = " ".join(action.action for action in approved_result.actions)
    assert "single fold split as provisional" in default_text
    assert "project-approved repeated or multiple-seed diagnostic" not in default_text
    assert "project-approved repeated or multiple-seed diagnostic" in approved_text


def test_secondary_aggregation_requires_validated_relationship() -> None:
    evidence = _titanic_evidence()
    without_relationship = _fallback(evidence)
    evidence["relationship_evidence"] = {
        "relationships": [{
            "table": "transactions.csv", "selected_join_key": "PassengerId",
            "relationship_type": "one_to_many", "requires_aggregation": True,
            "confidence": "high",
        }],
    }
    with_relationship = _fallback(evidence)

    assert not any(
        item.experiment_id == "fallback_exp_secondary_transactions_csv"
        for item in without_relationship.experiments
    )
    experiment = next(
        item for item in with_relationship.experiments
        if item.experiment_id == "fallback_exp_secondary_transactions_csv"
    )
    assert experiment.feature_inputs == ["PassengerId"]
    assert experiment.fit_scope == "within_fold"
    assert "relationship_evidence.relationships[0]" in experiment.evidence_refs


def test_ablation_delta_and_conflict_are_preserved_without_score_promise() -> None:
    result = _fallback(_titanic_evidence())
    actions = "\n".join(f"{item.action} {item.reason}" for item in result.actions)
    title_action = next(
        item for item in result.actions if "`title`" in item.action
    )

    assert "recorded delta=0.012" in actions
    assert "recorded delta=0.003" in actions
    assert title_action.priority == "P2"
    assert title_action.confidence == "low"
    assert "score promise" in actions


def test_rule_output_and_experiment_order_are_deterministic() -> None:
    evidence = _titanic_evidence()
    context = StrategyContext.from_evidence(
        competition_id="titanic-style", evidence_pack=evidence,
        task_type="binary_classification", metric_name="accuracy",
        validation_strategy="stratified_kfold",
    )

    first = compile_competition_strategy(context)
    second = compile_competition_strategy(context)

    assert first == second
    assert [item.rule_id for item in first.actions] == [
        item.rule_id for item in second.actions
    ]
    assert [item.experiment_id for item in first.experiments] == [
        item.experiment_id for item in second.experiments
    ]


def test_first_48_hours_orders_baseline_before_features_and_threshold() -> None:
    result = _fallback(_titanic_evidence())
    first_48 = next(section for section in result.sections if section.section_id == "first_48_hours")
    positions = {
        experiment_id: index
        for index, block in enumerate(first_48.time_blocks)
        for experiment_id in block.experiment_ids
    }

    assert positions["fallback_exp_reproduce_baseline"] <= positions["fallback_exp_family_size"]
    assert positions["fallback_exp_family_size"] <= positions["fallback_exp_model_family_comparison"]
    assert positions["fallback_exp_model_family_comparison"] <= positions["fallback_exp_threshold"]
    assert positions["fallback_exp_family_size"] <= positions["fallback_exp_threshold"]
    final_block_actions = {
        action.action_id: action.action for action in result.actions
        if action.action_id in first_48.time_blocks[-1].action_ids
    }
    assert any("submission" in action.casefold() for action in final_block_actions.values())
    markdown = render_final_strategy(result)
    assert "Feature inputs:" in markdown
    assert "Fit scope:" in markdown


def test_sparse_compiler_has_no_feature_or_model_outputs() -> None:
    context = StrategyContext.from_evidence(
        competition_id="sparse", evidence_pack={}, task_type="unknown",
        metric_name="unknown", validation_strategy="",
    )

    compilation = compile_competition_strategy(context)

    assert compilation.actions == ()
    assert compilation.experiments == ()


def test_compilation_does_not_mutate_evidence() -> None:
    evidence = _titanic_evidence()
    before = deepcopy(evidence)
    context = StrategyContext.from_evidence(
        competition_id="titanic-style", evidence_pack=evidence,
        task_type="binary_classification", metric_name="accuracy",
        validation_strategy="stratified_kfold",
    )

    compile_competition_strategy(context)

    assert evidence == before


def test_postprocessing_keeps_experiment_evidence_aligned_with_actions() -> None:
    evidence = _titanic_evidence()
    result = _fallback(evidence)

    finalized = postprocess_final_strategy_result(
        result,
        eda_evidence_pack=evidence,
    )
    action_by_experiment = {
        action.experiment_id: action for action in finalized.actions
        if action.experiment_id
    }

    assert finalized.experiments
    for experiment in finalized.experiments:
        action = action_by_experiment[experiment.experiment_id]
        assert experiment.evidence_refs == action.evidence_refs
        assert experiment.related_hypothesis_ids == action.related_hypothesis_ids
