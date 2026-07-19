from __future__ import annotations

import pytest

from kaggle_researcher.contracts.experiments import (
    CrossNamespaceReferenceError,
    build_experiment_registry,
    reference_registries,
    repair_final_experiment_references,
    validate_final_strategy_namespaces,
)
from kaggle_researcher.reasoning.experiment_planner import _assign_and_validate_experiment_ids
from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult
from kaggle_researcher.reasoning.skeptical_reviewer import _validate_experiment_decisions
from kaggle_researcher.schemas import ExperimentItem, ReviewResult


pytestmark = pytest.mark.contract


def _experiment(
    experiment_id: str,
    *hypothesis_ids: str,
) -> ExperimentItem:
    return ExperimentItem(
        experiment_id=experiment_id,
        source_hypothesis_ids=list(hypothesis_ids),
        priority="P1",
        experiment=f"Run {experiment_id}",
        why="Test one evidence-backed change.",
        cost="low",
        expected_gain="diagnostic",
        risk="Fold variance may hide the effect.",
        evidence_ids=["feature_diagnostics"],
    )


def _review(*, approved=(), rejected=(), reviewed=()) -> ReviewResult:
    return ReviewResult(
        confidence="medium",
        approved_experiment_ids=list(approved),
        rejected_experiment_ids=list(rejected),
        reviewed_experiment_ids=list(reviewed),
    )


def _strategy(experiment_ids: list[str], hypothesis_ids: list[str]) -> FinalStrategyResult:
    return FinalStrategyResult.model_validate({
        "competition_id": "demo",
        "synthesis_status": "llm_success",
        "llm_output_valid": True,
        "repair_attempted": False,
        "repair_succeeded": False,
        "fallback_used": False,
        "synthesis_diagnostics_path": None,
        "actions": [{
            "priority": "P1",
            "action": "Run the reviewed encoding experiment.",
            "reason": "It tests the EDA hypothesis under fixed folds.",
            "evidence_refs": ["feature_diagnostics"],
            "experiment_ids": experiment_ids,
            "hypothesis_ids": hypothesis_ids,
        }],
    })


def _issues(strategy: FinalStrategyResult, registry, hypothesis_ids: set[str]):
    return validate_final_strategy_namespaces(
        strategy,
        reference_registries(registry, hypothesis_ids=hypothesis_ids),
    )


def test_canonical_approved_experiment_reference_is_accepted() -> None:
    registry = build_experiment_registry(
        [_experiment("exp_001", "eda_hypothesis_001")],
        _review(approved=["exp_001"], reviewed=["exp_001"]),
    )

    assert _issues(
        _strategy(["exp_001"], ["eda_hypothesis_001"]),
        registry,
        {"eda_hypothesis_001"},
    ) == []


def test_hypothesis_in_experiment_field_reports_actual_namespace() -> None:
    registry = build_experiment_registry(
        [_experiment("exp_001", "eda_hypothesis_001")],
        _review(approved=["exp_001"]),
    )
    issues = _issues(
        _strategy(["eda_hypothesis_001"], ["eda_hypothesis_001"]),
        registry,
        {"eda_hypothesis_001"},
    )

    assert issues[0].expected_namespace == "approved_experiment"
    assert issues[0].actual_namespace == "hypothesis"
    with pytest.raises(CrossNamespaceReferenceError) as raised:
        raise CrossNamespaceReferenceError(issues)
    assert raised.value.stage == "final_strategy"
    assert raised.value.invalid_ids == ("eda_hypothesis_001",)


def test_unambiguous_hypothesis_reference_repairs_to_approved_experiment() -> None:
    registry = build_experiment_registry(
        [_experiment("exp_001", "eda_hypothesis_001")],
        _review(approved=["exp_001"]),
    )

    repaired = repair_final_experiment_references(
        _strategy(["eda_hypothesis_001"], ["eda_hypothesis_001"]), registry
    )

    assert repaired.result.actions[0].experiment_ids == ["exp_001"]
    assert repaired.applied_repairs[0].original_id == "eda_hypothesis_001"
    assert _issues(repaired.result, registry, {"eda_hypothesis_001"}) == []


def test_ambiguous_or_missing_hypothesis_mapping_is_not_repaired() -> None:
    registry = build_experiment_registry(
        [
            _experiment("exp_001", "eda_hypothesis_001"),
            _experiment("exp_002", "eda_hypothesis_001"),
        ],
        _review(approved=["exp_001", "exp_002"]),
    )
    strategy = _strategy(
        ["eda_hypothesis_001", "eda_hypothesis_009"],
        ["eda_hypothesis_001", "eda_hypothesis_009"],
    )

    repaired = repair_final_experiment_references(strategy, registry)

    assert repaired.applied_repairs == ()
    assert repaired.result.actions[0].experiment_ids == [
        "eda_hypothesis_001", "eda_hypothesis_009"
    ]
    assert len(_issues(
        repaired.result, registry, {"eda_hypothesis_001", "eda_hypothesis_009"}
    )) == 2


def test_rejected_and_unapproved_experiments_cannot_be_final_actions() -> None:
    registry = build_experiment_registry(
        [_experiment("exp_001"), _experiment("exp_002"), _experiment("exp_003")],
        _review(approved=["exp_001"], rejected=["exp_002"], reviewed=["exp_001", "exp_002"]),
    )

    issues = _issues(_strategy(["exp_002", "exp_003"], ["hyp_001"]), registry, {"hyp_001"})

    assert {issue.reason for issue in issues} == {
        "rejected_experiment", "unknown_or_unapproved_experiment"
    }


def test_planner_identity_is_stable_and_distinct_from_source_hypothesis() -> None:
    item = _experiment("", "eda_hypothesis_007").model_copy(update={"experiment_id": None})
    first = _assign_and_validate_experiment_ids(
        [item], hypothesis_ids={"eda_hypothesis_007"}
    )
    second = _assign_and_validate_experiment_ids(
        [item], hypothesis_ids={"eda_hypothesis_007"}
    )

    assert first[0].experiment_id == second[0].experiment_id
    assert first[0].experiment_id != "eda_hypothesis_007"
    assert first[0].source_hypothesis_ids == ["eda_hypothesis_007"]
    with pytest.raises(ValueError, match="reuses a hypothesis_id"):
        _assign_and_validate_experiment_ids(
            [_experiment("eda_hypothesis_007", "eda_hypothesis_007")],
            hypothesis_ids={"eda_hypothesis_007"},
        )


def test_reviewer_decisions_must_use_planned_experiment_ids() -> None:
    draft = {"experiments": [_experiment("exp_001").model_dump(mode="json")]}

    _validate_experiment_decisions(_review(approved=["exp_001"]), draft)
    with pytest.raises(ValueError, match="unknown experiment_ids"):
        _validate_experiment_decisions(
            _review(approved=["eda_hypothesis_001"]), draft
        )
    revised = _review(approved=["exp_001"])
    revised.revised_sections = {
        "experiments": [_experiment("exp_invented").model_dump(mode="json")]
    }
    with pytest.raises(ValueError, match="revised_sections adds unknown"):
        _validate_experiment_decisions(revised, draft)


def test_reported_hypothesis_id_regression_is_bounded_and_structured() -> None:
    failing_ids = [
        "eda_hypothesis_001", "eda_hypothesis_002", "eda_hypothesis_003",
        "eda_hypothesis_004", "eda_hypothesis_006", "eda_hypothesis_007",
        "eda_hypothesis_008", "eda_hypothesis_009",
    ]
    experiments = [_experiment(f"exp_{index:03d}", hypothesis_id) for index, hypothesis_id in enumerate(failing_ids[:4], 1)]
    registry = build_experiment_registry(
        experiments,
        _review(approved=[item.experiment_id for item in experiments]),
    )
    strategy = _strategy(failing_ids, failing_ids)

    repaired = repair_final_experiment_references(strategy, registry)
    issues = _issues(repaired.result, registry, set(failing_ids))

    assert len(repaired.applied_repairs) == 4
    assert {issue.invalid_value for issue in issues} == set(failing_ids[4:])
    error = CrossNamespaceReferenceError(issues)
    assert error.actual_namespaces == ("hypothesis",) * 4
    assert "expected approved_experiment" in str(error)


def test_namespace_fields_survive_json_round_trip() -> None:
    strategy = _strategy(["exp_001"], ["eda_hypothesis_007"])
    round_trip = FinalStrategyResult.model_validate_json(strategy.model_dump_json())

    assert round_trip.actions[0].experiment_ids == ["exp_001"]
    assert round_trip.actions[0].hypothesis_ids == ["eda_hypothesis_007"]
    assert round_trip.actions[0].evidence_refs == ["feature_diagnostics"]
