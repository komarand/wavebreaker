from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.normalization import normalize_contract_payload
from kaggle_researcher.reasoning.experiment_planner import (
    _assign_and_validate_experiment_ids,
    _parse_and_normalize_experiments,
)
from kaggle_researcher.reasoning.final_synthesizer import FinalStrategyResult, render_final_strategy
from kaggle_researcher.schemas import MetricResult, ReviewResult, ValidationResult


pytestmark = pytest.mark.contract
FIXTURES = Path("tests/fixtures/reasoning")


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_registered_null_collections_normalize_but_optional_object_stays_null() -> None:
    result = ValidationResult.model_validate(_fixture("validation_architect_null_secondary.json"))

    assert result.secondary_validation is None
    assert result.failure_modes == []
    assert result.do_not_use == []
    assert result.policy_notes == []


def test_normalization_is_allowlisted_and_idempotent() -> None:
    payload = {"evidence_ids": None, "secondary_validation": None, "unrelated": None}
    once = normalize_contract_payload(payload, "ValidationResult")

    assert once == normalize_contract_payload(once, "ValidationResult")
    assert once["evidence_ids"] == []
    assert once["secondary_validation"] is None
    assert once["unrelated"] is None


def test_experiment_alias_is_exact_and_null_evidence_collection_is_safe() -> None:
    payload = _fixture("experiment_planner_alias_evidence.json")
    experiments, replacements = _parse_and_normalize_experiments(payload["experiments"])

    assert experiments[0].evidence_ids == ["validation_result", "source-001"]
    assert replacements == [("validation_policy", "validation_result")]
    null_item = dict(payload["experiments"][0], evidence_ids=None)
    parsed, _ = _parse_and_normalize_experiments([null_item])
    assert parsed[0].evidence_ids == []


def test_experiment_ids_are_stable_and_duplicates_fail() -> None:
    payload = _fixture("experiment_planner_alias_evidence.json")
    experiments, _ = _parse_and_normalize_experiments(payload["experiments"])

    first = _assign_and_validate_experiment_ids(experiments)
    second = _assign_and_validate_experiment_ids(experiments)
    assert first[0].experiment_id == second[0].experiment_id

    duplicate = [
        first[0],
        first[0].model_copy(update={"experiment": "A distinct experiment"}),
    ]
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        _assign_and_validate_experiment_ids(duplicate)


def test_reviewer_null_collections_normalize_without_global_null_coercion() -> None:
    result = ReviewResult.model_validate(_fixture("skeptical_reviewer_no_issues.json"))

    assert result.evidence_ids == []
    assert result.unsupported_claims == []
    assert result.revised_sections == {}


def test_required_nested_object_and_strict_boolean_still_reject_bad_values() -> None:
    validation = _fixture("validation_architect_null_secondary.json")
    validation["primary_validation"] = {}
    with pytest.raises(ValidationError):
        ValidationResult.model_validate(validation)

    metric = {
        "confidence": "medium",
        "metric_explanation": "Ranking metric.",
        "needs_calibration": "yes",
        "rank_averaging_useful": True,
        "threshold_search_needed": False,
        "surrogate_loss_suggestion": "Use a ranking-compatible surrogate.",
    }
    with pytest.raises(ValidationError):
        MetricResult.model_validate(metric)


def test_final_strategy_null_collections_normalize_before_rendering() -> None:
    payload = _fixture("final_strategy_valid.json")
    payload["limitations"] = None
    payload["actions"][0]["experiment_ids"] = None
    payload["actions"][0]["source_refs"] = None
    result = FinalStrategyResult.model_validate(payload)

    assert result.limitations == []
    assert result.actions[0].experiment_ids == []
    rendered = render_final_strategy(result)
    assert "{}" not in rendered
