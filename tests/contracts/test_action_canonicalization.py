from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.action_canonicalization import (
    FinalStrategyActionCanonicalizationError,
    canonicalize_final_strategy_actions,
    canonicalize_semantic_strategy_actions,
    semantic_action_signature,
)
from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.reasoning.final_synthesizer import render_final_strategy


def _action(action_id: str | None = "action_1", **updates):
    value = {
        "action_id": action_id,
        "priority": "P1",
        "action": "Test a stable validation baseline.",
        "reason": "Establish a comparable score.",
        "evidence_refs": ["validation_evidence.primary_validation"],
        "related_hypothesis_ids": ["val_001"],
        "limitations": [],
    }
    value.update(updates)
    if action_id is None:
        value.pop("action_id")
    return value


def _payload(actions, sections):
    return {
        "competition_id": "canonical-test",
        "synthesis_status": "llm_success",
        "llm_output_valid": True,
        "repair_attempted": False,
        "repair_succeeded": False,
        "fallback_used": False,
        "synthesis_diagnostics_path": None,
        "actions": actions,
        "sections": sections,
    }


def test_top_level_and_section_action_become_one_canonical_action() -> None:
    action = _action()
    canonical, diagnostics = canonicalize_final_strategy_actions(_payload(
        [action],
        [{"section_id": "validation", "actions": [deepcopy(action)]}],
    ))

    assert len(canonical["actions"]) == 1
    assert canonical["sections"][0]["action_ids"] == ["action_1"]
    assert "actions" not in canonical["sections"][0]
    assert diagnostics.merged_duplicate_actions == ("action_1",)


def test_ten_duplicated_invalid_actions_produce_ten_canonical_errors() -> None:
    actions = [
        _action(f"action_{index}", priority="INVALID", action=f"Test action {index}.")
        for index in range(10)
    ]
    canonical, _ = canonicalize_final_strategy_actions(_payload(
        actions,
        [{
            "section_id": "validation",
            "title": "Validation",
            "summary": "Validation actions.",
            "actions": deepcopy(actions),
        }],
    ))

    with pytest.raises(ValidationError) as caught:
        FinalStrategyResult.model_validate(canonical)

    action_errors = [
        error for error in caught.value.errors()
        if error["loc"][:1] == ("actions",)
    ]
    assert len(action_errors) == 10
    assert not any(error["loc"][:1] == ("sections",) for error in caught.value.errors())


def test_duplicate_merge_unions_evidence_hypotheses_and_limitations() -> None:
    canonical, _ = canonicalize_final_strategy_actions(_payload(
        [_action(evidence_refs=["evidence.a"], related_hypothesis_ids=["h1"])],
        [{
            "section_id": "validation",
            "actions": [_action(
                evidence_refs=["evidence.b", "evidence.a"],
                related_hypothesis_ids=["h2", "h1"],
                limitations=["Needs measurement."],
            )],
        }],
    ))

    action = canonical["actions"][0]
    assert action["evidence_refs"] == ["evidence.a", "evidence.b"]
    assert action["related_hypothesis_ids"] == ["h1", "h2"]
    assert action["limitations"] == ["Needs measurement."]


def test_hypothesis_ids_merge_without_duplicates() -> None:
    canonical, _ = canonicalize_final_strategy_actions(_payload(
        [_action(related_hypothesis_ids=["h1"])],
        [{"section_id": "validation", "actions": [
            _action(related_hypothesis_ids=["h1", "h2"]),
        ]}],
    ))

    assert canonical["actions"][0]["related_hypothesis_ids"] == ["h1", "h2"]


def test_limitations_merge_without_duplicates() -> None:
    canonical, _ = canonicalize_final_strategy_actions(_payload(
        [_action(limitations=["First limitation."])],
        [{"section_id": "validation", "actions": [
            _action(limitations=["First limitation.", "Second limitation."]),
        ]}],
    ))

    assert canonical["actions"][0]["limitations"] == [
        "First limitation.",
        "Second limitation.",
    ]


def test_order_and_section_membership_are_stable_and_unique() -> None:
    canonical, diagnostics = canonicalize_final_strategy_actions(_payload(
        [_action("a"), _action("b", action="Test B.")],
        [{
            "section_id": "validation",
            "action_ids": ["b", "a", "b"],
        }],
    ))

    assert [action["action_id"] for action in canonical["actions"]] == ["a", "b"]
    assert canonical["sections"][0]["action_ids"] == ["b", "a"]
    assert diagnostics.section_memberships[0].action_ids == ("b", "a")


def test_missing_action_id_is_deterministic_and_canonicalization_is_idempotent() -> None:
    raw = _payload(
        [_action(None)],
        [{"section_id": "validation", "actions": [_action(None)]}],
    )
    first, diagnostics = canonicalize_final_strategy_actions(raw)
    second, _ = canonicalize_final_strategy_actions(first)

    assert first == second
    assert len(first["actions"]) == 1
    assert first["actions"][0]["action_id"].startswith("action_")
    assert diagnostics.generated_action_ids == (first["actions"][0]["action_id"],)


def test_canonical_payload_is_idempotent() -> None:
    first, _ = canonicalize_final_strategy_actions(_payload(
        [_action("stable")],
        [{"section_id": "validation", "action_ids": ["stable"]}],
    ))
    second, diagnostics = canonicalize_final_strategy_actions(first)

    assert second == first
    assert diagnostics.merged_duplicate_actions == ()


def test_top_level_action_is_not_injected_into_unrelated_section() -> None:
    canonical, _ = canonicalize_final_strategy_actions(_payload(
        [_action("top-only")],
        [{"section_id": "validation", "evidence_refs": ["evidence.section"]}],
    ))

    assert canonical["sections"][0]["action_ids"] == []


def test_conflicting_definitions_raise_typed_error_with_diagnostics() -> None:
    with pytest.raises(FinalStrategyActionCanonicalizationError) as caught:
        canonicalize_final_strategy_actions(_payload(
            [_action("same")],
            [{"section_id": "validation", "actions": [
                _action("same", reason="A conflicting reason."),
            ]}],
        ))

    assert caught.value.diagnostics.conflicting_action_definitions
    assert "reason" in caught.value.diagnostics.conflicting_action_definitions[0]


def test_dangling_section_action_id_raises_typed_error() -> None:
    with pytest.raises(FinalStrategyActionCanonicalizationError) as caught:
        canonicalize_final_strategy_actions(_payload(
            [_action("known")],
            [{"section_id": "validation", "action_ids": ["missing"]}],
        ))

    assert caught.value.diagnostics.dangling_action_ids == ("missing",)


def test_renderer_resolves_section_membership_from_canonical_action_map() -> None:
    result = FinalStrategyResult.model_validate(_payload(
        [_action("rendered")],
        [{
            "section_id": "validation",
            "title": "Validation",
            "summary": "Validation actions.",
            "action_ids": ["rendered"],
        }],
    ))

    rendered = render_final_strategy(result)

    assert "Test a stable validation baseline." in rendered
    dumped = result.model_dump(mode="json")
    assert dumped["sections"][0]["action_ids"] == ["rendered"]
    assert "actions" not in dumped["sections"][0]


def _semantic_sections(*action_ids: str):
    return [{
        "section_id": "metric_and_validation",
        "title": "Metric And Validation",
        "summary": "Validation policy.",
        "action_ids": list(action_ids),
    }, {
        "section_id": "what_not_to_do",
        "title": "What Not To Do",
        "summary": "Safety exclusions.",
        "evidence_refs": ["inferred_schema.primary_id_column"],
    }, {
        "section_id": "drift_and_leaderboard_risk",
        "title": "Drift And Leaderboard Risk",
        "summary": "Drift diagnostics.",
        "evidence_refs": ["drift_evidence"],
    }]


def test_semantic_exact_duplicate_text_merges_and_rewrites_sections() -> None:
    actions = [
        _action("exact_a", action="Run a stable diagnostic experiment."),
        _action("exact_b", action="Run a stable diagnostic experiment."),
    ]
    raw = _payload(actions, _semantic_sections("exact_a", "exact_b", "exact_b"))

    canonical, diagnostics = canonicalize_semantic_strategy_actions(raw)

    assert len(canonical["actions"]) == 1
    survivor = canonical["actions"][0]["action_id"]
    assert survivor == "exact_a"
    assert canonical["sections"][0]["action_ids"] == [survivor]
    assert [(item.original_id, item.replacement_id) for item in diagnostics.merges] == [
        ("exact_b", "exact_a")
    ]


def test_paraphrased_primary_validation_actions_merge_and_union_support() -> None:
    first = _action(
        "action_validation",
        priority="P1",
        confidence="medium",
        action="Use stratified_kfold as the primary validation method.",
        validation_strategy="stratified_kfold",
        evidence_refs=["validation_evidence.a"],
        related_hypothesis_ids=["h1"],
        source_refs=["source_1"],
        experiment_ids=["exp_1"],
    )
    second = _action(
        "eda_validation_stratified_kfold",
        priority="P0",
        confidence="high",
        action="Use stratified_kfold as the primary validation policy selected by EDA.",
        validation_strategy="stratified_kfold",
        evidence_refs=["validation_evidence.b"],
        eda_result_refs=["validation_evidence.b"],
        related_hypothesis_ids=["h2"],
        risk_ids=["risk_1"],
        validation_requirement_ids=["requirement_1"],
        safety_constraint_ids=["safety_1"],
        limitations=["Keep fold diagnostics."],
    )

    canonical, _ = canonicalize_semantic_strategy_actions(
        _payload([first, second], _semantic_sections(
            "action_validation", "eda_validation_stratified_kfold",
        ))
    )

    action = canonical["actions"][0]
    assert action["action_id"] == "eda_validation_stratified_kfold"
    assert action["priority"] == "P0"
    assert action["confidence"] == "high"
    assert action["evidence_refs"] == ["validation_evidence.a", "validation_evidence.b"]
    assert action["related_hypothesis_ids"] == ["h1", "h2"]
    assert action["source_refs"] == ["source_1"]
    assert action["eda_result_refs"] == ["validation_evidence.b"]
    assert action["experiment_ids"] == ["exp_1"]
    assert action["risk_ids"] == ["risk_1"]
    assert action["validation_requirement_ids"] == ["requirement_1"]
    assert action["safety_constraint_ids"] == ["safety_1"]
    assert action["limitations"] == ["Keep fold diagnostics."]


def test_primary_id_exclusion_paraphrases_share_canonical_key() -> None:
    explicit = _action(
        "exclude_passenger",
        action="Exclude PassengerId from model features.",
        evidence_refs=["inferred_schema.primary_id_column"],
    )
    generic = _action(
        "eda_primary_id",
        action=(
            "Keep the primary ID excluded from model features and use it only for "
            "row tracking."
        ),
        evidence_refs=["inferred_schema.primary_id_column"],
    )

    left = semantic_action_signature(explicit, primary_id="PassengerId")
    right = semantic_action_signature(generic, primary_id="PassengerId")
    canonical, _ = canonicalize_semantic_strategy_actions(
        _payload([explicit, generic], _semantic_sections(
            "exclude_passenger", "eda_primary_id",
        )),
        primary_id="PassengerId",
    )

    assert left.canonical_key == right.canonical_key
    assert left.canonical_key == "exclude_feature:PassengerId:primary_id"
    assert len(canonical["actions"]) == 1
    assert canonical["sections"][1]["action_ids"] == ["eda_primary_id"]


def test_drift_warning_paraphrases_merge() -> None:
    actions = [
        _action("drift_a", action="Audit high train/test drift before trusting public LB."),
        _action("eda_drift_high", action="Treat high train/test drift as leaderboard-risk diagnostics."),
    ]
    canonical, _ = canonicalize_semantic_strategy_actions(
        _payload(actions, _semantic_sections("drift_a", "eda_drift_high"))
    )

    assert len(canonical["actions"]) == 1
    assert canonical["actions"][0]["action_id"] == "eda_drift_high"
    assert canonical["sections"][2]["action_ids"] == ["eda_drift_high"]


def test_distinct_diagnostic_and_calibration_actions_are_not_overmerged() -> None:
    actions = [
        _action("exclude", action="Exclude PassengerId from model features."),
        _action("ablate", action="Test PassengerId as a diagnostic ablation."),
        _action("drift", action="Audit train/test drift before trusting public LB."),
        _action("temporal", action="Use temporal CV for ordered validation."),
        _action("threshold", action="Tune the decision threshold on validation folds."),
        _action("calibrate", action="Calibrate predicted probabilities on validation folds."),
    ]
    canonical, _ = canonicalize_semantic_strategy_actions(
        _payload(actions, _semantic_sections(*[item["action_id"] for item in actions])),
        primary_id="PassengerId",
    )

    assert len(canonical["actions"]) == len(actions)
    assert len({
        semantic_action_signature(action, primary_id="PassengerId").canonical_key
        for action in canonical["actions"]
    }) == len(actions)


def test_semantic_repairs_are_real_and_canonicalization_is_deterministic() -> None:
    actions = [
        _action("validation_b", action="Use stratified_kfold as the primary validation method."),
        _action("validation_a", action="Use stratified_kfold as the primary validation policy."),
    ]
    raw = _payload(actions, _semantic_sections("validation_b", "validation_a"))

    first, _ = canonicalize_semantic_strategy_actions(raw)
    second, second_diagnostics = canonicalize_semantic_strategy_actions(first)

    assert second == first
    assert second_diagnostics.merges == ()
    semantic_repairs = [
        repair for repair in first["reference_repairs"]
        if repair["field_path"].startswith("semantic_action_merge.")
    ]
    assert semantic_repairs
    assert all(
        repair["original_id"] != repair["replacement_id"]
        for repair in semantic_repairs
    )


def test_markdown_renders_surviving_primary_id_action_once() -> None:
    actions = [
        _action(
            "exclude_passenger",
            action="Exclude PassengerId from model features.",
            evidence_refs=["inferred_schema.primary_id_column"],
        ),
        _action(
            "eda_primary_id",
            action="Keep the primary ID excluded from model features and use it only for row tracking.",
            evidence_refs=["inferred_schema.primary_id_column"],
        ),
    ]
    canonical, _ = canonicalize_semantic_strategy_actions(
        _payload(actions, _semantic_sections("exclude_passenger", "eda_primary_id")),
        primary_id="PassengerId",
    )
    result = FinalStrategyResult.model_validate(canonical)
    rendered = render_final_strategy(result)

    assert len(result.actions) == 1
    assert rendered.count(result.actions[0].action) == 1
