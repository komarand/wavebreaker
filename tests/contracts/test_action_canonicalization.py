from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from kaggle_researcher.contracts.action_canonicalization import (
    FinalStrategyActionCanonicalizationError,
    canonicalize_final_strategy_actions,
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
