from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from kaggle_researcher.contracts.final_synthesis_diagnostics import FinalSynthesisDiagnostics
from tests.fixtures.final_synthesis import synthesize_for_test
from tests.test_final_synthesizer import (
    _doc,
    _eda_pack,
    _plan,
    _research_hypotheses,
)


SECTIONS = [
    "executive_summary", "metric_and_validation", "dataset_facts_from_eda",
    "leakage_and_data_quality", "drift_and_leaderboard_risk", "baseline_findings",
    "feature_priorities", "modeling_plan", "experiments_queue", "what_not_to_do",
    "first_48_hours",
]


class ProtocolClient:
    def __init__(self, selection: dict[str, Any], *, invalid_rendering: bool = False) -> None:
        self.selection = selection
        self.invalid_rendering = invalid_rendering
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        prompt = json.loads(kwargs["user_prompt"])
        if "selection_context" in prompt or "invalid_draft" in prompt:
            return deepcopy(self.selection)
        if self.invalid_rendering:
            return {"schema_version": "2.0", "skeleton_id": "changed", "skeleton_hash": "changed"}
        immutable = prompt["immutable_strategy_payload"]
        return {
            "schema_version": "2.0",
            "skeleton_id": prompt["skeleton_id"],
            "skeleton_hash": prompt["skeleton_hash"],
            "executive_summary": "Use the frozen validation-first strategy.",
            "action_wording": [
                {"action_id": item["action_id"], "display_action": item["action"], "display_reason": item["reason"]}
                for item in immutable["actions"]
            ],
            "experiment_wording": [
                {
                    "experiment_id": item["experiment_id"], "display_name": item["name"],
                    "display_hypothesis": item["hypothesis"], "display_exact_change": item["change"],
                    "display_acceptance_rule": item["acceptance_rule"], "display_risk": item["risks"][0],
                }
                for item in [*immutable["core_experiments"], *immutable["experiment_backlog"]]
            ],
            "family_wording": [
                {
                    "family_id": item["family_id"], "display_name": item["name"],
                    "display_hypothesis": item["hypothesis"],
                    "display_acceptance_rule": item["acceptance_rule"], "display_risks": item["risks"],
                }
                for item in immutable["feature_experiment_families"]
            ],
            "section_summaries": [
                {"section_id": item["section_id"], "summary": item["summary"]}
                for item in immutable["section_structure"]
            ],
            "limitation_wording": [],
            "uncertainty_summary": "Evidence remains bounded to the supplied catalogs.",
        }


def _selection() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "selected_actions": [{
            "client_action_key": "lock_validation", "action_kind": "validation_setup",
            "action": "Lock stratified validation.", "priority": "P0", "confidence": "high",
            "reason": "EDA selected this primary validation.",
            "primary_evidence_refs": ["validation_evidence.primary_validation"],
            "supporting_evidence_refs": [], "limitation_evidence_refs": [], "source_refs": [],
            "motivating_hypothesis_ids": [], "safety_hypothesis_ids": [],
            "validation_context_ids": ["val_001"], "rejected_hypothesis_ids": [],
            "safety_constraint_ids": [], "validation_requirement_ids": [],
            "feature_metadata": None, "dependencies": [], "limitations": [],
        }],
        "feature_experiment_families": [], "candidate_experiments": [],
        "proposed_core_experiment_ids": [], "proposed_backlog_experiment_ids": [],
        "section_plan": [
            {
                "section_id": section,
                "selected_action_keys": ["lock_validation"] if section in {
                    "executive_summary", "metric_and_validation", "first_48_hours"
                } else [],
                "selected_family_keys": [], "selected_experiment_keys": [],
                "summary_intent": "Communicate the validated, dependency-ordered plan.",
            }
            for section in SECTIONS
        ],
        "limitations": [],
    }


async def _run(client: ProtocolClient, tmp_path):
    return await synthesize_for_test(
        competition_desc="Generic iid binary classification.",
        plan_data=_plan(), retrieved_documents=[_doc()], domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={}, client=client, model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_valid_selection_and_rendering_produce_llm_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "two_call")
    client = ProtocolClient(_selection())
    result = await _run(client, tmp_path)
    assert len(client.calls) == 2
    assert result.synthesis_status == result.selection_status == "llm_success"
    assert result.rendering_status == "llm_success"
    assert result.skeleton_id and len(result.skeleton_hash or "") == 64
    assert result.selection_prompt_fingerprint["prompt_version"] == "2.0"
    assert result.rendering_prompt_fingerprint["prompt_version"] == "2.0"
    diagnostics = FinalSynthesisDiagnostics.model_validate_json(
        (tmp_path / "final_synthesis_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics.selection_attempts[0].schema_succeeded is True
    assert diagnostics.rendering_attempts[0].integrity_validation_succeeded is True
    assert diagnostics.bridge and diagnostics.bridge.canonical_action_count == 1
    assert diagnostics.prompt_fingerprints["selection"]["prompt_version"] == "2.0"


@pytest.mark.asyncio
async def test_rendering_failure_preserves_valid_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "two_call")
    client = ProtocolClient(_selection(), invalid_rendering=True)
    result = await _run(client, tmp_path)
    assert len(client.calls) == 3  # selection + rendering + bounded rendering repair
    assert result.synthesis_status == result.selection_status == "llm_success"
    assert result.rendering_status == "deterministic_render"
    assert result.fallback_used is False
    assert any("Call 2" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_selection_repair_status_is_independent_from_rendering(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "two_call")
    valid = _selection()
    class RepairClient(ProtocolClient):
        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            prompt = json.loads(kwargs["user_prompt"])
            if "selection_context" in prompt:
                return {"schema_version": "1.0", "executive_summary": "legacy", "warnings": []}
            if "invalid_draft" in prompt:
                return deepcopy(valid)
            immutable = prompt["immutable_strategy_payload"]
            return await _render_from_prompt(self, kwargs, prompt, immutable)
    client = RepairClient(valid)
    result = await _run(client, tmp_path)
    assert result.synthesis_status == result.selection_status == "repaired_success"
    assert result.rendering_status == "llm_success"
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_provider_failure_uses_deterministic_fallback_and_records_stage(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "two_call")
    class FailingClient:
        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("provider unavailable")
    result = await synthesize_for_test(
        competition_desc="Generic iid binary classification.",
        plan_data=_plan(), retrieved_documents=[_doc()], domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={}, client=FailingClient(), model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )
    assert result.synthesis_status == result.selection_status == "degraded_fallback"
    assert result.rendering_status == "deterministic_render"
    diagnostics = FinalSynthesisDiagnostics.model_validate_json(
        (tmp_path / "final_synthesis_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics.provider_failures[0]["stage"] == "selection"
    assert diagnostics.fallback_required is True


async def _render_from_prompt(client: ProtocolClient, kwargs: dict[str, Any], prompt: dict[str, Any], immutable: dict[str, Any]) -> dict[str, Any]:
    client.calls.pop()
    return await ProtocolClient.chat_json(client, **kwargs)
