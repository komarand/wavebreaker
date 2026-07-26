from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from tests.test_final_strategy_two_call_protocol import ProtocolClient, _run, _selection


class MutatingRenderingClient(ProtocolClient):
    def __init__(self, mutation: str) -> None:
        super().__init__(_selection())
        self.mutation = mutation

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        prompt = json.loads(kwargs["user_prompt"])
        if "selection_context" in prompt or "invalid_draft" in prompt:
            self.calls.append(kwargs)
            return deepcopy(self.selection)
        self.calls.append(kwargs)
        immutable = prompt["immutable_strategy_payload"]
        response = {
            "contract_family": "strategy_rendering_draft",
            "schema_version": "2.0",
            "skeleton_id": prompt.get("skeleton_id") or prompt["required_identity"]["skeleton_id"],
            "skeleton_hash": prompt.get("skeleton_hash") or prompt["required_identity"]["skeleton_hash"],
            "executive_summary": "Render only the frozen validation strategy.",
            "action_wording": [
                {
                    "action_id": item["action_id"],
                    "display_action": item["action"],
                    "display_reason": item["reason"],
                }
                for item in immutable["actions"]
            ],
            "experiment_wording": [],
            "family_wording": [],
            "section_summaries": [
                {"section_id": item["section_id"], "summary": item["summary"]}
                for item in immutable["section_structure"]
            ],
            "limitation_wording": [],
            "uncertainty_summary": "Evidence is limited to the frozen catalogs.",
        }
        if self.mutation == "structural_field":
            response["dependencies"] = ["invented"]
        elif self.mutation == "hash":
            response["skeleton_hash"] = "0" * 64
        return response


@pytest.mark.asyncio
async def test_call_2_valid_changes_wording_only() -> None:
    result = await _run(ProtocolClient(_selection()), None)
    assert result.selection_status == result.rendering_status == "llm_success"
    assert result.skeleton_hash and len(result.skeleton_hash) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["structural_field", "hash"])
async def test_call_2_structural_mutation_or_hash_mismatch_is_rejected(
    mutation: str,
    tmp_path,
) -> None:
    client = MutatingRenderingClient(mutation)
    result = await _run(client, tmp_path)
    assert result.synthesis_status == "llm_success"
    assert result.rendering_status == "deterministic_render"
    assert result.fallback_used is False
    assert len(client.calls) == 3  # Call 1 + Call 2 + one bounded rendering repair


@pytest.mark.asyncio
async def test_call_2_failure_does_not_degrade_valid_selection(tmp_path) -> None:
    result = await _run(MutatingRenderingClient("hash"), tmp_path)
    assert result.selection_status == result.synthesis_status == "llm_success"
    assert result.rendering_status == "deterministic_render"
    assert any("Call 2" in warning for warning in result.warnings)
