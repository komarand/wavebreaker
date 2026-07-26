from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import kaggle_researcher.contracts.evidence as evidence_module
import kaggle_researcher.reasoning.final_strategy_two_call as protocol_module
from kaggle_researcher.schemas import PlanData
from tests.fixtures.final_synthesis import synthesize_for_test
from tests.test_final_strategy_two_call_protocol import ProtocolClient, _selection
from tests.test_final_synthesizer import _doc, _eda_pack, _research_hypotheses


async def _synthesize(client: Any, *, plan: PlanData | None = None, pack=None):
    return await synthesize_for_test(
        competition_desc="Generic bounded tabular task.",
        plan_data=plan or PlanData(
            task_type="binary_classification", metric="roc_auc", domain="generic tabular"
        ),
        retrieved_documents=[_doc()],
        domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=pack or _eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={},
        client=client,
        model="deepseek-v4-pro",
    )


@pytest.mark.asyncio
async def test_call_1_valid_routes_both_llm_drafts_through_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    original = protocol_module.ingest_contract

    def counted(*args: Any, **kwargs: Any):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(protocol_module, "ingest_contract", counted)
    client = ProtocolClient(_selection())
    result = await _synthesize(client)

    assert result.synthesis_status == "llm_success"
    assert [call["expected_family"] for call in calls] == [
        "strategy_selection_draft",
        "strategy_rendering_draft",
    ]
    assert all(call["source_kind"] == "llm_generated" for call in calls)
    selection_prompt = json.loads(client.calls[0]["user_prompt"])
    assert "eda_evidence_pack" not in selection_prompt
    assert "retrieved_documents" not in selection_prompt
    assert set(_selection()["selected_actions"][0]["primary_evidence_refs"]) <= set(
        selection_prompt["allowed_evidence_refs"]
    )


@pytest.mark.asyncio
async def test_call_1_repaired_once_then_succeeds() -> None:
    valid = _selection()

    class RepairClient(ProtocolClient):
        async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            prompt = json.loads(kwargs["user_prompt"])
            if "selection_context" in prompt:
                self.calls.append(kwargs)
                return {"contract_family": "final_strategy", "schema_version": "2.0"}
            if "invalid_draft" in prompt:
                self.calls.append(kwargs)
                return deepcopy(valid)
            return await super().chat_json(**kwargs)

    client = RepairClient(valid)
    result = await _synthesize(client)

    assert result.selection_status == result.synthesis_status == "repaired_success"
    assert result.rendering_status == "llm_success"
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_exact_previous_monolithic_response_is_schema_failure_not_manifest_failure(
    tmp_path: Path,
) -> None:
    previous = json.loads(
        Path("tests/fixtures/reasoning/final_strategy_previous_llm_failure.json")
        .read_text(encoding="utf-8")
    )
    previous_diagnostics = json.loads(
        Path(
            "tests/fixtures/reasoning/"
            "final_synthesis_previous_llm_failure_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    client = ProtocolClient(previous)
    result = await synthesize_for_test(
        competition_desc="Generic iid binary classification.",
        plan_data=PlanData(
            task_type="binary_classification", metric="roc_auc", domain="generic tabular"
        ),
        retrieved_documents=[_doc()], domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={}, client=client, model="deepseek-v4-pro",
        diagnostics_dir=tmp_path,
    )
    diagnostics = json.loads(
        (tmp_path / "final_synthesis_diagnostics.json").read_text(encoding="utf-8")
    )

    assert result.synthesis_status == "degraded_fallback"
    assert result.rendering_status == "deterministic_render"
    assert len(client.calls) == 2  # initial + one bounded selection repair; no Call 2
    assert diagnostics["manifest_parity_succeeded"] is True
    assert diagnostics["selection_attempts"][0]["issues"]
    assert diagnostics["internal_contract_failure"] is None
    assert previous_diagnostics["manifest_parity_succeeded"] is True
    assert previous_diagnostics["attempts"][0]["issues"][0]["issue_type"] == (
        "missing_usable_actions"
    )


@pytest.mark.asyncio
async def test_prompt_and_bridge_share_the_same_allowed_evidence_catalog() -> None:
    invalid = _selection()
    invalid["selected_actions"][0]["primary_evidence_refs"] = ["invented.ref"]
    client = ProtocolClient(invalid)
    result = await _synthesize(client)
    prompt = json.loads(client.calls[0]["user_prompt"])

    assert "invented.ref" not in prompt["allowed_evidence_refs"]
    assert result.synthesis_status == "degraded_fallback"
    assert result.rendering_status == "deterministic_render"


@pytest.mark.asyncio
async def test_two_call_selection_performs_no_evidence_space_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("evidence-space recomputation is forbidden downstream")

    monkeypatch.setattr(evidence_module, "generate_semantic_evidence_refs", forbidden)
    monkeypatch.setattr(evidence_module, "generate_allowed_evidence_refs", forbidden)
    result = await _synthesize(ProtocolClient(_selection()))
    assert result.synthesis_status == "llm_success"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan", "sparse"),
    [
        (PlanData(task_type="binary_classification", metric="roc_auc", domain="generic"), False),
        (PlanData(task_type="regression", metric="rmse", domain="generic"), False),
        (PlanData(task_type="binary_classification", metric="roc_auc", domain="sparse"), True),
    ],
)
async def test_selection_is_generic_for_binary_regression_and_sparse_fixtures(
    plan: PlanData,
    sparse: bool,
) -> None:
    pack = _eda_pack(primary_method="stratified_kfold")
    if sparse:
        pack = pack.model_copy(update={
            "baseline_evidence": {},
            "baseline_ablation_evidence": {},
            "feature_diagnostics": {},
        })
    result = await _synthesize(ProtocolClient(_selection()), plan=plan, pack=pack)
    assert result.task_type == plan.task_type
    assert result.synthesis_status == "llm_success"
    assert len(result.actions) <= 15
