from __future__ import annotations

import importlib
import json
from dataclasses import replace

import pytest

import kaggle_researcher.contracts.evidence as evidence_module
from tests.fixtures.final_synthesis import synthesize_for_test
from kaggle_researcher.contracts.evidence_manifest import publish_eda_evidence_bundle
from kaggle_researcher.contracts.errors import EvidenceManifestPackMismatchError
from kaggle_researcher.contracts.hashing import sha256_contract
from kaggle_researcher.contracts.registries import build_contract_registries
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyResult,
    postprocess_final_strategy_result,
    synthesize_final_strategy,
)
from tests.fixtures.evidence_contract import stage_bundle
from tests.test_final_synthesizer import (
    FakeFinalSynthesizerClient,
    _doc,
    _eda_pack,
    _plan,
    _research_hypotheses,
    _strategy_payload,
)


@pytest.mark.asyncio
async def test_allowed_prompt_reference_survives_mocked_end_to_end_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "monolithic_legacy")
    cited = "validation_evidence.primary_validation"
    client = FakeFinalSynthesizerClient(_strategy_payload())
    result = await synthesize_for_test(
        competition_desc="Generic iid binary classification with ROC AUC.",
        plan_data=_plan(), retrieved_documents=[_doc()], domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={}, client=client, model="mock-model",
        diagnostics_dir=tmp_path,
    )
    supplied = json.loads(client.calls[0]["user_prompt"])

    assert cited in supplied["allowed_evidence_refs"]
    assert cited in supplied["allowed_eda_result_refs"]
    assert result.synthesis_status in {"llm_success", "repaired_success"}
    assert result.fallback_used is False
    assert any(cited in action.evidence_refs for action in result.actions)
    diagnostics = json.loads(
        (tmp_path / "final_synthesis_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["prompt_manifest_hash"] == diagnostics["validator_manifest_hash"]
    assert diagnostics["manifest_hash"] == diagnostics["prompt_manifest_hash"]
    assert diagnostics["manifest_parity_succeeded"] is True


@pytest.mark.asyncio
async def test_final_synthesizer_performs_zero_downstream_reference_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "monolithic_legacy")
    calls: list[str] = []
    original_semantic = evidence_module.generate_semantic_evidence_refs
    original_allowed = evidence_module.generate_allowed_evidence_refs

    def semantic(*args, **kwargs):
        calls.append("semantic")
        return original_semantic(*args, **kwargs)

    def allowed(*args, **kwargs):
        calls.append("allowed")
        return original_allowed(*args, **kwargs)

    monkeypatch.setattr(evidence_module, "generate_semantic_evidence_refs", semantic)
    monkeypatch.setattr(evidence_module, "generate_allowed_evidence_refs", allowed)
    client = FakeFinalSynthesizerClient(_strategy_payload())
    await synthesize_for_test(
        competition_desc="Generic classification.", plan_data=_plan(),
        retrieved_documents=[_doc()], domain_patterns=[],
        research_hypotheses=_research_hypotheses(),
        eda_evidence_pack=_eda_pack(primary_method="stratified_kfold"),
        reasoning_outputs={}, client=client, model="mock-model",
    )

    # The publication fixture is already manifested; Final Synthesizer must not
    # invoke either low-level generator at any downstream call site.
    assert calls == []
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_pack_hash_mismatch_fails_before_model_call_with_internal_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("FINAL_SYNTHESIS_PROTOCOL", "monolithic_legacy")
    research, eda, reasoning, _ = stage_bundle()
    published = publish_eda_evidence_bundle(eda.evidence_pack)
    published_stage = replace(
        eda,
        evidence_pack=published.evidence_pack,
        evidence_manifest=published.evidence_manifest,
        published_bundle=published,
    )
    registries = build_contract_registries(
        research=research, eda=published_stage, reasoning=reasoning
    )
    context = build_final_synthesis_context(
        competition_desc="Generic classification.",
        research=research,
        published_eda_bundle=published,
        reasoning=reasoning,
        registries=registries,
        eda_summary_text="# EDA",
    )
    context.published_eda_bundle.evidence_pack.baseline_evidence["metric_value"] = 0.99
    client = FakeFinalSynthesizerClient(_strategy_payload())

    with pytest.raises(EvidenceManifestPackMismatchError) as raised:
        await synthesize_final_strategy(
            context=context,
            registries=registries,
            client=client,
            model="mock-model",
            diagnostics_dir=tmp_path,
        )

    assert client.calls == []
    assert raised.value.expected_pack_hash != raised.value.actual_pack_hash
    diagnostics = json.loads(
        (tmp_path / "final_synthesis_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["attempts"][0]["issues"][0]["stage"] == "internal_contract_validation"
    assert diagnostics["attempts"][0]["issues"][0]["issue_type"] == "EvidenceManifestPackMismatchError"
    assert diagnostics["pack_hash"] == raised.value.expected_pack_hash
    assert diagnostics["actual_pack_hash"] == raised.value.actual_pack_hash
    assert diagnostics["internal_contract_failure"]["stage"] == "published_bundle_validation"
    assert diagnostics["manifest_parity_succeeded"] is False


def test_final_postprocessing_does_not_mutate_published_pack() -> None:
    pack = _eda_pack(primary_method="stratified_kfold")
    before = sha256_contract(pack)
    result = FinalStrategyResult.model_validate(_strategy_payload())

    postprocess_final_strategy_result(
        result,
        eda_evidence_pack=pack.model_dump(mode="json"),
        source_evidence=[],
    )

    assert sha256_contract(pack) == before


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Stage 3 (synthesis failure taxonomy) must expose distinct schema, evidence-space, "
        "provider, and manifest-mismatch classifications; remove strict xfail then"
    ),
)
def test_synthesis_failure_taxonomy_distinguishes_all_boundary_failures() -> None:
    diagnostics = importlib.import_module("kaggle_researcher.contracts.synthesis_failures")
    assert diagnostics.classifications() >= {
        "llm_schema_error",
        "evidence_space_mismatch",
        "provider_failure",
        "internal_manifest_mismatch",
    }
