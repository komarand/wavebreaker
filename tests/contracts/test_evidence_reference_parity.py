from __future__ import annotations

import json

import pytest

from kaggle_researcher.contracts.evidence import generate_allowed_evidence_refs
from kaggle_researcher.contracts.reference_catalog import build_final_strategy_reference_catalog
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.contracts.evidence_manifest import publish_eda_evidence_bundle
from kaggle_researcher.reasoning.final_synthesizer import (
    _build_final_synthesizer_prompt,
    _final_reference_issues,
)
from tests.fixtures.evidence_contract import (
    representative_evidence_pack,
    stage_bundle,
    strategy_citing,
)


pytestmark = pytest.mark.contract


def _context(bundle):
    research, eda, reasoning, registries = bundle
    return build_final_synthesis_context(
        competition_desc="Generic classification.", research=research,
        published_eda_bundle=publish_eda_evidence_bundle(eda.evidence_pack),
        reasoning=reasoning, registries=registries, eda_summary_text="# EDA",
    )


def test_prompt_allowlist_is_accepted_by_final_reference_validator() -> None:
    bundle = stage_bundle()
    context = _context(bundle)
    registries = bundle[-1]
    catalog = build_final_strategy_reference_catalog(
        bundle[1].evidence_pack,
        evidence_manifest=bundle[1].evidence_manifest,
        research_hypotheses=bundle[0].hypotheses,
        source_claim_ids=[],
    )
    prompt = _build_final_synthesizer_prompt(
        context=context, registries=registries, reference_catalog=catalog
    )
    prompt_payload = json.loads(prompt)
    shown_evidence = prompt_payload["allowed_evidence_refs"]
    shown_eda = prompt_payload["allowed_eda_result_refs"]
    issues = _final_reference_issues(
        strategy_citing(shown_evidence, eda_result_refs=shown_eda),
        context,
        registries,
    )

    assert shown_evidence == context.allowed_evidence_refs
    assert shown_eda == context.allowed_eda_result_refs
    dumped = context.model_dump(mode="json")
    assert "allowed_evidence_refs" not in dumped
    assert "allowed_eda_result_refs" not in dumped
    assert dumped["published_eda_bundle"]["manifest_hash"] == context.manifest_hash
    assert prompt_payload["evidence_manifest_metadata"]["manifest_hash"] == context.manifest_hash
    assert not [issue for issue in issues if issue.reason in {"unknown", "cross_namespace"}]


def test_manifest_validator_reports_unknown_and_namespace_mismatch_precisely() -> None:
    bundle = stage_bundle()
    context = _context(bundle)
    registries = bundle[-1]

    unknown = _final_reference_issues(
        strategy_citing(["not-published"]), context, registries
    )
    assert {issue.reason for issue in unknown} == {"unknown_reference"}

    mismatch = _final_reference_issues(
        strategy_citing(["risk-shift"]), context, registries
    )
    assert {issue.reason for issue in mismatch} == {"namespace_mismatch"}
    assert {issue.actual_namespace for issue in mismatch} == {"risk"}


def test_later_pack_snapshot_diverges_from_context_and_registry_deterministically() -> None:
    bundle = stage_bundle()
    context = _context(bundle)
    registries = bundle[-1]
    old_ref = "baseline_evidence.metric_value"
    assert old_ref in context.allowed_eda_result_refs

    later = bundle[1].evidence_pack.model_copy(deep=True)
    later.baseline_evidence = {"replacement_metric_value": 0.62}
    later_refs = set(generate_allowed_evidence_refs(later))

    assert old_ref not in later_refs
    assert old_ref in registries.evidence.ids("eda_evidence")
    assert _final_reference_issues(
        strategy_citing([old_ref]), context, registries
    ) == []
    assert set(context.allowed_eda_result_refs) != later_refs


def test_fixture_covers_required_reference_shapes() -> None:
    refs = set(generate_allowed_evidence_refs(representative_evidence_pack()))
    assert {
        "validation_evidence.primary_validation",
        "baseline_evidence.metric_value",
        "inferred_schema.primary_id_column",
        "eda_risks.risk-shift",
        "feature_probe_evidence.aggregate_signal",
        "feature_probe_evidence[0].details.columns[0]",
    } <= refs
