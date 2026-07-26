from __future__ import annotations

import json
import pytest

from kaggle_researcher.contracts.evidence import (
    AmbiguousEvidencePathError,
    generate_allowed_evidence_refs,
    generate_semantic_evidence_refs,
)
from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.errors import AmbiguousReferenceError
from kaggle_researcher.contracts.reference_catalog import build_final_strategy_reference_catalog
from kaggle_researcher.contracts.synthesis_context import build_final_synthesis_context
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceConflictPolicy,
    EvidenceReferenceConflict,
    EvidenceReferenceEntry,
    _collapse_conflicts,
    _normalize_entries,
    publish_eda_evidence_bundle,
)
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


def test_duplicate_semantic_identity_escapes_before_model_call() -> None:
    payload = representative_evidence_pack().model_dump(mode="json")
    payload["eda_risks"].append(dict(payload["eda_risks"][0]))
    model_calls = 0

    with pytest.raises(AmbiguousEvidencePathError) as raised:
        generate_allowed_evidence_refs(payload)

    assert raised.value.reference == "eda_risks.risk-shift"
    assert raised.value.match_count == 2
    assert "ambiguous_semantic_path" in str(raised.value)
    assert model_calls == 0


def test_generate_semantic_refs_reports_exact_duplicate_identity() -> None:
    payload = representative_evidence_pack().model_dump(mode="json")
    payload["feature_probe_evidence"].append(dict(payload["feature_probe_evidence"][0]))
    with pytest.raises(
        AmbiguousEvidencePathError,
        match=r"feature_probe_evidence\.aggregate_signal.*ambiguous_semantic_path",
    ):
        generate_semantic_evidence_refs(payload)


def test_cross_namespace_collision_is_reported_through_manifest() -> None:
    payload = representative_evidence_pack().model_dump(mode="python")
    collision = "risk-shift"
    payload["hypothesis_results"] = [{
        "hypothesis_id": collision,
        "category": "validation",
        "status": "confirmed",
        "confidence_after_eda": "high",
        "finding": "The split is usable.",
        "impact_on_strategy": "Use fixed folds.",
    }]
    pack = EdaEvidencePack.model_validate(payload)
    bundle = stage_bundle(pack)
    registries = bundle[-1]
    published = publish_eda_evidence_bundle(
        pack, conflict_policy=EvidenceConflictPolicy.DEGRADED
    )
    context = build_final_synthesis_context(
        competition_desc="Generic classification.", research=bundle[0],
        published_eda_bundle=published,
        reasoning=bundle[2], registries=registries, eda_summary_text="# EDA",
    )
    issues = _final_reference_issues(
        strategy_citing([collision]), context, registries
    )
    assert len(issues) == 2
    assert {issue.reason for issue in issues} == {"manifest_conflict"}
    assert {issue.actual_namespace for issue in issues} == {"hypothesis,risk"}
    catalog = build_final_strategy_reference_catalog(
        pack,
        evidence_manifest=published.evidence_manifest,
        research_hypotheses=bundle[0].hypotheses,
    )
    prompt = json.loads(_build_final_synthesizer_prompt(
        context=context, registries=registries, reference_catalog=catalog
    ))
    assert collision not in prompt["allowed_evidence_refs"]
    assert collision not in prompt["allowed_eda_result_refs"]


def test_semantic_alias_collision_is_blocking_and_never_silently_collapsed() -> None:
    entries = [
        EvidenceReferenceEntry(
            ref="feature_probe_evidence.aggregate_signal",
            namespace="eda_evidence",
            reference_kind="semantic_ref",
            canonical_path=f"feature_probe_evidence[{index}]",
            value_type="object",
            source_component="feature_probe_evidence",
            semantic_identity="aggregate_signal",
            metadata={"identity_field": "feature_family"},
        )
        for index in (0, 1)
    ]
    normalized, conflicts = _normalize_entries(entries)

    assert len(normalized) == 1
    assert normalized[0].available is False
    assert conflicts[0].conflict_type == "semantic_alias_collision"
    assert conflicts[0].severity == "error"
    assert conflicts[0].canonical_paths == [
        "feature_probe_evidence[0]", "feature_probe_evidence[1]",
    ]


def test_exact_duplicate_registration_collapses_with_observable_warning() -> None:
    entry = EvidenceReferenceEntry(
        ref="baseline_evidence.metric_value",
        namespace="eda_evidence",
        reference_kind="direct_path",
        canonical_path="baseline_evidence.metric_value",
        value_type="number",
        source_component="baseline_evidence",
    )
    normalized, conflicts = _normalize_entries([entry, entry.model_copy(deep=True)])

    assert normalized == [entry]
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "exact_duplicate_registration"
    assert conflicts[0].severity == "warning"


def test_distinct_conflicts_with_same_ref_and_type_are_not_swallowed() -> None:
    conflicts = [
        EvidenceReferenceConflict(
            ref="shared",
            namespaces=["eda_evidence"],
            canonical_paths=[path],
            conflict_type="semantic_alias_collision",
            severity="error",
            message=f"Collision at {path}.",
        )
        for path in ("first[0]", "second[0]")
    ]

    assert _collapse_conflicts([*conflicts, conflicts[0]]) == conflicts


def test_unique_legacy_unqualified_reference_migrates_to_structured_identity() -> None:
    manifest = publish_eda_evidence_bundle(representative_evidence_pack()).evidence_manifest

    migrated = manifest.migrate_legacy_ref("risk-shift")

    assert migrated.model_dump(mode="json") == {"namespace": "risk", "ref": "risk-shift"}
    assert manifest.entry_for(migrated).canonical_path == "eda_risks[0].risk_id"


def test_ambiguous_legacy_unqualified_reference_is_rejected() -> None:
    payload = representative_evidence_pack().model_dump(mode="python")
    payload["hypothesis_results"] = [{
        "hypothesis_id": "risk-shift",
        "category": "validation",
        "status": "confirmed",
        "confidence_after_eda": "high",
        "finding": "The split is usable.",
        "impact_on_strategy": "Use fixed folds.",
    }]
    published = publish_eda_evidence_bundle(
        EdaEvidencePack.model_validate(payload),
        conflict_policy=EvidenceConflictPolicy.DEGRADED,
    )

    with pytest.raises(AmbiguousReferenceError, match="ambiguous namespace ownership"):
        published.evidence_manifest.migrate_legacy_ref("risk-shift")
