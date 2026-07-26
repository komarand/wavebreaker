from __future__ import annotations

from dataclasses import replace

import pytest

from kaggle_researcher.contracts.eda import EdaEvidencePack
from kaggle_researcher.contracts.errors import EvidenceManifestConflictError
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceConflictPolicy,
    build_evidence_reference_manifest,
    publish_eda_evidence_bundle,
)
from kaggle_researcher.contracts.hashing import sha256_contract
from kaggle_researcher.contracts.registries import build_contract_registries
from tests.fixtures.evidence_contract import representative_evidence_pack, stage_bundle


pytestmark = pytest.mark.contract


def test_same_final_pack_produces_identical_versioned_hashes() -> None:
    pack = representative_evidence_pack()
    first = publish_eda_evidence_bundle(pack)
    second = publish_eda_evidence_bundle(pack.model_copy(deep=True))

    assert first.pack_hash == second.pack_hash
    assert first.manifest_hash == second.manifest_hash
    assert first.bundle_hash == second.bundle_hash
    assert first.evidence_manifest.hash_policy_version == "1.0"
    assert first.evidence_manifest.model_dump(mode="json") == second.evidence_manifest.model_dump(mode="json")
    with pytest.raises(Exception, match="frozen"):
        first.pack_hash = "0" * 64


def test_publication_invokes_authoritative_builder_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import kaggle_researcher.contracts.evidence_manifest as manifest_module

    calls = 0
    original = manifest_module.build_evidence_reference_manifest

    def counted(pack: EdaEvidencePack, **kwargs):
        nonlocal calls
        calls += 1
        return original(pack, **kwargs)

    monkeypatch.setattr(manifest_module, "build_evidence_reference_manifest", counted)
    manifest_module.publish_eda_evidence_bundle(representative_evidence_pack())
    assert calls == 1


def test_published_registry_consumes_manifest_without_legacy_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kaggle_researcher.contracts.evidence as evidence_module

    research, eda, reasoning, _ = stage_bundle()
    published = publish_eda_evidence_bundle(eda.evidence_pack)
    published_stage = replace(
        eda,
        evidence_pack=published.evidence_pack,
        evidence_manifest=published.evidence_manifest,
        published_bundle=published,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("published registry must not regenerate the evidence space")

    monkeypatch.setattr(evidence_module, "build_evidence_registry", forbidden)
    registries = build_contract_registries(
        research=research,
        eda=published_stage,
        reasoning=reasoning,
    )
    expected = {
        entry.ref for entry in published.evidence_manifest.entries
        if entry.available and entry.namespace == "eda_evidence"
    }
    assert set(registries.evidence.ids("eda_evidence")) == expected


def test_final_scalar_change_changes_pack_manifest_and_bundle_hashes() -> None:
    original = representative_evidence_pack()
    changed = original.model_copy(deep=True)
    changed.baseline_evidence["metric_value"] = 0.62

    before = publish_eda_evidence_bundle(original)
    after = publish_eda_evidence_bundle(changed)
    assert before.pack_hash != after.pack_hash
    assert before.manifest_hash != after.manifest_hash
    assert before.bundle_hash != after.bundle_hash


def test_manifest_contains_direct_semantic_and_contract_refs_with_fixed_namespace() -> None:
    manifest = build_evidence_reference_manifest(representative_evidence_pack())
    by_ref = manifest.entries_by_ref

    assert by_ref["baseline_evidence.metric_value"][0].reference_kind == "direct_path"
    semantic = by_ref["eda_risks.risk-shift"][0]
    assert semantic.reference_kind == "semantic_ref"
    assert semantic.canonical_path == "eda_risks[0]"
    assert semantic.namespace == "eda_evidence"
    contract = by_ref["risk-shift"][0]
    assert contract.reference_kind == "contract_ref"
    assert contract.namespace == "risk"
    assert manifest.namespace_for("risk-shift") == "risk"


def test_exact_contract_duplicates_collapse_deterministically() -> None:
    pack = representative_evidence_pack()
    pack.eda_risk_register = list(pack.eda_risks)
    manifest = build_evidence_reference_manifest(pack)
    contract_entries = [
        entry for entry in manifest.entries
        if entry.ref == "risk-shift" and entry.reference_kind == "contract_ref"
    ]

    assert len(contract_entries) == 1
    assert contract_entries[0].metadata["canonical_aliases"] == [
        "eda_risk_register[0].risk_id",
        "eda_risks[0].risk_id",
    ]


def test_duplicate_semantic_identity_is_structured_and_unavailable() -> None:
    payload = representative_evidence_pack().model_dump(mode="python")
    payload["eda_risks"].append(dict(payload["eda_risks"][0]))
    manifest = build_evidence_reference_manifest(EdaEvidencePack.model_validate(payload))

    conflict = next(item for item in manifest.conflicts if item.ref == "eda_risks.risk-shift")
    assert conflict.conflict_type == "duplicate_identity"
    assert conflict.severity == "error"
    ambiguous = manifest.entries_by_ref["eda_risks.risk-shift"][0]
    assert ambiguous.available is False
    # Unavailable aliases still preserve a deterministic canonical path for auditability.
    assert ambiguous.canonical_path == "eda_risks[0]"
    assert "eda_risks.risk-shift" not in manifest.available_refs


def test_cross_namespace_collision_is_structured_and_owned_refs_are_unavailable() -> None:
    payload = representative_evidence_pack().model_dump(mode="python")
    payload["hypothesis_results"] = [{
        "hypothesis_id": "risk-shift",
        "category": "validation",
        "status": "confirmed",
        "confidence_after_eda": "high",
        "finding": "The split is usable.",
        "impact_on_strategy": "Use fixed folds.",
    }]
    manifest = build_evidence_reference_manifest(EdaEvidencePack.model_validate(payload))
    conflict = next(item for item in manifest.conflicts if item.ref == "risk-shift")

    assert conflict.conflict_type == "cross_namespace_collision"
    assert conflict.namespaces == ["hypothesis", "risk"]
    assert all(not entry.available for entry in manifest.entries_by_ref["risk-shift"])
    assert manifest.namespace_for("risk-shift") is None


def test_strict_policy_blocks_and_exposes_diagnostic_manifest() -> None:
    payload = representative_evidence_pack().model_dump(mode="python")
    payload["eda_risks"].append(dict(payload["eda_risks"][0]))
    pack = EdaEvidencePack.model_validate(payload)

    with pytest.raises(EvidenceManifestConflictError) as raised:
        publish_eda_evidence_bundle(pack, conflict_policy=EvidenceConflictPolicy.STRICT)

    assert raised.value.stage == "eda_publication_boundary"
    assert raised.value.manifest.conflicts[0].severity == "error"
    assert "eda_risks.risk-shift" not in raised.value.manifest.available_refs


def test_degraded_policy_publishes_without_ambiguous_refs() -> None:
    payload = representative_evidence_pack().model_dump(mode="python")
    payload["eda_risks"].append(dict(payload["eda_risks"][0]))
    bundle = publish_eda_evidence_bundle(
        EdaEvidencePack.model_validate(payload),
        conflict_policy=EvidenceConflictPolicy.DEGRADED,
    )

    assert "eda_risks.risk-shift" not in bundle.evidence_manifest.available_refs
    assert any("degraded evidence manifest" in item for item in bundle.warnings)
    assert any("Ambiguous evidence references" in item for item in bundle.limitations)
    assert bundle.pack_hash == sha256_contract(bundle.evidence_pack)
