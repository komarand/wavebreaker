from __future__ import annotations

import json

import pytest

from kaggle_researcher.contracts.artifacts import (
    load_eda_publication_bundle,
    load_published_eda_evidence_bundle,
    write_json_atomic,
)
from kaggle_researcher.contracts.errors import (
    EvidenceManifestBuildError,
    EvidenceManifestPackMismatchError,
)
from kaggle_researcher.contracts.evidence import resolve_evidence_path
from kaggle_researcher.contracts.evidence_manifest import (
    EvidenceReferenceManifest,
    PublishedEdaEvidenceBundle,
    publish_eda_evidence_bundle,
    validate_published_eda_bundle,
)
from kaggle_researcher.contracts.hashing import sha256_contract
from tests.fixtures.evidence_contract import representative_evidence_pack


pytestmark = pytest.mark.contract


def _rehash_manifest_and_bundle(
    bundle: PublishedEdaEvidenceBundle,
    manifest: EvidenceReferenceManifest,
) -> PublishedEdaEvidenceBundle:
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload.pop("manifest_hash")
    manifest = EvidenceReferenceManifest(
        **manifest_payload,
        manifest_hash=sha256_contract(manifest_payload),
    )
    identity = {
        "contract_family": bundle.contract_family,
        "schema_version": bundle.schema_version,
        "pack_hash": bundle.pack_hash,
        "manifest_hash": manifest.manifest_hash,
    }
    return bundle.model_copy(update={
        "evidence_manifest": manifest,
        "manifest_hash": manifest.manifest_hash,
        "bundle_hash": sha256_contract(identity),
    })


def test_canonical_path_missing_is_rejected_even_with_consistent_hashes() -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    entries = list(bundle.evidence_manifest.entries)
    entries[0] = entries[0].model_copy(update={"canonical_path": None})
    manifest = bundle.evidence_manifest.model_copy(update={"entries": entries})
    invalid = _rehash_manifest_and_bundle(bundle, manifest)

    with pytest.raises(EvidenceManifestBuildError, match="invariant validation failed") as raised:
        validate_published_eda_bundle(invalid)

    assert "evidence_manifest.entries[0].canonical_path" in raised.value.field_paths


def test_pack_and_manifest_from_different_runs_are_rejected() -> None:
    first = publish_eda_evidence_bundle(representative_evidence_pack())
    other_pack = representative_evidence_pack().model_copy(update={"run_id": "other-run"})
    second = publish_eda_evidence_bundle(other_pack)
    mixed = first.model_copy(update={
        "evidence_manifest": second.evidence_manifest,
        "manifest_hash": second.manifest_hash,
        "bundle_hash": sha256_contract({
            "contract_family": first.contract_family,
            "schema_version": first.schema_version,
            "pack_hash": first.pack_hash,
            "manifest_hash": second.manifest_hash,
        }),
    })

    with pytest.raises(EvidenceManifestPackMismatchError):
        validate_published_eda_bundle(mixed)


def test_tampered_pack_and_manifest_are_rejected() -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    tampered_pack = bundle.evidence_pack.model_copy(deep=True)
    tampered_pack.baseline_evidence["metric_value"] = 0.99
    with pytest.raises(EvidenceManifestPackMismatchError):
        validate_published_eda_bundle(bundle.model_copy(update={"evidence_pack": tampered_pack}))

    entries = list(bundle.evidence_manifest.entries)
    entries[0] = entries[0].model_copy(update={"metadata": {"tampered": True}})
    tampered_manifest = bundle.evidence_manifest.model_copy(update={"entries": entries})
    with pytest.raises(EvidenceManifestBuildError, match="manifest hash mismatch"):
        validate_published_eda_bundle(bundle.model_copy(update={
            "evidence_manifest": tampered_manifest,
        }))


def test_disk_load_runs_bundle_validation(tmp_path) -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    payload = bundle.model_dump(mode="json")
    payload["evidence_pack"]["baseline_evidence"]["metric_value"] = 0.99
    path = tmp_path / "published_eda_evidence_bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceManifestPackMismatchError):
        load_published_eda_evidence_bundle(path)


def test_round_trip_preserves_all_three_hashes(tmp_path) -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    path = tmp_path / "published_eda_evidence_bundle.json"
    write_json_atomic(path, bundle)

    loaded = load_published_eda_evidence_bundle(path)
    result = validate_published_eda_bundle(loaded)

    assert (loaded.pack_hash, loaded.manifest_hash, loaded.bundle_hash) == (
        bundle.pack_hash, bundle.manifest_hash, bundle.bundle_hash,
    )
    assert result.valid is True
    assert result.available_entry_count > 0


def test_all_available_entries_resolve_against_embedded_pack() -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())

    for entry in bundle.evidence_manifest.entries:
        if entry.available:
            resolved = resolve_evidence_path(entry.canonical_path, bundle.evidence_pack)
            assert resolved.reference == entry.canonical_path


def test_dictionary_keys_with_dots_use_resolvable_quoted_canonical_paths() -> None:
    pack = representative_evidence_pack().model_copy(deep=True)
    pack.inferred_schema["train.csv"] = {"role": "train"}

    bundle = publish_eda_evidence_bundle(pack)
    entry = next(
        item for item in bundle.evidence_manifest.entries
        if item.ref == 'inferred_schema["train.csv"].role'
    )

    assert entry.canonical_path == 'inferred_schema["train.csv"].role'
    assert resolve_evidence_path(entry.canonical_path, bundle.evidence_pack).value == "train"


def test_legacy_separate_artifacts_require_matching_hashes(tmp_path) -> None:
    first = publish_eda_evidence_bundle(representative_evidence_pack())
    second_pack = representative_evidence_pack().model_copy(update={"run_id": "other-run"})
    write_json_atomic(tmp_path / "eda_evidence_pack.json", second_pack)
    write_json_atomic(
        tmp_path / "evidence_reference_manifest.json",
        first.evidence_manifest,
    )

    with pytest.raises(EvidenceManifestPackMismatchError):
        load_eda_publication_bundle(tmp_path)


def test_legacy_matching_artifacts_migrate_with_warning(tmp_path) -> None:
    bundle = publish_eda_evidence_bundle(representative_evidence_pack())
    write_json_atomic(tmp_path / "eda_evidence_pack.json", bundle.evidence_pack)
    write_json_atomic(tmp_path / "evidence_reference_manifest.json", bundle.evidence_manifest)

    migrated, warnings = load_eda_publication_bundle(tmp_path)

    assert migrated.bundle_hash == bundle.bundle_hash
    assert warnings and "Verified and migrated separate" in warnings[0]
