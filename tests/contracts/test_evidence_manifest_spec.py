from __future__ import annotations

import pytest

from kaggle_researcher.contracts.errors import EvidenceManifestPackMismatchError
from kaggle_researcher.contracts.evidence_manifest import (
    publish_eda_evidence_bundle,
    validate_published_eda_evidence_bundle,
)
from kaggle_researcher.contracts.hashing import canonical_contract_hash
from tests.fixtures.evidence_contract import representative_evidence_pack


pytestmark = pytest.mark.contract


def test_mutated_pack_fails_with_structured_hash_mismatch_before_model_call() -> None:
    original = representative_evidence_pack()
    bundle = publish_eda_evidence_bundle(original)
    expected_hash = canonical_contract_hash(original)
    bundle.evidence_pack.baseline_evidence["metric_value"] = 0.99
    actual_hash = canonical_contract_hash(bundle.evidence_pack)
    provider_calls = 0

    def provider(_: object) -> None:
        nonlocal provider_calls
        provider_calls += 1

    def synthesis_boundary() -> None:
        validate_published_eda_evidence_bundle(bundle)
        provider(bundle)

    with pytest.raises(EvidenceManifestPackMismatchError) as raised:
        synthesis_boundary()

    assert provider_calls == 0
    assert raised.value.expected_hash == expected_hash
    assert raised.value.actual_hash == actual_hash
    assert raised.value.classification == "internal_contract_error"
    assert raised.value.classification != "llm_validation_failure"
