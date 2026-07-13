from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


HYPOTHESIS_CATEGORY_ALIASES = {
    "relationships": "relationship",
    "feature_engineering": "feature",
    "dataset_schema": "schema",
    "notebook_reverse_engineering": "notebook",
}

EXPERIMENT_EVIDENCE_ID_ALIASES = {
    "validation_policy": "validation_result",
    "primary_validation": "validation_result",
    "validation_strategy": "validation_result",
}

# Null normalization is intentionally allowlisted. Missing values continue to use
# model defaults, and unrelated optional objects remain None.
NULL_COLLECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "ValidationResult": {
        "evidence_ids": [],
        "failure_modes": [],
        "do_not_use": [],
        "policy_notes": [],
    },
    "LeakageRiskResult": {
        "evidence_ids": [],
        "possible_issues": [],
        "recommended_checks": [],
    },
    "MetricResult": {"evidence_ids": []},
    "LeaderboardAuditResult": {"evidence_ids": [], "warnings": []},
    "ExperimentItem": {"evidence_ids": []},
    "ReviewResult": {
        "evidence_ids": [],
        "unsupported_claims": [],
        "too_generic": [],
        "unnecessary_experiments": [],
        "approved_experiment_ids": [],
        "rejected_experiment_ids": [],
        "revised_sections": {},
    },
    "FinalStrategyAction": {
        "evidence_refs": [],
        "related_hypothesis_ids": [],
        "experiment_ids": [],
        "source_refs": [],
        "eda_result_refs": [],
        "limitations": [],
    },
    "FinalStrategySection": {
        "actions": [],
        "evidence_refs": [],
        "related_hypothesis_ids": [],
    },
    "FinalStrategyResult": {
        "sections": [],
        "actions": [],
        "source_to_hypothesis_links": [],
        "hypothesis_to_eda_links": [],
        "limitations": [],
        "models_used": {},
    },
}


def normalize_contract_payload(payload: Any, contract_name: str) -> Any:
    """Apply only deterministic, registered normalizations for one contract."""

    if not isinstance(payload, Mapping):
        return payload
    normalized = dict(payload)
    for field_name, default in NULL_COLLECTION_DEFAULTS.get(contract_name, {}).items():
        if field_name in normalized and normalized[field_name] is None:
            normalized[field_name] = deepcopy(default)
    return normalized


def normalize_experiment_evidence_id(reference_id: str) -> str:
    return EXPERIMENT_EVIDENCE_ID_ALIASES.get(reference_id, reference_id)
