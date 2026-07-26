from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping, get_args


ContractFamily = Literal[
    "research_hypotheses",
    "eda_task_plan",
    "eda_evidence_pack",
    "evidence_reference_manifest",
    "published_eda_evidence_bundle",
    "final_synthesis_context",
    "validation_result",
    "metric_result",
    "leakage_risk_result",
    "leaderboard_audit_result",
    "experiment_plan",
    "skeptical_review",
    "final_strategy",
    "strategy_selection_draft",
    "strategy_rendering_draft",
    "strategy_skeleton",
    "run_manifest",
]

CURRENT_SCHEMA_VERSION = "1.0"
CURRENT_CONTRACT_VERSIONS: Mapping[str, str] = MappingProxyType({
    family: "2.0" if family in {
        "final_strategy", "strategy_selection_draft", "strategy_rendering_draft",
        "strategy_skeleton",
    } else CURRENT_SCHEMA_VERSION
    for family in get_args(ContractFamily)
})


def current_version(family: ContractFamily) -> str:
    return CURRENT_CONTRACT_VERSIONS[family]
