from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping, get_args


ContractFamily = Literal[
    "research_hypotheses",
    "eda_task_plan",
    "eda_evidence_pack",
    "validation_result",
    "experiment_plan",
    "skeptical_review",
    "final_strategy",
    "run_manifest",
]

CURRENT_SCHEMA_VERSION = "1.0"
CURRENT_CONTRACT_VERSIONS: Mapping[str, str] = MappingProxyType(
    {family: CURRENT_SCHEMA_VERSION for family in get_args(ContractFamily)}
)


def current_version(family: ContractFamily) -> str:
    return CURRENT_CONTRACT_VERSIONS[family]

