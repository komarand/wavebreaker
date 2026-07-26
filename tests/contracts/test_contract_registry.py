from __future__ import annotations

import ast
from pathlib import Path

import pytest
from typing import Literal

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.eda_task_plan import EdaTaskPlan
from kaggle_researcher.contracts.errors import DuplicateContractRegistrationError
from kaggle_researcher.contracts.registry import (
    CONTRACT_DEFINITIONS,
    CONTRACT_REGISTRY,
    ContractHeader,
    ContractRegistry,
    contract_by_id,
)
from kaggle_researcher.contracts.research_hypotheses import ResearchHypotheses
from kaggle_researcher.eda.schemas import EdaTaskPlan as EdaEdaTaskPlan
from kaggle_researcher.eda.schemas import ResearchHypotheses as EdaResearchHypotheses


pytestmark = pytest.mark.contract


def test_contract_inventory_covers_every_full_pipeline_artifact() -> None:
    contract_ids = {definition.contract_id for definition in CONTRACT_DEFINITIONS}

    assert contract_ids == {
        "research_hypotheses",
        "eda_task_plan",
        "eda_evidence_pack",
        "evidence_reference_manifest",
        "published_eda_evidence_bundle",
        "validation_result",
        "metric_result",
        "leakage_risk_result",
        "leaderboard_audit_result",
        "experiment_plan",
        "skeptical_review",
        "final_strategy",
        "run_manifest",
        "full_run_result",
        "final_report",
    }
    assert contract_by_id("final_strategy").renderer_consumers == ("final_report",)
    assert all(definition.producer_stage and definition.consumer_stages for definition in CONTRACT_DEFINITIONS)


def test_scout_and_eda_use_the_same_canonical_contract_classes() -> None:
    assert EdaResearchHypotheses is ResearchHypotheses
    assert EdaEdaTaskPlan is EdaTaskPlan


def test_no_duplicate_canonical_class_definitions() -> None:
    definitions: dict[str, list[Path]] = {"ResearchHypotheses": [], "EdaTaskPlan": []}
    for path in Path("kaggle_researcher").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in definitions:
                definitions[node.name].append(path)

    assert definitions == {
        "ResearchHypotheses": [Path("kaggle_researcher/contracts/research.py")],
        "EdaTaskPlan": [Path("kaggle_researcher/contracts/eda.py")],
    }


def test_dispatch_registry_contains_required_public_contracts_in_sorted_order() -> None:
    assert list(CONTRACT_REGISTRY) == sorted(CONTRACT_REGISTRY)
    assert CONTRACT_REGISTRY.current_version("final_strategy") == "2.0"
    assert {
        "eda_evidence_pack",
        "evidence_reference_manifest",
        "published_eda_evidence_bundle",
        "research_hypotheses",
        "final_synthesis_context",
        "final_strategy",
        "strategy_selection_draft",
        "strategy_rendering_draft",
    }.issubset(CONTRACT_REGISTRY.families())


def test_contract_header_parses_only_strict_dispatch_fields() -> None:
    header = ContractHeader.model_validate({
        "contract_family": "eda_evidence_pack",
        "schema_version": "1.0",
        "payload": "ignored by header parser",
    })
    assert header.model_dump() == {
        "contract_family": "eda_evidence_pack",
        "schema_version": "1.0",
    }


def test_duplicate_family_version_registration_fails() -> None:
    class ExampleContract(ContractModel):
        contract_family: Literal["example"] = "example"
        schema_version: Literal["1.0"] = "1.0"

    registry = ContractRegistry()
    registry.register("example", "1.0", ExampleContract, current=True)
    with pytest.raises(DuplicateContractRegistrationError):
        registry.register("example", "1.0", ExampleContract)
