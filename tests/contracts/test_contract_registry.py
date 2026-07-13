from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kaggle_researcher.contracts.eda_task_plan import EdaTaskPlan
from kaggle_researcher.contracts.registry import CONTRACT_DEFINITIONS, contract_by_id
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
        "ResearchHypotheses": [Path("kaggle_researcher/contracts/research_hypotheses.py")],
        "EdaTaskPlan": [Path("kaggle_researcher/contracts/eda_task_plan.py")],
    }
