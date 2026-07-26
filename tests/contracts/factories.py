from __future__ import annotations

from copy import deepcopy
from typing import Any

from kaggle_researcher.contracts.eda import EdaTaskPlan
from kaggle_researcher.contracts.evidence_manifest import build_evidence_reference_manifest
from kaggle_researcher.contracts.reference_catalog import (
    build_final_strategy_reference_catalog as _build_final_strategy_reference_catalog,
)
from kaggle_researcher.contracts.research import ResearchHypotheses


CORE_SEQUENCE = [
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "metric_analyzer",
    "validation_analyzer",
    "leakage_checker",
]


def valid_research_payload(*, competition_id: str = "fixture-competition") -> dict[str, Any]:
    return {
        "contract_family": "research_hypotheses",
        "schema_version": "1.0",
        "competition_id": competition_id,
        "created_at": "2026-01-02T03:04:05+00:00",
        "hypotheses": [
            {
                "hypothesis_id": "schema_core",
                "category": "schema",
                "claim": "Check whether generic train/test roles and target columns can be inferred.",
                "rationale": "Schema evidence is required before downstream analysis.",
                "expected_eda_checks": ["schema_inferer.detect_target"],
                "priority": "P0",
                "confidence_before_eda": "medium",
                "source_refs": [],
                "status": "needs_eda",
                "limitations": [],
            },
            {
                "hypothesis_id": "metric_core",
                "category": "metric",
                "claim": "EDA should verify the metric and required prediction semantics.",
                "rationale": "Metric semantics constrain evaluation.",
                "expected_eda_checks": ["metric_analyzer.resolve_metric"],
                "priority": "P0",
                "confidence_before_eda": "medium",
                "source_refs": [],
                "status": "needs_eda",
                "limitations": [],
            },
            {
                "hypothesis_id": "val_core",
                "category": "validation",
                "claim": "EDA should verify an evidence-supported primary validation policy.",
                "rationale": "The split policy must be selected from observed structure.",
                "expected_eda_checks": ["validation_analyzer.select_primary_validation"],
                "priority": "P0",
                "confidence_before_eda": "medium",
                "source_refs": [],
                "status": "needs_eda",
                "limitations": [],
            },
            {
                "hypothesis_id": "leak_core",
                "category": "leakage",
                "claim": "Check whether generic train/test overlap or target proxies exist.",
                "rationale": "Leakage can invalidate validation.",
                "expected_eda_checks": ["leakage_checker.train_test_id_overlap"],
                "priority": "P0",
                "confidence_before_eda": "medium",
                "source_refs": [],
                "status": "needs_eda",
                "limitations": [],
            },
        ],
        "eda_tasks": [],
        "structured_findings": [],
        "scout_limitations": [],
        "models_used": {"fixture": "deterministic"},
    }


def valid_task_plan_payload(
    *,
    competition_id: str = "fixture-competition",
    task_type: str = "binary_classification",
    metric_name: str = "roc_auc",
) -> dict[str, Any]:
    tasks = [
        _task("inventory", "file_inventory", "schema_core", blocking=True),
        _task("schema", "schema_inferer", "schema_core", blocking=True),
        _task("profile", "table_profiler", "schema_core"),
        _task("metric", "metric_analyzer", "metric_core"),
        _task("validation", "validation_analyzer", "val_core", blocking=True),
        _task("leakage", "leakage_checker", "leak_core", blocking=True),
    ]
    return {
        "contract_family": "eda_task_plan",
        "schema_version": "1.0",
        "competition_id": competition_id,
        "task_type": task_type,
        "metric": {"name": metric_name},
        "dataset": {"download_required": False},
        "eda_tasks": tasks,
        "hypothesis_index": {
            "schema_core": ["inventory", "schema", "profile"],
            "metric_core": ["metric"],
            "val_core": ["validation"],
            "leak_core": ["leakage"],
        },
        "recommended_module_sequence": list(CORE_SEQUENCE),
        "recommended_human_checklist": ["Confirm target and metric before modeling."],
        "blocking_tasks": [
            "file_inventory", "schema_inferer", "validation_analyzer", "leakage_checker"
        ],
    }


def make_valid_research_hypotheses(
    *, competition_id: str = "fixture-competition"
) -> ResearchHypotheses:
    return ResearchHypotheses.model_validate(valid_research_payload(competition_id=competition_id))


def make_valid_eda_task_plan(
    *,
    competition_id: str = "fixture-competition",
    task_type: str = "binary_classification",
    metric_name: str = "roc_auc",
) -> EdaTaskPlan:
    return EdaTaskPlan.model_validate(valid_task_plan_payload(
        competition_id=competition_id,
        task_type=task_type,
        metric_name=metric_name,
    ))


def payload_copy(value: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(value)


def build_final_strategy_reference_catalog(evidence_pack: Any, **kwargs: Any):
    """Test helper that explicitly publishes the immutable manifest snapshot."""
    kwargs.setdefault(
        "evidence_manifest", build_evidence_reference_manifest(evidence_pack)
    )
    return _build_final_strategy_reference_catalog(evidence_pack, **kwargs)


def _task(
    task_id: str, module: str, hypothesis_id: str, *, blocking: bool = False
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "module": module,
        "priority": "P0",
        "blocking": blocking,
        "related_hypothesis_ids": [hypothesis_id],
        "dependencies": [],
        "expected_outputs": [],
        "params": {},
    }
