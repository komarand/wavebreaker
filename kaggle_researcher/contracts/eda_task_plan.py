"""Deprecated compatibility import for the canonical EDA task-plan contract."""

from kaggle_researcher.contracts.artifacts import (
    load_eda_task_plan,
    validate_research_artifact_bundle,
    write_eda_task_plan_atomic,
)
from kaggle_researcher.contracts.eda import EdaTask, EdaTaskPlan, HypothesisIndexEntry
from kaggle_researcher.contracts.errors import ContractError as ResearchArtifactContractError
from kaggle_researcher.contracts.migration import (
    EdaTaskPlanMigrationResult,
    migrate_eda_task_plan_payload,
)

__all__ = [
    "EdaTask",
    "EdaTaskPlan",
    "EdaTaskPlanMigrationResult",
    "HypothesisIndexEntry",
    "ResearchArtifactContractError",
    "load_eda_task_plan",
    "migrate_eda_task_plan_payload",
    "validate_research_artifact_bundle",
    "write_eda_task_plan_atomic",
]
