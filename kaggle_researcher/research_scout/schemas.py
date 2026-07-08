from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator


Confidence = Literal["low", "medium", "high"]
Priority = Literal["P0", "P1", "P2", "P3"]
HypothesisCategory = Literal[
    "schema",
    "metric",
    "validation",
    "leakage",
    "relationship",
    "drift",
    "baseline",
    "feature",
    "notebook",
    "leaderboard",
    "data_quality",
]
FindingCategory = HypothesisCategory
LimitationSeverity = Literal["low", "medium", "high"]
EdaModule = Literal[
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "metric_analyzer",
    "validation_analyzer",
    "leakage_checker",
    "relationship_inferer",
    "drift_analyzer",
    "baseline_runner",
    "feature_probe",
    "notebook_static_analysis",
]

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

CATEGORY_PREFIXES = {
    "schema": ("schema_",),
    "metric": ("metric_",),
    "validation": ("val_", "validation_"),
    "leakage": ("leak_", "leakage_"),
    "relationship": ("rel_", "relationship_"),
    "drift": ("drift_",),
    "baseline": ("base_", "baseline_"),
    "feature": ("feat_", "feature_"),
    "notebook": ("nb_", "notebook_"),
    "leaderboard": ("lb_", "leaderboard_"),
    "data_quality": ("dq_", "data_quality_"),
}
REQUIRED_CORE_HYPOTHESES = {"schema_001", "metric_001", "val_001", "leak_001"}


class ScoutHypothesis(BaseModel):
    hypothesis_id: NonEmptyString
    category: HypothesisCategory
    claim: NonEmptyString
    rationale: NonEmptyString
    expected_eda_checks: list[NonEmptyString] = Field(default_factory=list)
    priority: Priority
    confidence_before_eda: Confidence
    source_refs: list[NonEmptyString] = Field(default_factory=list)
    status: Literal["needs_eda", "source_supported", "heuristic"] = "needs_eda"

    @model_validator(mode="after")
    def _validate_stable_prefix(self) -> "ScoutHypothesis":
        prefixes = CATEGORY_PREFIXES[self.category]
        if not self.hypothesis_id.startswith(prefixes):
            prefix_text = " or ".join(prefixes)
            raise ValueError(
                f"hypothesis_id '{self.hypothesis_id}' must start with {prefix_text}"
            )
        return self


class ScoutEdaTask(BaseModel):
    task_id: NonEmptyString
    module: EdaModule
    priority: Priority
    blocking: bool = False
    related_hypothesis_ids: list[NonEmptyString] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ScoutStructuredFinding(BaseModel):
    finding_id: NonEmptyString
    category: FindingCategory
    finding: NonEmptyString
    evidence_refs: list[NonEmptyString] = Field(default_factory=list)
    source_refs: list[NonEmptyString] = Field(default_factory=list)
    confidence: Confidence = "medium"


class ScoutLimitation(BaseModel):
    limitation_id: NonEmptyString
    description: NonEmptyString
    severity: LimitationSeverity = "medium"
    affected_outputs: list[NonEmptyString] = Field(default_factory=list)


class EdaTaskPlanDraft(BaseModel):
    schema_version: str = "1.0"
    competition_id: NonEmptyString
    task_type: NonEmptyString | None = None
    metric: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)

    eda_tasks: list[ScoutEdaTask] = Field(default_factory=list)
    hypothesis_index: dict[NonEmptyString, list[NonEmptyString]] = Field(default_factory=dict)
    recommended_module_sequence: list[EdaModule] = Field(default_factory=list)
    recommended_human_checklist: list[NonEmptyString] = Field(default_factory=list)
    blocking_tasks: list[EdaModule] = Field(default_factory=list)

    @field_validator("recommended_module_sequence")
    @classmethod
    def _dedupe_sequence(cls, values: list[EdaModule]) -> list[EdaModule]:
        return _unique(values)

    def to_eda_task_plan_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "competition_id": self.competition_id,
            "task_type": self.task_type,
            "metric": self.metric,
            "dataset": self.dataset,
            "eda_tasks": [
                task.model_dump(mode="json") for task in self.eda_tasks
            ],
            "hypothesis_index": {
                str(key): list(value) for key, value in self.hypothesis_index.items()
            },
            "recommended_module_sequence": list(self.recommended_module_sequence),
            "recommended_human_checklist": list(self.recommended_human_checklist),
            "blocking_tasks": list(self.blocking_tasks),
        }


class ResearchScoutOutput(BaseModel):
    schema_version: str = "1.0"
    competition_id: NonEmptyString
    competition_url: str | None = None
    created_at: str | None = None

    task_type: NonEmptyString | None = None
    metric: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)

    hypotheses: list[ScoutHypothesis] = Field(default_factory=list)
    eda_task_plan: EdaTaskPlanDraft
    structured_findings: list[ScoutStructuredFinding] = Field(default_factory=list)
    scout_limitations: list[ScoutLimitation] = Field(default_factory=list)
    models_used: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None

    @model_validator(mode="after")
    def _validate_output(self) -> "ResearchScoutOutput":
        if self.eda_task_plan.competition_id != self.competition_id:
            raise ValueError("eda_task_plan.competition_id must match competition_id")
        hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in self.hypotheses}
        missing_core = REQUIRED_CORE_HYPOTHESES - hypothesis_ids
        if missing_core:
            raise ValueError(
                "ResearchScoutOutput missing required core hypotheses: "
                + ", ".join(sorted(missing_core))
            )
        if "drift_001" not in hypothesis_ids:
            raise ValueError("ResearchScoutOutput must include stable drift_001 hypothesis")

        task_hypothesis_ids = {
            hypothesis_id
            for task in self.eda_task_plan.eda_tasks
            for hypothesis_id in task.related_hypothesis_ids
        }
        unknown_ids = task_hypothesis_ids - hypothesis_ids
        if unknown_ids:
            raise ValueError(
                "EDA tasks reference unknown hypothesis ids: "
                + ", ".join(sorted(unknown_ids))
            )
        return self

    def to_research_hypotheses_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "competition_id": self.competition_id,
            "created_at": self.created_at or datetime.now().astimezone().isoformat(),
            "hypotheses": [
                hypothesis.model_dump(mode="json") for hypothesis in self.hypotheses
            ],
            "eda_tasks": [
                task.model_dump(mode="json") for task in self.eda_task_plan.eda_tasks
            ],
            "structured_findings": [
                finding.model_dump(mode="json") for finding in self.structured_findings
            ],
            "scout_limitations": [
                limitation.description for limitation in self.scout_limitations
            ],
            "models_used": self.models_used,
        }

    def to_eda_task_plan_payload(self) -> dict[str, Any]:
        payload = self.eda_task_plan.to_eda_task_plan_payload()
        payload["competition_id"] = self.competition_id
        payload["task_type"] = payload.get("task_type") or self.task_type
        payload["metric"] = payload.get("metric") or self.metric
        payload["dataset"] = payload.get("dataset") or self.dataset
        return payload

    def to_summary_markdown(self) -> str:
        if self.summary:
            return self.summary
        lines = [
            "# Research Scout Summary",
            "",
            f"Competition: `{self.competition_id}`",
            f"Task type: `{self.task_type or 'unknown'}`",
            f"Metric: `{self.metric.get('name', 'unknown')}`",
            "",
            "## Hypotheses",
        ]
        for hypothesis in self.hypotheses:
            lines.append(f"- {hypothesis.hypothesis_id}: {hypothesis.claim}")
        if self.scout_limitations:
            lines.extend(["", "## Limitations"])
            for limitation in self.scout_limitations:
                lines.append(f"- {limitation.description}")
        return "\n".join(lines) + "\n"

    def write_outputs(self, output_dir: Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        research_path = root / "research_hypotheses.json"
        task_plan_path = root / "eda_task_plan.json"
        summary_path = root / "research_scout_summary.md"
        research_path.write_text(
            _json_dumps(self.to_research_hypotheses_payload()),
            encoding="utf-8",
        )
        task_plan_path.write_text(
            _json_dumps(self.to_eda_task_plan_payload()),
            encoding="utf-8",
        )
        summary_path.write_text(self.to_summary_markdown(), encoding="utf-8")
        return {
            "research_hypotheses": research_path,
            "eda_task_plan": task_plan_path,
            "research_scout_summary": summary_path,
        }


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "EdaTaskPlanDraft",
    "ScoutEdaTask",
    "ScoutHypothesis",
    "ScoutLimitation",
    "ScoutStructuredFinding",
    "ResearchScoutOutput",
]
