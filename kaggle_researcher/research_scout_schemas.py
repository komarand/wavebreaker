from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProvenanceLabel = Literal[
    "kaggle",
    "arxiv",
    "github",
    "huggingface_papers",
    "domain_memory",
    "heuristic",
    "not_verified_on_data",
]

Priority = Literal["P0", "P1", "P2", "P3"]

HypothesisStatus = Literal[
    "untested",
    "source_supported",
    "heuristic",
    "needs_eda",
]

HypothesisCategory = Literal[
    "validation",
    "leakage",
    "metric",
    "dataset_schema",
    "relationships",
    "drift",
    "feature_engineering",
    "baseline",
    "notebook_reverse_engineering",
    "leaderboard_risk",
]

EdaModule = Literal[
    "file_inventory",
    "schema_inferer",
    "table_profiler",
    "relationship_inferer",
    "validation_analyzer",
    "leakage_checker",
    "drift_analyzer",
    "metric_analyzer",
    "baseline_runner",
    "feature_probe",
    "notebook_reverse_engineering",
]

FindingType = Literal[
    "observed_in_sources",
    "recommendation",
    "warning",
    "caveat",
    "limitation",
]


class VerificationStep(BaseModel):
    id: str = Field(min_length=1)
    module: EdaModule
    operation: str = Field(min_length=3)
    question: str = Field(min_length=10)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_criteria: list[str] = Field(default_factory=list)
    related_task_ids: list[str] = Field(default_factory=list)


class ScoutFinding(BaseModel):
    id: str = Field(min_length=1)
    finding_type: FindingType
    claim: str = Field(min_length=10)
    implication: str | None = None
    caveat: str | None = None
    provenance: list[ProvenanceLabel] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    related_hypothesis_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


class ResearchHypothesis(BaseModel):
    id: str
    category: HypothesisCategory
    priority: Priority
    claim: str
    why_it_matters: str
    how_to_verify: list[str] = Field(default_factory=list)
    verification_steps: list[VerificationStep] = Field(default_factory=list)
    expected_evidence_keys: list[str] = Field(default_factory=list)
    failure_condition: str | None = None
    success_condition: str | None = None
    provenance: list[ProvenanceLabel] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    status: HypothesisStatus = "untested"


class EdaTask(BaseModel):
    id: str = Field(min_length=1)
    priority: Priority
    module: EdaModule
    question: str = Field(min_length=10)
    rationale: str = Field(min_length=10)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(min_length=1)
    related_hypothesis_ids: list[str] = Field(default_factory=list)
    blocking: bool = False


class ResearchHypothesesPayload(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    competition_url: str | None = None
    competition_desc: str
    task_type: str
    metric: dict
    domain: str | None = None

    source_summary: dict = Field(default_factory=dict)
    source_quality_summary: dict | None = None

    hypotheses: list[ResearchHypothesis] = Field(default_factory=list)
    eda_tasks: list[EdaTask] = Field(default_factory=list)

    structured_findings: list[ScoutFinding] = Field(default_factory=list)
    scout_findings: list[str] = Field(default_factory=list)
    scout_limitations: list[str] = Field(default_factory=list)
    category_corrections: list[dict] = Field(default_factory=list)
    recommended_module_sequence: list[str] = Field(default_factory=list)
    recommended_human_checklist: list[str] = Field(default_factory=list)
    recommended_eda_sequence: list[str] = Field(default_factory=list)

    models_used: dict = Field(default_factory=dict)
