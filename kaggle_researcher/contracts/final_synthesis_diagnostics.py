from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ValidationStage = Literal[
    "llm_parse",
    "llm_schema_validation",
    "llm_reference_validation",
    "repair_parse",
    "repair_schema_validation",
    "repair_reference_validation",
]
SynthesisAttempt = Literal["initial_llm", "deterministic_repair"]


class DiagnosticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationIssue(DiagnosticsModel):
    stage: ValidationStage
    issue_type: str
    field_path: str | None = None
    message: str
    invalid_value_type: str | None = None
    invalid_reference: str | None = None
    expected_contract: str | None = None


class SynthesisAttemptDiagnostic(DiagnosticsModel):
    attempt: SynthesisAttempt
    model: str | None = None
    output_received: bool = False
    json_parse_succeeded: bool = False
    schema_validation_succeeded: bool = False
    reference_validation_succeeded: bool = False
    output_hash: str | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SelectionAttemptDiagnostic(DiagnosticsModel):
    attempt: Literal["initial", "repair"]
    model: str | None = None
    prompt_version: str
    prompt_fingerprint: str
    response_hash: str | None = None
    parse_succeeded: bool = False
    schema_succeeded: bool = False
    reference_validation_succeeded: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RenderingAttemptDiagnostic(DiagnosticsModel):
    attempt: Literal["initial", "repair"]
    model: str | None = None
    prompt_version: str
    prompt_fingerprint: str
    skeleton_id: str
    skeleton_hash: str
    response_hash: str | None = None
    parse_succeeded: bool = False
    integrity_validation_succeeded: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BridgeDiagnostic(DiagnosticsModel):
    initial_action_count: int = 0
    canonical_action_count: int = 0
    duplicates_merged: int = 0
    evidence_refs_removed: int = 0
    evidence_refs_retained: int = 0
    source_links_preserved: int = 0
    hypothesis_roles_reassigned: int = 0
    self_model_comparisons_removed: int = 0
    feature_experiments_grouped: int = 0
    candidate_experiment_count: int = 0
    core_experiment_count: int = 0
    backlog_experiment_count: int = 0
    first_48h_experiment_count: int = 0
    dependency_repairs: int = 0
    quality_gate_issues: list[str] = Field(default_factory=list)
    client_key_map: dict[str, dict[str, str]] = Field(default_factory=dict)


class FinalSynthesisDiagnostics(DiagnosticsModel):
    schema_version: str = "2.0"
    competition_id: str
    attempts: list[SynthesisAttemptDiagnostic] = Field(default_factory=list)
    initial_output_valid: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    fallback_required: bool = False
    fallback_reason: str | None = None
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    provenance_telemetry: dict[str, int] = Field(default_factory=dict)
    protocol: Literal["monolithic_legacy", "two_call"] = "two_call"
    selection_status: str | None = None
    rendering_status: str | None = None
    selection_attempts: list[SelectionAttemptDiagnostic] = Field(default_factory=list)
    rendering_attempts: list[RenderingAttemptDiagnostic] = Field(default_factory=list)
    bridge: BridgeDiagnostic | None = None
    prompt_fingerprints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    provider_failures: list[dict[str, str]] = Field(default_factory=list)


__all__ = [
    "FinalSynthesisDiagnostics",
    "BridgeDiagnostic", "RenderingAttemptDiagnostic", "SelectionAttemptDiagnostic",
    "SynthesisAttempt",
    "SynthesisAttemptDiagnostic",
    "ValidationIssue",
    "ValidationStage",
]
