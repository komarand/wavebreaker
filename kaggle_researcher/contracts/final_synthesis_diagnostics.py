from __future__ import annotations

from typing import Literal

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


class FinalSynthesisDiagnostics(DiagnosticsModel):
    schema_version: str = "1.0"
    competition_id: str
    attempts: list[SynthesisAttemptDiagnostic] = Field(default_factory=list)
    initial_output_valid: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    fallback_required: bool = False
    fallback_reason: str | None = None


__all__ = [
    "FinalSynthesisDiagnostics",
    "SynthesisAttempt",
    "SynthesisAttemptDiagnostic",
    "ValidationIssue",
    "ValidationStage",
]
