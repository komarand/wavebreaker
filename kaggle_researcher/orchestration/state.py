from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from kaggle_researcher.contracts.artifacts import (
    EdaStageResult,
    FinalStageResult,
    ReasoningStageResult,
    ResearchStageResult,
)
from kaggle_researcher.contracts.ids import StageId
from kaggle_researcher.contracts.manifest import RunManifest
from kaggle_researcher.progress import ProgressConfig


Profile = Literal["minimal", "standard", "full"]


class ReasoningClientProtocol(Protocol):
    async def chat_json(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass
class FullRunConfig:
    competition_id: str
    competition_url: str | None = None
    competition_description: str = ""
    local_dataset_path: Path | None = None
    download_dataset: bool = True
    output_root: Path = Path("runs")
    profile: Profile = "standard"
    enable_p1_modules: bool = False
    enable_baseline: bool = False
    enable_baseline_ablations: bool = False
    enable_interaction_diagnostics: bool = False
    enable_slice_diagnostics: bool = False
    enable_source_claim_validation: bool = False
    enable_visual_diagnostics: bool = False
    fail_fast: bool = False
    require_valid_final_synthesis: bool = False
    resume_run_dir: Path | None = None
    force_rerun_stages: set[str] = field(default_factory=set)
    disable_progress: bool = False


@dataclass(frozen=True)
class RuntimeServices:
    reasoning_client: ReasoningClientProtocol
    reasoning_model: str
    progress: ProgressConfig
    logger: logging.Logger


@dataclass(frozen=True)
class InputValidationResult:
    competition_id: str


@dataclass(frozen=True)
class StageFailure:
    stage_id: StageId
    message: str


class MissingStageDependencyError(RuntimeError):
    def __init__(self, *, stage_id: StageId | str, missing_dependency: StageId | str) -> None:
        self.stage_id = StageId(str(stage_id))
        self.missing_dependency = StageId(str(missing_dependency))
        super().__init__(
            f"Stage {self.stage_id!r} requires completed dependency {self.missing_dependency!r}"
        )


@dataclass
class FullRunState:
    run_dir: Path
    config: FullRunConfig
    services: RuntimeServices
    manifest: RunManifest
    input_result: InputValidationResult | None = None
    research_result: ResearchStageResult | None = None
    eda_result: EdaStageResult | None = None
    reasoning_result: ReasoningStageResult | None = None
    final_result: FinalStageResult | None = None
    optional_stage_failures: tuple[StageFailure, ...] = ()

    def require_research(self, stage_id: StageId | str) -> ResearchStageResult:
        if self.research_result is None:
            raise MissingStageDependencyError(stage_id=stage_id, missing_dependency="research_scout")
        return self.research_result

    def require_eda(self, stage_id: StageId | str) -> EdaStageResult:
        if self.eda_result is None:
            raise MissingStageDependencyError(stage_id=stage_id, missing_dependency="eda_engine")
        return self.eda_result

    def require_reasoning(self, stage_id: StageId | str) -> ReasoningStageResult:
        if self.reasoning_result is None:
            raise MissingStageDependencyError(stage_id=stage_id, missing_dependency="reasoning_context")
        return self.reasoning_result

    def require_final(self, stage_id: StageId | str) -> FinalStageResult:
        if self.final_result is None:
            raise MissingStageDependencyError(stage_id=stage_id, missing_dependency="final_strategy")
        return self.final_result


__all__ = [
    "FullRunConfig", "FullRunState", "InputValidationResult",
    "MissingStageDependencyError", "Profile", "ReasoningClientProtocol",
    "RuntimeServices", "StageFailure",
]
