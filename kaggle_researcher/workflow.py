from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


WorkflowStatus = Literal["success", "completed_with_degradation", "failed"]
FinalSynthesisStageStatus = Literal[
    "success",
    "repaired_success",
    "degraded_fallback",
    "failed",
]


class FinalSynthesisDegradedError(RuntimeError):
    """Raised after degraded synthesis artifacts have been persisted in strict mode."""

    def __init__(
        self,
        diagnostics_path: str | Path,
        *,
        result: Any | None = None,
    ) -> None:
        self.diagnostics_path = Path(diagnostics_path)
        self.result = result
        super().__init__(
            "Final synthesis produced a deterministic degraded fallback while valid "
            "synthesis was required. Diagnostics: " + str(self.diagnostics_path)
        )


__all__ = [
    "FinalSynthesisDegradedError",
    "FinalSynthesisStageStatus",
    "WorkflowStatus",
]
