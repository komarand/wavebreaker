from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def run_slice_diagnostics(baseline_evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize fold-safe baseline OOF predictions without rerunning a model."""
    baseline = baseline_evidence or {}
    artifact = _as_dict(baseline.get("artifacts")).get("oof_predictions")
    if baseline.get("status") != "completed" or not artifact or not Path(str(artifact)).is_file():
        return _skipped()
    try:
        frame = pl.read_csv(str(artifact))
    except Exception as exc:
        return {**_skipped(), "reason": "missing_oof_predictions", "warnings": [str(exc)[:240]]}
    required = {"fold", "target", "prediction"}
    if not required.issubset(frame.columns):
        return {**_skipped(), "reason": "missing_oof_predictions", "warnings": ["OOF artifact lacks fold, target, or prediction columns."]}
    slices = []
    for fold in sorted(frame["fold"].unique().to_list()):
        part = frame.filter(pl.col("fold") == fold)
        slices.append({
            "slice_id": f"fold_{int(fold):03d}", "slice_type": "validation_fold", "fold": int(fold), "row_count": part.height,
            "target_mean": _mean(part["target"].to_list()), "prediction_mean": _mean(part["prediction"].to_list()),
            "reliability": "reliable" if part.height >= 20 else "caution_small_sample",
            "evidence_refs": ["baseline_evidence.artifacts.oof_predictions", "baseline_evidence.validation_policy"],
        })
    return {"status": "completed", "source": "fold_safe_baseline_oof", "oof_predictions_path": str(artifact), "slices": slices, "summary": {"slice_count": len(slices), "total_rows": frame.height}, "warnings": [], "limitations": ["This initial diagnostic reports validation-fold slices only; feature/category slices require additional OOF slice definitions."]}


def _skipped() -> dict[str, Any]:
    return {"status": "skipped", "reason": "missing_oof_predictions", "slices": [], "summary": {}, "warnings": [], "limitations": ["Fold-safe baseline OOF predictions are required for slice diagnostics."]}


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), 6) if numeric else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["run_slice_diagnostics"]
