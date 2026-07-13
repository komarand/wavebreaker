from __future__ import annotations

import re
from typing import Any


MODULE_CLASSIFICATION = {
    "file_inventory": "core_eda",
    "schema_inferer": "core_eda",
    "table_profiler": "core_eda",
    "metric_analyzer": "core_eda",
    "validation_analyzer": "core_eda",
    "target_diagnostics": "core_eda",
    "leakage_checker": "core_eda",
    "relationship_inferer": "core_eda",
    "drift_analyzer": "core_eda",
    "feature_diagnostics": "core_eda",
    "baseline_runner": "model_assisted_diagnostic",
    "baseline_ablations": "model_assisted_diagnostic",
    "interaction_diagnostics": "model_assisted_diagnostic",
    "slice_diagnostics": "model_assisted_diagnostic",
    "visual_diagnostics": "evidence_rendering",
    "source_claim_validation": "post_eda_reasoning",
}

DEPRECATED_OUTPUTS = {
    "strategy_hints": {
        "replacement": "eda_implications",
        "reason": "EDA emits evidence implications rather than final strategy.",
    },
    "eda_strategy_hints": {
        "replacement": "eda_implications",
        "reason": "EDA emits evidence implications rather than final strategy.",
    },
    "recommended_next_actions": {
        "replacement": "testable_hypotheses",
        "reason": "The downstream reasoning layer owns the ordered action backlog.",
    },
    "experiment_candidates": {
        "replacement": "testable_hypotheses",
        "reason": "EDA emits unresolved hypotheses; the reasoning layer owns experiment planning.",
    },
    "eda_risk_register": {
        "replacement": "eda_risks",
        "reason": "The canonical register contains EDA-local risks only.",
    },
}

_SCOPE_MAP = {
    "feature_engineering": "feature", "drift_and_leaderboard": "drift",
    "baseline_ablations": "baseline", "first_experiments": "feature",
    "risk_register": "data_quality", "do_not_do": "leakage",
}


def build_eda_implications(hints: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Project legacy hints into conservative, stable diagnostic implications."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for category in sorted(hints):
        if category == "source_claim_validation":
            continue
        for item in hints.get(category, []):
            rows.append((category, dict(item)))
    result = []
    for index, (category, item) in enumerate(rows, 1):
        action = str(item.get("action") or item.get("hint") or "Follow-up evidence review is warranted.")
        implication = _diagnostic_wording(action)
        refs = [str(ref) for ref in item.get("evidence_refs", [])]
        result.append({
            "implication_id": f"eda_implication_{index:03d}",
            "finding": str(item.get("why") or item.get("finding") or "EDA produced diagnostic evidence."),
            "implication": implication,
            "scope": _SCOPE_MAP.get(category, category if category in {"validation", "leakage", "drift", "target", "feature", "missingness", "interaction", "slice", "baseline", "relationship", "data_quality"} else "data_quality"),
            "confidence": str(item.get("confidence") or "medium"),
            "priority_signal": {"P0": "urgent", "P1": "important", "P2": "optional", "P3": "informational"}.get(str(item.get("priority")), "informational"),
            "evidence_refs": refs,
            "limitations": list(item.get("limitations") or []),
            "evidence_origin": "dataset_measurement",
        })
    return result


def stage_status(module_statuses: dict[str, str]) -> dict[str, str]:
    groups = {
        "dataset_resolution": ["file_inventory"],
        "core_eda": [name for name, cls in MODULE_CLASSIFICATION.items() if cls == "core_eda"],
        "model_assisted_diagnostics": [name for name, cls in MODULE_CLASSIFICATION.items() if cls == "model_assisted_diagnostic"],
        "visual_artifacts": ["visual_diagnostics"],
        "post_eda_reasoning": ["source_claim_validation"],
    }
    return {name: _aggregate([module_statuses.get(module) for module in modules]) for name, modules in groups.items()}


def qualify_validation_evidence(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    primary = dict(result.get("primary_validation") or result.get("recommended_validation") or {})
    if primary:
        result["recommended_validation_candidate"] = {
            "method": primary.get("method"),
            "status": "evidence_supported_candidate",
            "confidence": result.get("confidence") or primary.get("confidence") or "medium",
            "supporting_evidence_refs": result.get("evidence_refs") or ["validation_evidence"],
            "alternatives": result.get("diagnostic_validations") or [],
            "conditions_that_would_change_it": result.get("limitations") or result.get("warnings") or [],
            "evidence_origin": "statistical_diagnostic",
        }
    return result


def _aggregate(values: list[str | None]) -> str:
    present = [value for value in values if value]
    if not present or all(value == "skipped" for value in present): return "skipped"
    executed = [value for value in present if value != "skipped"]
    if executed and all(value == "completed" for value in executed): return "completed"
    if any(value == "completed" for value in present): return "partial"
    return "failed" if any(value == "failed" for value in present) else "partial"


def _diagnostic_wording(text: str) -> str:
    replacements = [
        (r"(?i)^use ", "Test whether using "), (r"(?i)^prioritize ", "Compare "),
        (r"(?i)^start with ", "Test "), (r"(?i)^add ", "Test whether adding "),
        (r"(?i)^keep ", "Evaluate whether to keep "), (r"(?i)^tune ", "Evaluate tuning "),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    if not text.endswith("."): text += "."
    return text


__all__ = ["DEPRECATED_OUTPUTS", "MODULE_CLASSIFICATION", "build_eda_implications", "qualify_validation_evidence", "stage_status"]
