from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from pydantic import BaseModel

from kaggle_researcher.contracts.evidence import (
    EvidencePathResolutionError,
    resolve_evidence_ref,
)
from kaggle_researcher.contracts.final_strategy import (
    EvidenceBinding,
    FinalStrategyAction,
)


@dataclass(frozen=True)
class EvidenceConsistencyIssue:
    code: str
    action_id: str | None
    message: str
    ref: str | None = None


class FinalStrategyEvidenceConsistencyError(ValueError):
    def __init__(self, issues: Iterable[EvidenceConsistencyIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(issue.message for issue in self.issues[:8])
        super().__init__(f"Final strategy evidence consistency failed: {detail}")


_HIGH_SEVERITIES = {"high", "critical", "severe"}
_SUCCESS_STATUSES = {"completed", "complete", "success", "succeeded", "successful"}
_VALIDATION_METHODS = (
    "stratified_group_kfold", "ranking_group_cv", "temporal_holdout", "temporal_cv",
    "stratified_kfold", "group_kfold", "kfold", "custom_required",
)


def validate_action_evidence_consistency(
    action: FinalStrategyAction,
    evidence_pack: Mapping[str, Any] | BaseModel,
    *,
    allowed_non_eda_refs: Iterable[str] = (),
) -> list[EvidenceConsistencyIssue]:
    """Validate action claims against concrete, resolved EDA values."""

    issues: list[EvidenceConsistencyIssue] = []
    exempt = set(allowed_non_eda_refs) | {"final_synthesizer.repaired"}
    resolved: dict[str, Any] = {}
    for ref in action.evidence_refs:
        if ref in exempt:
            continue
        try:
            resolved[ref] = resolve_evidence_ref(evidence_pack, ref)
        except EvidencePathResolutionError:
            issues.append(_issue(action, "broken_evidence_ref", f"Broken EDA evidence path: {ref}", ref))

    missing_subset = set(action.eda_result_refs) - set(action.evidence_refs)
    if missing_subset:
        issues.append(_issue(
            action,
            "eda_result_ref_not_evidence_ref",
            f"eda_result_refs are not present in evidence_refs: {sorted(missing_subset)}",
        ))

    text = _action_text(action)
    refs = set(action.evidence_refs)
    primary_id = _optional_value(evidence_pack, "inferred_schema.primary_id_column")
    if _is_primary_id_claim(text, primary_id):
        required = "inferred_schema.primary_id_column"
        _require_precise_ref(action, refs, required, issues, "primary_id_requires_precise_ref")
        if required in resolved and not str(resolved[required] or "").strip():
            issues.append(_issue(action, "primary_id_value_missing", "Primary-ID evidence resolves to an empty value.", required))

    if _is_primary_validation_claim(action, text):
        required = "validation_evidence.primary_validation"
        _require_precise_ref(action, refs, required, issues, "validation_requires_precise_ref")
        if required in resolved:
            actual_method = _validation_method(resolved[required])
            claimed_method = action.validation_strategy or _method_in_text(text)
            if claimed_method and actual_method and claimed_method != actual_method:
                issues.append(_issue(
                    action,
                    "validation_value_contradiction",
                    f"Action claims {claimed_method!r}, but primary_validation resolves to {actual_method!r}.",
                    required,
                ))

    if _is_high_drift_claim(text):
        precise = _first_existing_path(evidence_pack, (
            "drift_evidence.feature_drift_severity",
            "drift_evidence.overall_severity",
            "drift_evidence.severity",
        ))
        if precise is None:
            issues.append(_issue(action, "drift_severity_missing", "High-drift claim has no concrete severity field."))
        else:
            _require_precise_ref(action, refs, precise, issues, "drift_requires_precise_ref")
            if precise in resolved and str(resolved[precise]).strip().lower() not in _HIGH_SEVERITIES:
                issues.append(_issue(
                    action,
                    "drift_value_contradiction",
                    f"High-drift claim contradicts resolved severity {resolved[precise]!r}.",
                    precise,
                ))

    if _is_positive_threshold_claim(text):
        precise = _first_existing_path(evidence_pack, (
            "metric_evidence.requires_threshold",
            "metric_evidence.threshold_search_needed",
        ))
        if precise is None:
            issues.append(_issue(action, "threshold_requirement_missing", "Threshold-tuning claim has no concrete metric flag."))
        else:
            _require_precise_ref(action, refs, precise, issues, "threshold_requires_precise_ref")
            if precise in resolved and resolved[precise] is not True:
                issues.append(_issue(
                    action,
                    "threshold_value_contradiction",
                    f"Threshold tuning is recommended although {precise} resolves to {resolved[precise]!r}.",
                    precise,
                ))

    if _is_baseline_reproduction_claim(text):
        required = "baseline_evidence.status"
        _require_precise_ref(action, refs, required, issues, "baseline_requires_precise_ref")
        if required in resolved and str(resolved[required]).strip().lower() not in _SUCCESS_STATUSES:
            issues.append(_issue(
                action,
                "baseline_value_contradiction",
                f"Baseline reproduction requires a successful baseline, not {resolved[required]!r}.",
                required,
            ))

    if _is_feature_priority_claim(text):
        matching_ref = _matching_feature_family_ref(evidence_pack, text)
        if matching_ref and matching_ref not in refs:
            issues.append(_issue(
                action,
                "feature_priority_requires_item_ref",
                f"Feature-priority claim must cite concrete item {matching_ref!r}.",
                matching_ref,
            ))
    return issues


def require_action_evidence_consistency(
    action: FinalStrategyAction,
    evidence_pack: Mapping[str, Any] | BaseModel,
    *,
    allowed_non_eda_refs: Iterable[str] = (),
) -> None:
    issues = validate_action_evidence_consistency(
        action, evidence_pack, allowed_non_eda_refs=allowed_non_eda_refs
    )
    if issues:
        raise FinalStrategyEvidenceConsistencyError(issues)


def bounded_evidence_preview(value: Any, *, max_chars: int = 512) -> Any:
    """Return a JSON-safe scalar or bounded structural summary."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > max_chars:
            return value[: max_chars - 1] + "…"
        return value
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)[:max_chars]
    if len(encoded) <= max_chars:
        return json.loads(encoded)
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "field_count": len(value),
            "fields": sorted(str(key) for key in value)[:20],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "list",
            "item_count": len(value),
            "sample": [bounded_evidence_preview(item, max_chars=96) for item in value[:3]],
        }
    return str(value)[:max_chars]


def build_action_evidence_bindings(
    action: FinalStrategyAction,
    evidence_pack: Mapping[str, Any] | BaseModel,
    *,
    allowed_non_eda_refs: Iterable[str] = (),
) -> list[EvidenceBinding]:
    exempt = set(allowed_non_eda_refs) | {"final_synthesizer.repaired"}
    primary_refs = _required_precise_refs(action, evidence_pack)
    bindings: list[EvidenceBinding] = []
    for ref in action.evidence_refs:
        if ref in exempt:
            continue
        try:
            value = resolve_evidence_ref(evidence_pack, ref)
        except EvidencePathResolutionError:
            continue
        bindings.append(EvidenceBinding(
            ref=ref,
            resolved_value_preview=bounded_evidence_preview(value),
            role="primary" if ref in primary_refs else "supporting",
        ))
    return bindings


def _required_precise_refs(
    action: FinalStrategyAction,
    evidence_pack: Mapping[str, Any] | BaseModel,
) -> set[str]:
    text = _action_text(action)
    refs: set[str] = set()
    primary_id = _optional_value(evidence_pack, "inferred_schema.primary_id_column")
    if _is_primary_id_claim(text, primary_id):
        refs.add("inferred_schema.primary_id_column")
    if _is_primary_validation_claim(action, text):
        refs.add("validation_evidence.primary_validation")
    if _is_high_drift_claim(text):
        path = _first_existing_path(evidence_pack, (
            "drift_evidence.feature_drift_severity", "drift_evidence.overall_severity",
            "drift_evidence.severity",
        ))
        if path:
            refs.add(path)
    if _is_positive_threshold_claim(text):
        path = _first_existing_path(evidence_pack, (
            "metric_evidence.requires_threshold", "metric_evidence.threshold_search_needed",
        ))
        if path:
            refs.add(path)
    if _is_baseline_reproduction_claim(text):
        refs.add("baseline_evidence.status")
    matching = _matching_feature_family_ref(evidence_pack, text)
    if matching:
        refs.add(matching)
    return refs


def _require_precise_ref(
    action: FinalStrategyAction,
    refs: set[str],
    required: str,
    issues: list[EvidenceConsistencyIssue],
    code: str,
) -> None:
    if required not in refs:
        issues.append(_issue(action, code, f"Factual claim requires precise evidence ref {required!r}.", required))


def _issue(action: FinalStrategyAction, code: str, message: str, ref: str | None = None) -> EvidenceConsistencyIssue:
    return EvidenceConsistencyIssue(code, action.action_id, message, ref)


def _action_text(action: FinalStrategyAction) -> str:
    return f"{action.action} {action.reason}".lower().replace("_", " ")


def _is_primary_id_claim(text: str, primary_id: Any) -> bool:
    subject = str(primary_id or "").lower()
    identifies = "primary id" in text or "primary identifier" in text or (subject and subject in text)
    return bool(identifies)


def _is_primary_validation_claim(action: FinalStrategyAction, text: str) -> bool:
    return bool(action.validation_strategy) or any(
        token in text for token in ("primary validation", "validation policy selected", "eda-selected validation")
    ) or ("validation" in text and _method_in_text(text) is not None)


def _is_high_drift_claim(text: str) -> bool:
    return "drift" in text and any(token in text for token in ("high drift", "critical drift", "drift severity is high", "high severity"))


def _is_positive_threshold_claim(text: str) -> bool:
    if "threshold" not in text or not any(token in text for token in ("tune", "tuning", "search")):
        return False
    return not any(token in text for token in ("do not tune", "not tune", "unless", "if required"))


def _is_baseline_reproduction_claim(text: str) -> bool:
    return "baseline" in text and any(token in text for token in ("reproduce", "re-run", "rerun completed"))


def _is_feature_priority_claim(text: str) -> bool:
    return any(token in text for token in ("prioritize", "priority feature", "prefer feature"))


def _method_in_text(text: str) -> str | None:
    normalized = text.replace(" ", "_").replace("-", "_")
    return next((method for method in _VALIDATION_METHODS if method in normalized), None)


def _validation_method(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("method") or value.get("name")
    method = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return method or None


def _first_existing_path(evidence_pack: Mapping[str, Any] | BaseModel, paths: Iterable[str]) -> str | None:
    for path in paths:
        try:
            resolve_evidence_ref(evidence_pack, path)
        except EvidencePathResolutionError:
            continue
        return path
    return None


def _optional_value(evidence_pack: Mapping[str, Any] | BaseModel, path: str) -> Any:
    try:
        return resolve_evidence_ref(evidence_pack, path)
    except EvidencePathResolutionError:
        return None


def _matching_feature_family_ref(evidence_pack: Mapping[str, Any] | BaseModel, text: str) -> str | None:
    payload = evidence_pack.model_dump(mode="json") if isinstance(evidence_pack, BaseModel) else evidence_pack
    probes = payload.get("feature_probe_evidence") if isinstance(payload, Mapping) else None
    if not isinstance(probes, list):
        return None
    for item in probes:
        if not isinstance(item, Mapping):
            continue
        family = str(item.get("feature_family") or "").strip()
        normalized = re.sub(r"[^a-z0-9]+", " ", family.lower()).strip()
        if normalized and normalized in re.sub(r"[^a-z0-9]+", " ", text.lower()):
            return f"feature_probe_evidence.{family}"
    return None


__all__ = [
    "EvidenceConsistencyIssue", "FinalStrategyEvidenceConsistencyError",
    "bounded_evidence_preview", "build_action_evidence_bindings",
    "require_action_evidence_consistency", "validate_action_evidence_consistency",
]
