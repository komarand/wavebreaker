from __future__ import annotations

from typing import Any

from kaggle_researcher.eda.schemas import HypothesisResult, RecommendedNextAction


ACTION_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def build_recommended_next_actions(
    evidence_pack_partial: dict,
    hypothesis_results: list[HypothesisResult],
) -> list[RecommendedNextAction]:
    """Build evidence-backed next actions from confirmed/partial findings."""

    if not _has_actionable_hypotheses(hypothesis_results):
        return []

    actions: list[RecommendedNextAction] = []
    validation = _as_dict(evidence_pack_partial.get("validation_evidence"))
    metric = _as_dict(evidence_pack_partial.get("metric_evidence"))
    leakage = [_as_dict(item) for item in evidence_pack_partial.get("leakage_evidence", [])]

    actions.extend(_validation_actions(validation))
    actions.extend(_metric_actions(metric))
    actions.extend(_leakage_actions(leakage))
    relationship_action = _relationship_action(evidence_pack_partial)
    if relationship_action is not None:
        actions.append(relationship_action)

    return _sort_and_dedupe(actions)


def _validation_actions(validation: dict[str, Any]) -> list[RecommendedNextAction]:
    primary = _as_dict(validation.get("primary_validation"))
    method = _normalize_method(primary.get("method"))
    if method in {"stratified_kfold", "stratifiedkfold"}:
        return [
            _action(
                "P0",
                "Use StratifiedKFold for model validation.",
                "Validation evidence selected stratified CV for iid classification.",
                ["validation_evidence.primary_validation"],
            )
        ]
    if method in {"kfold"}:
        return [
            _action(
                "P0",
                "Use KFold for model validation.",
                "Validation evidence selected ordinary KFold for iid regression.",
                ["validation_evidence.primary_validation"],
            )
        ]
    if method in {"group_kfold", "groupkfold", "stratified_group_kfold", "stratifiedgroupkfold"}:
        group_column = primary.get("group_column")
        suffix = f" on {group_column}" if group_column else ""
        return [
            _action(
                "P0",
                f"Respect grouped validation splits{suffix}.",
                "Validation evidence selected a group-aware policy to avoid entity leakage.",
                ["validation_evidence.primary_validation"],
            )
        ]
    if method in {"ranking_group_cv"}:
        return [
            _action(
                "P0",
                "Use query/group-aware validation for ranking metrics.",
                "Validation evidence selected ranking_group_cv.",
                ["validation_evidence.primary_validation"],
            )
        ]
    if method in {"temporal_holdout", "expanding_window"}:
        return [
            _action(
                "P0",
                "Use temporal validation for model selection.",
                "Validation evidence selected a temporal primary policy.",
                ["validation_evidence.primary_validation"],
            )
        ]
    return []


def _metric_actions(metric: dict[str, Any]) -> list[RecommendedNextAction]:
    actions: list[RecommendedNextAction] = []
    if metric.get("requires_probabilities"):
        actions.append(
            _action(
                "P0",
                "Output probabilities or ranking scores, not hard labels.",
                "Metric evidence requires probability/rank-style predictions.",
                ["metric_evidence.requires_probabilities"],
            )
        )
    if metric.get("requires_threshold"):
        actions.append(
            _action(
                "P0",
                "Tune classification thresholds on validation data only.",
                "Metric evidence requires thresholded predictions.",
                ["metric_evidence.requires_threshold"],
            )
        )
    if metric.get("requires_calibration"):
        actions.append(
            _action(
                "P1",
                "Check probability calibration and clipping.",
                "Metric evidence requires calibrated probability estimates.",
                ["metric_evidence.requires_calibration"],
            )
        )
    if metric.get("metric_family") == "regression_error":
        actions.append(
            _action(
                "P0",
                "Optimize the regression loss and inspect target transforms.",
                "Metric evidence belongs to the regression_error family.",
                ["metric_evidence.metric_family"],
            )
        )
    return actions


def _leakage_actions(leakage: list[dict[str, Any]]) -> list[RecommendedNextAction]:
    risky = [
        item
        for item in leakage
        if item.get("status") in {"failed", "warning"}
    ]
    if not risky:
        return []
    check_ids = ", ".join(str(item.get("check_id")) for item in risky)
    return [
        _action(
            "P0",
            "Fix or exclude leakage-risk sources before modeling.",
            f"Leakage evidence contains failed/warning checks: {check_ids}.",
            ["leakage_evidence"],
        )
    ]


def _relationship_action(evidence_pack: dict[str, Any]) -> RecommendedNextAction | None:
    table_profiles = evidence_pack.get("table_profiles", [])
    has_secondary_tables = any(
        _as_dict(profile).get("path", "").lower().startswith(("train_", "test_"))
        and "base" not in _as_dict(profile).get("path", "").lower()
        for profile in table_profiles
    )
    relationship_evidence = evidence_pack.get("relationship_evidence")
    if has_secondary_tables and not relationship_evidence:
        return _action(
            "P1",
            "Run relationship inference before aggregating secondary tables.",
            "Table profile evidence includes secondary tables, "
            "but relationship evidence is absent.",
            ["table_profiles"],
        )
    return None


def _has_actionable_hypotheses(hypothesis_results: list[HypothesisResult]) -> bool:
    return any(
        result.status in {"confirmed", "partially_confirmed"}
        for result in hypothesis_results
    )


def _action(
    priority: str,
    action: str,
    why: str,
    evidence_refs: list[str],
) -> RecommendedNextAction:
    return RecommendedNextAction(
        priority=priority,
        action=action,
        why=why,
        evidence_refs=evidence_refs,
    )


def _sort_and_dedupe(actions: list[RecommendedNextAction]) -> list[RecommendedNextAction]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[RecommendedNextAction] = []
    for action in actions:
        if not action.evidence_refs:
            continue
        key = (action.action, tuple(action.evidence_refs))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return sorted(
        deduped,
        key=lambda item: (ACTION_PRIORITY_ORDER[item.priority], item.action),
    )


def _normalize_method(method: Any) -> str:
    return str(method or "").strip().lower().replace("-", "_").replace(" ", "_")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {}


__all__ = ["build_recommended_next_actions"]
