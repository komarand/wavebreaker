from __future__ import annotations

import re
from typing import Any

from kaggle_researcher.eda.schemas import HypothesisResult, RecommendedNextAction


ACTION_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
CATEGORY_ORDER = {
    "leakage": 0,
    "do_not_do": 1,
    "primary_id_safety": 1,
    "submission_safety": 1,
    "role_exclusion": 1,
    "validation": 2,
    "threshold_tuning": 3,
    "metric": 3,
    "target": 3,
    "baseline": 4,
    "baseline_ablations": 4,
    "feature_engineering": 5,
    "feature_probe": 5,
    "numeric_features": 5,
    "categorical_features": 5,
    "missingness": 5,
    "text_features": 5,
    "drift_and_leaderboard": 6,
    "drift": 6,
    "first_experiments": 7,
}
SINGLETON_INTENTS = {
    "validation_policy",
    "threshold_tuning",
    "baseline_completed_sanity_floor",
    "build_baseline_experiment",
    "feature_block_start",
    "missingness_indicators",
}
STRATEGY_SOURCE_RANK = 0
LEGACY_SOURCE_RANK = 1


def build_recommended_next_actions(
    evidence_pack_partial: dict[str, Any] | None = None,
    hypothesis_results: list[HypothesisResult] | None = None,
    *,
    eda_strategy_hints: dict[str, Any] | None = None,
    legacy_actions: list[dict[str, Any] | RecommendedNextAction] | None = None,
    metric_evidence: dict[str, Any] | None = None,
    validation_evidence: dict[str, Any] | None = None,
    leakage_evidence: list[dict[str, Any]] | None = None,
    drift_evidence: dict[str, Any] | None = None,
    feature_probe_evidence: list[dict[str, Any]] | None = None,
    feature_diagnostics: dict[str, Any] | None = None,
    baseline_evidence: dict[str, Any] | None = None,
) -> list[RecommendedNextAction]:
    """Build a deterministic, evidence-backed, deduplicated action list.

    The positional signature is kept for the orchestrator and existing tests.
    Keyword evidence arguments allow the aggregation layer to be exercised
    directly without building a full EDA evidence pack.
    """

    if evidence_pack_partial is None:
        evidence_pack_partial = {
            "eda_strategy_hints": eda_strategy_hints or {},
            "metric_evidence": metric_evidence or {},
            "validation_evidence": validation_evidence or {},
            "leakage_evidence": leakage_evidence or [],
            "drift_evidence": drift_evidence or {},
            "feature_probe_evidence": feature_probe_evidence or [],
            "feature_diagnostics": feature_diagnostics or {},
            "baseline_evidence": baseline_evidence or {},
        }
    elif eda_strategy_hints is not None:
        evidence_pack_partial = {**evidence_pack_partial, "eda_strategy_hints": eda_strategy_hints}

    if hypothesis_results is not None and not _has_actionable_hypotheses(hypothesis_results):
        return []

    generated_legacy_actions = _legacy_actions_from_evidence(evidence_pack_partial)
    if legacy_actions:
        generated_legacy_actions.extend(
            _action_from_payload(payload, source_rank=LEGACY_SOURCE_RANK)
            for payload in legacy_actions
        )

    actions = [
        *_strategy_hint_actions(evidence_pack_partial),
        *generated_legacy_actions,
    ]
    return _aggregate_actions(actions)


def dedupe_recommended_next_actions(
    actions: list[dict[str, Any] | RecommendedNextAction],
) -> list[RecommendedNextAction]:
    """Deduplicate already-built action payloads without generating new ones."""

    return _aggregate_actions(
        [_action_from_payload(action, source_rank=LEGACY_SOURCE_RANK) for action in actions]
    )


def normalize_action_key(action: str, evidence_refs: list[str] | None = None) -> str:
    """Return a conservative normalized action intent key."""

    text = _normalize_action_text(action)
    refs = [str(ref) for ref in evidence_refs or []]
    evidence = " ".join(refs).lower()

    if "metric_evidence.requires_threshold" in evidence or (
        "threshold" in text and _has_any(text, ("validation", "fold", "cv", "oof"))
    ):
        return "threshold_tuning"
    if "stratified kfold" in text:
        return "validation_policy:stratified_kfold"
    if "kfold" in text and _has_any(text, ("validation", "cv", "comparison", "split")):
        return "validation_policy:kfold"
    if (
        "validation_evidence.primary_validation" in evidence
        and _has_any(text, ("validation", "split", "fold", "cv", "comparison"))
    ):
        return "validation_policy"
    if "missingness indicator" in text or "missing indicators" in text:
        return "missingness_indicators"
    if "baseline_evidence" in evidence or (
        "baseline" in text
        and _has_any(text, ("sanity", "floor", "check"))
        and not text.startswith("build")
    ):
        return "baseline_completed_sanity_floor"
    if (
        text.startswith("build a baseline")
        or text.startswith("start with baseline")
        or text.startswith("run a simple fold safe baseline")
        or text.startswith("run simple fold safe baseline")
        or (
        "baseline" in text and _has_any(text, ("safe numeric", "safe features", "first feature", "feature block"))
        )
    ):
        return "build_baseline_experiment"
    if _has_any(text, ("safe numeric base features", "first feature block", "base feature block")):
        return "feature_block_start"
    if "primary id" in text or "id column" in text or "row id" in text:
        return "primary_id_safety"
    if "sample submission" in text or "submission output" in text:
        return "submission_safety"
    if _has_any(text, ("target encoding", "mean encoding", "weight of evidence", "woe")):
        prefix = "fold_fitted_" if _has_any(text, ("fold", "oof", "out of fold")) else ""
        return f"{prefix}target_encoding"
    if _has_any(text, ("public lb", "public leaderboard", "leaderboard")):
        return "leaderboard_safety"
    return _remove_intent_fillers(text)


def _legacy_actions_from_evidence(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    validation = _as_dict(evidence_pack.get("validation_evidence"))
    metric = _as_dict(evidence_pack.get("metric_evidence"))
    leakage = [_as_dict(item) for item in evidence_pack.get("leakage_evidence", [])]

    actions: list[RecommendedNextAction] = []
    actions.extend(_validation_actions(validation))
    actions.extend(_metric_actions(metric))
    actions.extend(_leakage_actions(leakage))
    relationship_action = _relationship_action(evidence_pack)
    if relationship_action is not None:
        actions.append(relationship_action)
    actions.extend(_relationship_evidence_actions(evidence_pack))
    actions.extend(_drift_actions(evidence_pack))
    actions.extend(_baseline_actions(evidence_pack))
    actions.extend(_feature_probe_actions(evidence_pack))
    actions.extend(_notebook_actions(evidence_pack))
    return actions


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
                risk="low",
                applies_to=["validation", "model_selection"],
                source_categories=["legacy", "validation"],
            )
        ]
    if method in {"kfold"}:
        return [
            _action(
                "P0",
                "Use KFold for model validation.",
                "Validation evidence selected ordinary KFold for iid regression.",
                ["validation_evidence.primary_validation"],
                risk="low",
                applies_to=["validation", "model_selection"],
                source_categories=["legacy", "validation"],
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
                risk="high",
                applies_to=["validation"],
                source_categories=["legacy", "validation"],
            )
        ]
    if method in {"ranking_group_cv"}:
        return [
            _action(
                "P0",
                "Use query/group-aware validation for ranking metrics.",
                "Validation evidence selected ranking_group_cv.",
                ["validation_evidence.primary_validation"],
                risk="high",
                applies_to=["validation", "metric"],
                source_categories=["legacy", "validation"],
            )
        ]
    if method in {"temporal_holdout", "expanding_window"}:
        return [
            _action(
                "P0",
                "Use temporal validation for model selection.",
                "Validation evidence selected a temporal primary policy.",
                ["validation_evidence.primary_validation"],
                risk="medium",
                applies_to=["validation"],
                source_categories=["legacy", "validation"],
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
                risk="medium",
                applies_to=["metric"],
                source_categories=["legacy", "metric"],
            )
        )
    if metric.get("requires_threshold"):
        actions.append(
            _action(
                "P0",
                "Tune classification thresholds on validation data only.",
                "Metric evidence requires thresholded predictions.",
                ["metric_evidence.requires_threshold"],
                risk="medium",
                applies_to=["validation", "metric"],
                source_categories=["legacy", "metric"],
            )
        )
    if metric.get("requires_calibration"):
        actions.append(
            _action(
                "P1",
                "Check probability calibration and clipping.",
                "Metric evidence requires calibrated probability estimates.",
                ["metric_evidence.requires_calibration"],
                risk="medium",
                applies_to=["metric"],
                source_categories=["legacy", "metric"],
            )
        )
    if metric.get("metric_family") == "regression_error":
        actions.append(
            _action(
                "P0",
                "Optimize the regression loss and inspect target transforms.",
                "Metric evidence belongs to the regression_error family.",
                ["metric_evidence.metric_family"],
                risk="low",
                applies_to=["metric", "model_selection"],
                source_categories=["legacy", "metric"],
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
            risk="high",
            applies_to=["leakage", "features"],
            source_categories=["legacy", "leakage"],
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
            risk="medium",
            applies_to=["feature_engineering"],
            source_categories=["legacy", "feature_engineering"],
        )
    return None


def _relationship_evidence_actions(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    relationship_evidence = _as_dict(evidence_pack.get("relationship_evidence"))
    relationships = [_as_dict(item) for item in relationship_evidence.get("relationships", [])]
    risky = [
        item
        for item in relationships
        if item.get("relationship_type") in {"one_to_many", "many_to_many"}
        or item.get("requires_aggregation")
    ]
    if not risky:
        return []
    tables = ", ".join(str(item.get("table")) for item in risky if item.get("table"))
    return [
        _action(
            "P1",
            "Aggregate one-to-many secondary tables before joining.",
            f"Relationship evidence found row-multiplying relationships: {tables}.",
            ["relationship_evidence.relationships"],
            risk="medium",
            applies_to=["feature_engineering"],
            source_categories=["legacy", "feature_engineering"],
        )
    ]


def _drift_actions(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    drift = _as_dict(evidence_pack.get("drift_evidence"))
    if drift.get("severity") not in {"medium", "high", "critical"}:
        return []
    return [
        _action(
            "P1",
            "Treat train/test drift as leaderboard-risk diagnostics.",
            "Drift evidence shows medium/high severity; keep validation robust and avoid public LB overfitting.",
            ["drift_evidence"],
            risk="medium" if drift.get("severity") == "medium" else "high",
            applies_to=["drift", "leaderboard"],
            source_categories=["legacy", "drift_and_leaderboard"],
        )
    ]


def _baseline_actions(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    baseline = _as_dict(evidence_pack.get("baseline_evidence"))
    if baseline.get("status") != "completed":
        return []
    evidence_refs = ["baseline_evidence"]
    if baseline.get("metric_value") is not None and baseline.get("preprocessing_policy"):
        evidence_refs = ["baseline_evidence.metric_value", "baseline_evidence.preprocessing_policy"]
    return [
        _action(
            "P1",
            "Compare future experiments against the completed EDA baseline.",
            "Baseline runner completed with documented fold-safe preprocessing.",
            evidence_refs,
            risk="low",
            applies_to=["baseline", "model_selection"],
            source_categories=["legacy", "baseline"],
        )
    ]


def _feature_probe_actions(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    probes = [_as_dict(item) for item in evidence_pack.get("feature_probe_evidence", [])]
    actions: list[RecommendedNextAction] = []
    high_potential = [
        item for item in probes if item.get("status") == "high_potential"
    ]
    if high_potential:
        families = ", ".join(str(item.get("feature_family")) for item in high_potential)
        actions.append(
            _action(
                "P1",
                f"Prioritize high-potential feature families: {families}.",
                "Feature probe evidence marked these families as high potential.",
                ["feature_probe_evidence"],
                risk="low",
                applies_to=["feature_engineering"],
                source_categories=["legacy", "feature_engineering"],
            )
        )
    unsafe = [item for item in probes if item.get("status") == "unsafe"]
    if unsafe:
        families = ", ".join(str(item.get("feature_family")) for item in unsafe)
        actions.append(
            _action(
                "P1",
                f"Audit or avoid risky feature families: {families}.",
                "Feature probe evidence marked these families as unsafe or leakage-prone.",
                ["feature_probe_evidence"],
                risk="high",
                applies_to=["feature_engineering", "leakage"],
                source_categories=["legacy", "feature_engineering"],
            )
        )
    regression_transform = next(
        (
            item
            for item in probes
            if item.get("feature_family") == "regression_target_transform"
            and item.get("status") in {"medium_potential", "high_potential"}
        ),
        None,
    )
    if regression_transform is not None:
        actions.append(
            _action(
                "P1",
                "Evaluate a regression target transform inside validation.",
                "Feature probe evidence reports skewed regression target behavior.",
                ["feature_probe_evidence"],
                risk="medium",
                applies_to=["feature_engineering", "validation"],
                source_categories=["legacy", "feature_engineering"],
            )
        )
    return actions


def _notebook_actions(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    notebook = _as_dict(evidence_pack.get("notebook_static_analysis"))
    risky = notebook.get("suspicious_leaderboard_overfit_patterns", []) if notebook else []
    feature_patterns = notebook.get("feature_families", []) if notebook else []
    if not risky and not feature_patterns:
        return []
    actions: list[RecommendedNextAction] = []
    if risky:
        actions.append(
            _action(
                "P1",
                "Audit risky public-notebook patterns before copying them.",
                "Static notebook analysis observed leaderboard-overfit or risky notebook patterns.",
                ["notebook_static_analysis.suspicious_leaderboard_overfit_patterns"],
                risk="high",
                applies_to=["leaderboard", "feature_engineering"],
                source_categories=["legacy", "drift_and_leaderboard"],
            )
        )
    if feature_patterns:
        actions.append(
            _action(
                "P2",
                "Use notebook feature patterns only as static inspiration.",
                "Notebook patterns are observations, not proof; validate any copied idea locally.",
                ["notebook_static_analysis.feature_families"],
                risk="medium",
                applies_to=["feature_engineering"],
                source_categories=["legacy", "feature_engineering"],
            )
        )
    return actions


def _strategy_hint_actions(evidence_pack: dict[str, Any]) -> list[RecommendedNextAction]:
    hints = _as_dict(evidence_pack.get("eda_strategy_hints"))
    actions: list[RecommendedNextAction] = []
    for category, items in hints.items():
        if not isinstance(items, list):
            continue
        for item in items:
            payload = _as_dict(item)
            refs = [str(ref) for ref in payload.get("evidence_refs") or [] if str(ref).strip()]
            if not refs:
                continue
            source_categories = _merge_ordered_values(
                [str(category)],
                [str(value) for value in payload.get("source_categories") or []],
            )
            actions.append(
                _action(
                    _normalize_priority(payload.get("priority")),
                    str(payload.get("action") or "Follow EDA strategy hint."),
                    str(payload.get("why") or "Generated from EDA strategy hints."),
                    refs,
                    risk=_normalize_risk(payload.get("risk")),
                    applies_to=[str(value) for value in payload.get("applies_to") or []],
                    source_categories=source_categories,
                    source_rank=STRATEGY_SOURCE_RANK,
                )
            )
    return actions


def _aggregate_actions(actions: list[RecommendedNextAction]) -> list[RecommendedNextAction]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for action in actions:
        if not _is_valid_action(action):
            continue
        intent = normalize_action_key(action.action, action.evidence_refs)
        if not intent:
            continue
        category = _primary_evidence_category(action.evidence_refs)
        key_category = intent if _is_singleton_intent(intent) else category
        key = (intent, key_category)
        if key not in groups:
            groups[key] = _group_from_action(action, category)
            continue
        _merge_group(groups[key], action)

    merged = [
        action
        for action in (_action_from_group(group) for group in groups.values())
        if _is_valid_action(action)
    ]
    return sorted(
        merged,
        key=lambda item: (
            ACTION_PRIORITY_ORDER[item.priority],
            RISK_ORDER.get(str(item.risk or ""), 3),
            _category_sort_value(item),
            _normalize_action_text(item.action),
        ),
    )


def _group_from_action(action: RecommendedNextAction, category: str) -> dict[str, Any]:
    return {
        "priority": action.priority,
        "action": action.action,
        "intent": normalize_action_key(action.action, action.evidence_refs),
        "why": action.why,
        "evidence_refs": set(action.evidence_refs),
        "risk": action.risk,
        "applies_to": set(action.applies_to),
        "source_categories": set(action.source_categories),
        "source_rank": _source_rank(action),
        "category": category,
        "why_values": {action.why} if action.why else set(),
    }


def _merge_group(group: dict[str, Any], action: RecommendedNextAction) -> None:
    group["priority"] = _highest_priority(group["priority"], action.priority)
    group["risk"] = _highest_risk(group.get("risk"), action.risk)
    group["evidence_refs"].update(action.evidence_refs)
    group["applies_to"].update(action.applies_to)
    group["source_categories"].update(action.source_categories)
    if action.why:
        group["why_values"].add(action.why)
    if _action_text_rank(action) < _action_text_rank_from_group(group):
        group["action"] = action.action
        group["why"] = action.why
        group["source_rank"] = _source_rank(action)


def _action_from_group(group: dict[str, Any]) -> RecommendedNextAction:
    why_values = sorted(
        [why for why in group["why_values"] if why and why != group["why"]],
        key=_normalize_action_text,
    )
    why = " ".join([str(group["why"]), *why_values[:1]]).strip()
    evidence_refs = _sorted_evidence_refs(group["evidence_refs"])
    if group.get("intent") == "baseline_completed_sanity_floor" and "baseline_evidence" in evidence_refs:
        evidence_refs = [ref for ref in evidence_refs if ref.startswith("baseline_evidence")]
    return _action(
        group["priority"],
        group["action"],
        why or "Merged from evidence-backed EDA recommendations.",
        evidence_refs,
        risk=group.get("risk"),
        applies_to=_sorted_categories(group["applies_to"]),
        source_categories=_sorted_categories(group["source_categories"]),
        source_rank=group["source_rank"],
    )


def _action_from_payload(
    payload: dict[str, Any] | RecommendedNextAction,
    *,
    source_rank: int,
) -> RecommendedNextAction:
    if isinstance(payload, RecommendedNextAction):
        data = payload.model_dump(mode="json")
    else:
        data = dict(payload)
    action_text = _clean_display_text(data.get("action"))
    why = _clean_display_text(data.get("why") or "Evidence-backed EDA recommendation.")
    return _action(
        _normalize_priority(data.get("priority")),
        action_text,
        why,
        [str(ref) for ref in data.get("evidence_refs") or [] if str(ref).strip()],
        risk=_normalize_risk(data.get("risk")),
        applies_to=[str(value) for value in data.get("applies_to") or []],
        source_categories=[str(value) for value in data.get("source_categories") or []],
        source_rank=source_rank,
    )


def _action(
    priority: str,
    action: str,
    why: str,
    evidence_refs: list[str],
    *,
    risk: str | None = None,
    applies_to: list[str] | None = None,
    source_categories: list[str] | None = None,
    source_rank: int = LEGACY_SOURCE_RANK,
) -> RecommendedNextAction:
    result = RecommendedNextAction(
        priority=_normalize_priority(priority),
        action=_clean_display_text(action),
        why=_clean_display_text(why),
        evidence_refs=evidence_refs,
        risk=_normalize_risk(risk),
        applies_to=applies_to or [],
        source_categories=source_categories or [],
    )
    result._source_rank = source_rank  # type: ignore[attr-defined]
    return result


def _has_actionable_hypotheses(hypothesis_results: list[HypothesisResult]) -> bool:
    return any(
        result.status in {"confirmed", "partially_confirmed"}
        for result in hypothesis_results
    )


def _is_singleton_intent(intent: str) -> bool:
    return intent in SINGLETON_INTENTS or intent.startswith("validation_policy")


def _is_valid_action(action: RecommendedNextAction) -> bool:
    if not action.evidence_refs:
        return False
    if not _normalize_priority(action.priority):
        return False
    if not _is_meaningful_text(action.action):
        return False
    if not _is_meaningful_text(action.why):
        return False
    return True


def _is_meaningful_text(value: Any) -> bool:
    text = _clean_display_text(value)
    normalized = _normalize_action_text(text)
    return bool(normalized) and normalized not in {"", "-", "."} and len(normalized) >= 3


def _clean_display_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _normalize_action_text(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"[`*_#>\[\](){}]", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("cross validation", "cv")
    text = text.replace("crossvalidation", "cv")
    text = text.replace("out of fold", "oof")
    text = text.replace("out of sample", "oof")
    text = text.replace("public leaderboard", "public lb")
    text = text.replace("weight of evidence", "woe")
    text = text.replace("mean encoding", "target encoding")
    text = re.sub(r"\bstratified\s*k\s*fold\b", "stratified kfold", text)
    text = re.sub(r"\bstratifiedkfold\b", "stratified kfold", text)
    text = re.sub(r"\bk\s*fold\b", "kfold", text)
    text = re.sub(r"\bvalidation folds\b|\bcv folds\b", "validation folds", text)
    text = re.sub(r"\bclassification thresholds?\b", "threshold", text)
    text = re.sub(r"\bthreshold tuning\b|\btune thresholds?\b|\btuning thresholds?\b", "tune threshold", text)
    text = re.sub(r"\bprimary id\b|\bid column\b|\brow id\b", "primary id", text)
    text = re.sub(r"\bsanity check\b|\bsanity floor\b", "sanity floor", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _remove_intent_fillers(text: str) -> str:
    fillers = {
        "a",
        "an",
        "and",
        "as",
        "for",
        "inside",
        "model",
        "only",
        "run",
        "the",
        "to",
        "use",
        "with",
    }
    return " ".join(token for token in text.split() if token not in fillers)


def _primary_evidence_category(evidence_refs: list[str]) -> str:
    mapped = [_evidence_category(ref) for ref in evidence_refs]
    for category in (
        "leakage",
        "threshold_tuning",
        "validation",
        "metric",
        "baseline",
        "primary_id_safety",
        "submission_safety",
        "role_exclusion",
        "missingness",
        "categorical_features",
        "numeric_features",
        "text_features",
        "feature_probe",
        "drift",
    ):
        if category in mapped:
            return category
    return mapped[0] if mapped else "unknown"


def _evidence_category(ref: str) -> str:
    value = str(ref)
    if value.startswith("validation_evidence."):
        return "validation"
    if value == "metric_evidence.requires_threshold" or value.endswith(".requires_threshold"):
        return "threshold_tuning"
    if value.startswith("metric_evidence."):
        return "metric"
    if value == "inferred_schema.primary_id_column":
        return "primary_id_safety"
    if value == "inferred_schema.sample_submission_table":
        return "submission_safety"
    if value == "inferred_schema.global_roles":
        return "role_exclusion"
    if value.startswith("leakage_evidence"):
        return "leakage"
    if value.startswith("feature_probe_evidence"):
        return "feature_probe"
    if value.startswith("feature_diagnostics.numeric_feature_diagnostics"):
        return "numeric_features"
    if value.startswith("feature_diagnostics.categorical_feature_diagnostics"):
        return "categorical_features"
    if value.startswith("feature_diagnostics.missingness_diagnostics"):
        return "missingness"
    if value.startswith("feature_diagnostics.text_feature_diagnostics"):
        return "text_features"
    if value.startswith("feature_diagnostics"):
        return "feature_engineering"
    if value.startswith("drift_evidence"):
        return "drift"
    if value.startswith("baseline_evidence"):
        return "baseline"
    if value.startswith("baseline_ablation_evidence"):
        return "baseline_ablations"
    if value.startswith("interaction_diagnostics"):
        return "feature_engineering"
    if value.startswith("source_claim_validation"):
        return "feature_engineering"
    if value == "target_diagnostics.target_by_missingness":
        return "missingness"
    if value.startswith("target_diagnostics.imbalance"):
        return "validation"
    if value.startswith("target_diagnostics.distribution"):
        return "validation"
    if value.startswith("target_diagnostics.suspicious_patterns"):
        return "leakage"
    if value.startswith("target_diagnostics"):
        return "target"
    return value.split(".", 1)[0] if value else "unknown"


def _sorted_evidence_refs(refs: set[str]) -> list[str]:
    return sorted(refs, key=lambda ref: (CATEGORY_ORDER.get(_evidence_category(ref), 99), ref))


def _sorted_categories(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (CATEGORY_ORDER.get(value, 99), value))


def _category_sort_value(action: RecommendedNextAction) -> int:
    for category in action.source_categories:
        if category in CATEGORY_ORDER:
            return CATEGORY_ORDER[category]
    for ref in action.evidence_refs:
        category = _evidence_category(ref)
        if category in CATEGORY_ORDER:
            return CATEGORY_ORDER[category]
    return 99


def _action_text_rank(action: RecommendedNextAction) -> tuple[int, int, int, str]:
    return (
        -_safety_boundary_score(action.action),
        _source_rank(action),
        len(_normalize_action_text(action.action)),
        _normalize_action_text(action.action),
    )


def _action_text_rank_from_group(group: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        -_safety_boundary_score(str(group["action"])),
        int(group["source_rank"]),
        len(_normalize_action_text(str(group["action"]))),
        _normalize_action_text(str(group["action"])),
    )


def _safety_boundary_score(action: str) -> int:
    text = _normalize_action_text(action)
    score = 0
    if "inside validation folds" in text:
        score += 4
    if "validation only" in text or "validation data only" in text:
        score += 3
    if "oof" in text or "fold fitted" in text:
        score += 2
    if "test" in text and _has_any(text, ("do not", "avoid", "exclude")):
        score += 2
    return score


def _highest_priority(left: str, right: str) -> str:
    left = _normalize_priority(left)
    right = _normalize_priority(right)
    return left if ACTION_PRIORITY_ORDER[left] <= ACTION_PRIORITY_ORDER[right] else right


def _highest_risk(left: str | None, right: str | None) -> str | None:
    left = _normalize_risk(left)
    right = _normalize_risk(right)
    if left is None:
        return right
    if right is None:
        return left
    return left if RISK_ORDER[left] <= RISK_ORDER[right] else right


def _source_rank(action: RecommendedNextAction) -> int:
    return int(getattr(action, "_source_rank", LEGACY_SOURCE_RANK))


def _normalize_priority(value: Any) -> str:
    text = str(value or "P2").upper()
    return text if text in ACTION_PRIORITY_ORDER else "P2"


def _normalize_risk(value: Any) -> str | None:
    text = str(value or "").lower()
    return text if text in RISK_ORDER else None


def _merge_ordered_values(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


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


__all__ = [
    "build_recommended_next_actions",
    "dedupe_recommended_next_actions",
    "normalize_action_key",
]
