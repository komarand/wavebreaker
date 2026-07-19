from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Literal, Mapping

from pydantic import ConfigDict

from kaggle_researcher.contracts.base import ContractModel


_MERGED_LIST_FIELDS = (
    "evidence_refs",
    "related_hypothesis_ids",
    "hypothesis_ids",
    "limitations",
    "eda_result_refs",
    "experiment_ids",
    "source_refs",
    "risk_ids",
    "validation_requirement_ids",
    "safety_constraint_ids",
)
_IDENTITY_FIELDS = ("action", "reason", "priority")
ActionKind = Literal[
    "use_primary_validation",
    "tune_threshold",
    "exclude_feature",
    "audit_drift",
    "prevent_target_encoding_leakage",
    "reproduce_baseline",
    "prioritize_feature_family",
    "run_experiment",
    "preserve_schema_role",
    "other",
]
_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_ORIGIN_RANK = {
    "Fallback-generated": 0,
    "Hypothesis-to-test": 1,
    "Source-supported": 2,
    "EDA-inferred": 3,
    "EDA-confirmed": 4,
    "Safety-warning": 5,
}
_PRIMARY_SECTION_BY_KIND = {
    "use_primary_validation": "metric_and_validation",
    "tune_threshold": "metric_and_validation",
    "exclude_feature": "what_not_to_do",
    "audit_drift": "drift_and_leaderboard_risk",
    "prevent_target_encoding_leakage": "leakage_and_data_quality",
    "reproduce_baseline": "baseline_findings",
    "prioritize_feature_family": "feature_priorities",
    "run_experiment": "experiments_queue",
    "preserve_schema_role": "dataset_facts_from_eda",
}
_MANDATORY_SECTION_IDS = frozenset({
    "executive_summary", "metric_and_validation", "dataset_facts_from_eda",
    "leakage_and_data_quality", "drift_and_leaderboard_risk", "baseline_findings",
    "feature_priorities", "modeling_plan", "experiments_queue", "what_not_to_do",
    "first_48_hours",
})


class SectionActionMembership(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str
    action_ids: tuple[str, ...] = ()


class ActionCanonicalizationDiagnostics(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_action_ids: tuple[str, ...] = ()
    merged_duplicate_actions: tuple[str, ...] = ()
    conflicting_action_definitions: tuple[str, ...] = ()
    dangling_action_ids: tuple[str, ...] = ()
    section_memberships: tuple[SectionActionMembership, ...] = ()


class SemanticActionSignature(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_kind: ActionKind
    subject: str | None = None
    validation_strategy: str | None = None
    feature_or_column: str | None = None
    risk_type: str | None = None
    normalized_action_text: str
    canonical_key: str


class SemanticActionMerge(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_key: str
    original_id: str
    replacement_id: str


class SemanticActionCanonicalizationDiagnostics(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signatures: tuple[SemanticActionSignature, ...] = ()
    merges: tuple[SemanticActionMerge, ...] = ()


class FinalStrategyActionCanonicalizationError(RuntimeError):
    phase = "action_canonicalization"

    def __init__(self, diagnostics: ActionCanonicalizationDiagnostics) -> None:
        self.diagnostics = diagnostics
        parts: list[str] = []
        if diagnostics.conflicting_action_definitions:
            parts.append(
                f"{len(diagnostics.conflicting_action_definitions)} conflicting definitions"
            )
        if diagnostics.dangling_action_ids:
            parts.append(f"{len(diagnostics.dangling_action_ids)} dangling action IDs")
        super().__init__(
            "Final strategy action canonicalization failed: " + ", ".join(parts)
        )


def canonicalize_final_strategy_actions(
    raw_result: Mapping[str, Any],
) -> tuple[dict[str, Any], ActionCanonicalizationDiagnostics]:
    """Build one ordered action list and section membership references."""

    result = deepcopy(dict(raw_result))
    canonical: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    generated: list[str] = []
    merged: list[str] = []
    conflicts: list[str] = []
    dangling: list[str] = []

    def register(candidate: Any, source_path: str) -> str | None:
        candidate_mapping = _mapping_value(candidate)
        if candidate_mapping is None:
            return None
        action = deepcopy(dict(candidate_mapping))
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            action_id = _deterministic_action_id(action)
            action["action_id"] = action_id
            _append_unique(generated, action_id)
        existing = canonical.get(action_id)
        if existing is None:
            canonical[action_id] = action
            ordered_ids.append(action_id)
            return action_id
        definition_conflicts = [
            field
            for field in _IDENTITY_FIELDS
            if _has_conflict(existing.get(field), action.get(field))
        ]
        if definition_conflicts:
            conflicts.append(
                f"{action_id} at {source_path}: {', '.join(definition_conflicts)}"
            )
            return action_id
        for field in _MERGED_LIST_FIELDS:
            existing[field] = _unique_values([
                *_list_values(existing.get(field)),
                *_list_values(action.get(field)),
            ])
        for field, value in action.items():
            if field not in existing or existing[field] in (None, "", [], {}):
                existing[field] = deepcopy(value)
        _append_unique(merged, action_id)
        return action_id

    for index, action in enumerate(_list_values(result.get("actions"))):
        register(action, f"actions[{index}]")

    memberships: list[SectionActionMembership] = []
    sections: list[dict[str, Any]] = []
    for section_index, raw_section in enumerate(_list_values(result.get("sections"))):
        section_mapping = _mapping_value(raw_section)
        if section_mapping is None:
            continue
        section = deepcopy(dict(section_mapping))
        section_id = str(section.get("section_id") or f"section_{section_index + 1}")
        member_ids: list[str] = []
        for action_index, action in enumerate(_list_values(section.pop("actions", None))):
            action_id = register(
                action,
                f"sections[{section_index}].actions[{action_index}]",
            )
            if action_id:
                _append_unique(member_ids, action_id)
        for action_id_value in _list_values(section.get("action_ids")):
            action_id = str(action_id_value).strip()
            if action_id:
                _append_unique(member_ids, action_id)
        section["action_ids"] = member_ids
        sections.append(section)
        memberships.append(SectionActionMembership(
            section_id=section_id,
            action_ids=tuple(member_ids),
        ))

    for membership in memberships:
        for action_id in membership.action_ids:
            if action_id not in canonical:
                _append_unique(dangling, action_id)

    result["actions"] = [canonical[action_id] for action_id in ordered_ids]
    result["sections"] = sections
    diagnostics = ActionCanonicalizationDiagnostics(
        generated_action_ids=tuple(generated),
        merged_duplicate_actions=tuple(merged),
        conflicting_action_definitions=tuple(conflicts),
        dangling_action_ids=tuple(dangling),
        section_memberships=tuple(memberships),
    )
    if conflicts or dangling:
        raise FinalStrategyActionCanonicalizationError(diagnostics)
    return result, diagnostics


def canonicalize_semantic_strategy_actions(
    raw_result: Mapping[str, Any],
    *,
    primary_id: str | None = None,
) -> tuple[dict[str, Any], SemanticActionCanonicalizationDiagnostics]:
    """Merge exact-text and known semantic action duplicates deterministically."""

    result, _ = canonicalize_final_strategy_actions(raw_result)
    raw_actions = [
        dict(action) for action in _list_values(result.get("actions"))
        if isinstance(action, Mapping)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    signatures_by_id: dict[str, SemanticActionSignature] = {}
    for action in raw_actions:
        action_id = str(action.get("action_id") or "").strip()
        signature = semantic_action_signature(action, primary_id=primary_id)
        signatures_by_id[action_id] = signature
        grouped.setdefault(signature.canonical_key, []).append(action)

    canonical_actions: list[dict[str, Any]] = []
    old_to_new: dict[str, str] = {}
    signatures: list[SemanticActionSignature] = []
    merges: list[SemanticActionMerge] = []
    action_kind_by_id: dict[str, ActionKind] = {}
    for canonical_key, candidates in grouped.items():
        signature = semantic_action_signature(candidates[0], primary_id=primary_id)
        signatures.append(signature)
        survivor_id = _surviving_action_id(candidates, signature)
        merged = _merge_semantic_action_group(candidates, survivor_id)
        canonical_actions.append(merged)
        action_kind_by_id[survivor_id] = signature.action_kind
        for candidate in candidates:
            original_id = str(candidate["action_id"])
            old_to_new[original_id] = survivor_id
            if len(candidates) > 1 and original_id != survivor_id:
                merges.append(SemanticActionMerge(
                    canonical_key=canonical_key,
                    original_id=original_id,
                    replacement_id=survivor_id,
                ))

    sections = [
        deepcopy(dict(section))
        for section in _list_values(result.get("sections"))
        if isinstance(section, Mapping)
    ]
    original_action_by_id = {
        str(action["action_id"]): action for action in raw_actions
    }
    original_section_evidence: dict[int, list[str]] = {}
    for section_index, section in enumerate(sections):
        original_section_evidence[section_index] = sorted({
            str(value)
            for action_id in _list_values(section.get("action_ids"))
            for value in _list_values(
                original_action_by_id.get(str(action_id), {}).get("evidence_refs")
            )
            if str(value).strip()
        })
    memberships: dict[str, list[int]] = {action["action_id"]: [] for action in canonical_actions}
    for section_index, section in enumerate(sections):
        rewritten: list[str] = []
        for action_id in _list_values(section.get("action_ids")):
            survivor_id = old_to_new.get(str(action_id), str(action_id))
            if survivor_id in memberships and survivor_id not in rewritten:
                rewritten.append(survivor_id)
        section["action_ids"] = rewritten
        for block in _list_values(section.get("time_blocks")):
            if not isinstance(block, dict):
                continue
            block["action_ids"] = _unique_values([
                old_to_new.get(str(action_id), str(action_id))
                for action_id in _list_values(block.get("action_ids"))
            ])
        for action_id in rewritten:
            memberships[action_id].append(section_index)

    section_index_by_id = {
        str(section.get("section_id") or ""): index
        for index, section in enumerate(sections)
    }
    for action in canonical_actions:
        action_id = str(action["action_id"])
        preferred_section_id = _PRIMARY_SECTION_BY_KIND.get(
            action_kind_by_id[action_id]
        )
        preferred_index = section_index_by_id.get(preferred_section_id or "")
        locations = memberships[action_id]
        if preferred_index is not None:
            chosen_index = preferred_index
        elif locations:
            chosen_index = min(locations)
        elif sections:
            chosen_index = 0
        else:
            continue
        for section in sections:
            section["action_ids"] = [
                value for value in section.get("action_ids", [])
                if value != action_id
            ]
        sections[chosen_index].setdefault("action_ids", []).append(action_id)

    action_by_id = {
        str(action["action_id"]): action for action in canonical_actions
    }
    retained_sections: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections):
        section_actions = [
            action_by_id[action_id]
            for action_id in section.get("action_ids", [])
            if action_id in action_by_id
        ]
        if not section_actions:
            if not _list_values(section.get("evidence_refs")):
                section_id = str(section.get("section_id") or "")
                fallback_evidence = original_section_evidence.get(section_index, [])
                if fallback_evidence:
                    section["evidence_refs"] = fallback_evidence
                elif section_id not in _MANDATORY_SECTION_IDS:
                    continue
                else:
                    section["evidence_refs"] = ["final_synthesizer.repaired"]
            retained_sections.append(section)
            continue
        for field in (
            "evidence_refs", "source_refs", "eda_result_refs",
            "related_hypothesis_ids",
        ):
            source_field = "hypothesis_ids" if field == "related_hypothesis_ids" else field
            section[field] = sorted({
                str(value)
                for action in section_actions
                for value in _list_values(action.get(source_field))
                if str(value).strip()
            })
        retained_sections.append(section)
    sections = retained_sections

    repairs = [
        dict(repair) for repair in _list_values(result.get("reference_repairs"))
        if isinstance(repair, Mapping)
    ]
    for merge in merges:
        repair = {
            "field_path": f"semantic_action_merge.{merge.canonical_key}",
            "original_id": merge.original_id,
            "replacement_id": merge.replacement_id,
        }
        if repair not in repairs:
            repairs.append(repair)

    result["actions"] = canonical_actions
    result["sections"] = sections
    result["reference_repairs"] = repairs
    result.pop("action_provenance", None)
    diagnostics = SemanticActionCanonicalizationDiagnostics(
        signatures=tuple(signatures),
        merges=tuple(sorted(
            merges,
            key=lambda item: (
                item.canonical_key, item.original_id, item.replacement_id
            ),
        )),
    )
    validate_semantic_action_postconditions(result, primary_id=primary_id)
    return result, diagnostics


def semantic_action_signature(
    action: Mapping[str, Any],
    *,
    primary_id: str | None = None,
) -> SemanticActionSignature:
    raw_text = str(action.get("action") or "")
    text = _semantic_text(raw_text)
    local_experiment_id = str(action.get("experiment_id") or "").strip()
    if local_experiment_id:
        return SemanticActionSignature(
            action_kind="run_experiment",
            subject=local_experiment_id,
            validation_strategy=None,
            feature_or_column=None,
            risk_type=str(action.get("risk") or "experiment"),
            normalized_action_text=text,
            canonical_key=f"run_experiment:{local_experiment_id}",
        )
    strategy = _validation_strategy(action, text)
    action_kind: ActionKind = "other"
    subject: str | None = None
    feature_or_column: str | None = None
    risk_type: str | None = None

    if strategy and "validation" in text and any(
        marker in text for marker in ("primary", "selected by eda", "policy", "method")
    ):
        action_kind = "use_primary_validation"
        subject = strategy
    elif (
        "threshold" in text
        and "calibrat" not in text
        and any(marker in text for marker in ("tune", "select", "optimize", "search"))
    ):
        action_kind = "tune_threshold"
        subject = "decision_threshold"
    elif _is_feature_exclusion(text):
        action_kind = "exclude_feature"
        feature_or_column = _excluded_feature(raw_text, text, primary_id)
        subject = "primary_id" if _is_primary_id_reference(
            raw_text, text, primary_id
        ) else "model_feature"
    elif "drift" in text and "temporal cv" not in text and "temporal validation" not in text:
        action_kind = "audit_drift"
        if any(marker in text for marker in ("primary id", "identifier", "index drift")):
            subject = "identifier_drift_artifact"
            risk_type = "drift_artifact"
        else:
            subject = "train_test_drift"
            risk_type = "drift"
    elif (
        ("target encoding" in text or "woe" in text)
        and any(marker in text for marker in ("oof", "fold fitted", "avoid", "do not", "leak"))
    ):
        action_kind = "prevent_target_encoding_leakage"
        subject = "target_encoding"
        risk_type = "leakage"
    elif "baseline" in text and any(
        marker in text for marker in ("reproduce", "establish", "run", "anchor")
    ):
        action_kind = "reproduce_baseline"
        subject = "baseline"
    elif "feature" in text and any(
        marker in text for marker in ("prioritize", "priority", "focus")
    ):
        action_kind = "prioritize_feature_family"
        subject = "feature_family"
    elif "schema" in text and any(
        marker in text for marker in ("preserve", "role", "alignment")
    ):
        action_kind = "preserve_schema_role"
        subject = "schema_role"
    elif text.startswith("run ") or " experiment" in f" {text}":
        action_kind = "run_experiment"
        subject = text

    if action_kind == "use_primary_validation":
        canonical_key = f"use_primary_validation:{strategy}"
    elif action_kind == "exclude_feature":
        canonical_key = (
            f"exclude_feature:{feature_or_column or 'unknown'}:{subject or 'model_feature'}"
        )
    elif action_kind == "audit_drift":
        canonical_key = f"audit_drift:{subject}"
    elif action_kind == "prevent_target_encoding_leakage":
        canonical_key = f"prevent_target_encoding_leakage:{risk_type or subject}"
    elif action_kind in {
        "tune_threshold", "reproduce_baseline", "prioritize_feature_family",
        "preserve_schema_role",
    }:
        canonical_key = f"{action_kind}:{subject}"
    else:
        canonical_key = f"{action_kind}:{text}"
    return SemanticActionSignature(
        action_kind=action_kind,
        subject=subject,
        validation_strategy=strategy,
        feature_or_column=feature_or_column,
        risk_type=risk_type,
        normalized_action_text=text,
        canonical_key=canonical_key,
    )


def validate_semantic_action_postconditions(
    result: Mapping[str, Any],
    *,
    primary_id: str | None = None,
) -> None:
    actions = [
        action for action in _list_values(result.get("actions"))
        if isinstance(action, Mapping)
    ]
    action_ids = [str(action.get("action_id") or "") for action in actions]
    if not all(action_ids) or len(action_ids) != len(set(action_ids)):
        raise ValueError("Semantic action postcondition failed: duplicate action IDs")
    keys = [
        semantic_action_signature(action, primary_id=primary_id).canonical_key
        for action in actions
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Semantic action postcondition failed: duplicate canonical semantic keys"
        )
    memberships: dict[str, int] = {action_id: 0 for action_id in action_ids}
    for section in _list_values(result.get("sections")):
        if not isinstance(section, Mapping):
            continue
        section_action_ids = [str(value) for value in _list_values(section.get("action_ids"))]
        if len(section_action_ids) != len(set(section_action_ids)):
            raise ValueError(
                "Semantic action postcondition failed: duplicate section action IDs"
            )
        unknown = set(section_action_ids) - set(action_ids)
        if unknown:
            raise ValueError(
                "Semantic action postcondition failed: unresolved section action IDs "
                + repr(sorted(unknown))
            )
        for action_id in section_action_ids:
            memberships[action_id] += 1
    if result.get("sections"):
        orphaned = sorted(
            action_id for action_id, count in memberships.items() if count == 0
        )
        duplicated = sorted(
            action_id for action_id, count in memberships.items() if count > 1
        )
        if orphaned or duplicated:
            raise ValueError(
                "Semantic action postcondition failed: invalid section membership; "
                f"orphaned={orphaned}, repeated={duplicated}"
            )


def _deterministic_action_id(action: Mapping[str, Any]) -> str:
    stable = {
        "priority": _normalized_text(action.get("priority")),
        "action": _normalized_text(action.get("action")),
        "reason": _normalized_text(action.get("reason")),
    }
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"action_{digest}"


def _surviving_action_id(
    candidates: list[dict[str, Any]],
    signature: SemanticActionSignature,
) -> str:
    identifiers = sorted({str(action["action_id"]) for action in candidates})
    stable = sorted(identifiers, key=lambda value: (_stable_id_rank(value), value))
    if stable and _stable_id_rank(stable[0]) < 2:
        return stable[0]
    if len(candidates) == 1:
        return identifiers[0]
    slug = re.sub(r"[^a-z0-9]+", "_", signature.action_kind).strip("_")
    digest = hashlib.sha256(
        signature.canonical_key.encode("utf-8")
    ).hexdigest()[:12]
    return f"action_{slug}_{digest}"


def _stable_id_rank(value: str) -> int:
    if value.startswith((
        "eda_", "contract_", "safety_", "validation_", "risk_",
    )):
        return 0
    if not re.fullmatch(r"action_[0-9a-f]{16}", value):
        return 1
    return 2


def _merge_semantic_action_group(
    candidates: list[dict[str, Any]],
    survivor_id: str,
) -> dict[str, Any]:
    wording_source = max(
        candidates,
        key=lambda action: _wording_specificity(action.get("action")),
    )
    reason_source = max(
        candidates,
        key=lambda action: _wording_specificity(action.get("reason")),
    )
    merged = deepcopy(wording_source)
    merged["action_id"] = survivor_id
    merged["action"] = wording_source.get("action")
    merged["reason"] = reason_source.get("reason")
    merged["priority"] = min(
        (str(action.get("priority") or "P3") for action in candidates),
        key=lambda value: (_PRIORITY_RANK.get(value, 99), value),
    )
    merged["confidence"] = max(
        (str(action.get("confidence") or "low") for action in candidates),
        key=lambda value: (_CONFIDENCE_RANK.get(value, -1), value),
    )
    for field in _MERGED_LIST_FIELDS:
        merged[field] = sorted({
            str(value)
            for action in candidates
            for value in _list_values(action.get(field))
            if str(value).strip()
        })
    origins = [
        str(action.get("evidence_origin") or "Hypothesis-to-test")
        for action in candidates
    ]
    merged["evidence_origin"] = max(
        origins,
        key=lambda value: (_ORIGIN_RANK.get(value, -1), value),
    )
    source_claims = [
        str(action.get("source_claim") or "").strip()
        for action in candidates if str(action.get("source_claim") or "").strip()
    ]
    merged["source_claim"] = (
        max(source_claims, key=_wording_specificity) if source_claims else None
    )
    strategies = sorted({
        str(action.get("validation_strategy"))
        for action in candidates if action.get("validation_strategy")
    })
    merged["validation_strategy"] = strategies[0] if strategies else None
    return merged


def _wording_specificity(value: Any) -> tuple[int, int, str]:
    text = " ".join(str(value or "").split())
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    return len(set(tokens)), len(text), text.casefold()


def _validation_strategy(action: Mapping[str, Any], text: str) -> str | None:
    provided = str(action.get("validation_strategy") or "").strip().casefold()
    if provided:
        return provided.replace("-", "_").replace(" ", "_")
    aliases = (
        ("stratified group kfold", "stratified_group_kfold"),
        ("stratified group cv", "stratified_group_kfold"),
        ("stratified kfold", "stratified_kfold"),
        ("stratified k fold", "stratified_kfold"),
        ("stratified cv", "stratified_kfold"),
        ("group kfold", "group_kfold"),
        ("group cv", "group_kfold"),
        ("temporal holdout", "temporal_holdout"),
        ("temporal cv", "temporal_cv"),
        ("ranking group cv", "ranking_group_cv"),
        ("kfold", "kfold"),
        ("k fold", "kfold"),
    )
    for marker, canonical in aliases:
        if marker in text:
            return canonical
    return None


def _is_feature_exclusion(text: str) -> bool:
    if any(marker in text for marker in ("test ", "ablation", "diagnostic feature")):
        return False
    exclusion = any(marker in text for marker in (
        "exclude ", "excluded from", "do not use", "keep the primary id excluded",
        "not use", "remove from model features",
    ))
    feature_context = any(marker in text for marker in (
        "feature", "predictive", "primary id", "identifier", "model input",
    ))
    return exclusion and feature_context


def _is_primary_id_reference(
    raw_text: str,
    text: str,
    primary_id: str | None,
) -> bool:
    if any(marker in text for marker in ("primary id", "primary identifier")):
        return True
    return bool(primary_id and primary_id.casefold() in raw_text.casefold())


def _excluded_feature(
    raw_text: str,
    text: str,
    primary_id: str | None,
) -> str:
    if _is_primary_id_reference(raw_text, text, primary_id):
        return str(primary_id or "primary_id")
    patterns = (
        r"(?i)\bexclude\s+[`'\"]?([A-Za-z_][A-Za-z0-9_]*)",
        r"(?i)\bdo not use\s+[`'\"]?([A-Za-z_][A-Za-z0-9_]*)",
        r"(?i)\bkeep\s+[`'\"]?([A-Za-z_][A-Za-z0-9_]*)\s+excluded",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_text)
        if match and match.group(1).casefold() not in {"the", "a", "an"}:
            return match.group(1)
    return "unknown"


def _semantic_text(value: Any) -> str:
    text = str(value or "").casefold().replace("`", " ")
    replacements = {
        "cross-validation": "cross validation",
        "stratifiedkfold": "stratified kfold",
        "stratified_kfold": "stratified kfold",
        "stratified-group-kfold": "stratified group kfold",
        "stratified_group_kfold": "stratified group kfold",
        "group_kfold": "group kfold",
        "temporal_cv": "temporal cv",
        "target-encoding": "target encoding",
        "fold-fitted": "fold fitted",
        "public leaderboard": "public lb",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b(?:p0|p1|p2|p3)\b", " ", text)
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return " ".join(text.split())


def _has_conflict(left: Any, right: Any) -> bool:
    if left in (None, "") or right in (None, ""):
        return False
    return _normalized_text(left) != _normalized_text(right)


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _list_values(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _mapping_value(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _unique_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


__all__ = [
    "ActionKind",
    "ActionCanonicalizationDiagnostics",
    "FinalStrategyActionCanonicalizationError",
    "SemanticActionCanonicalizationDiagnostics",
    "SemanticActionMerge",
    "SemanticActionSignature",
    "SectionActionMembership",
    "canonicalize_final_strategy_actions",
    "canonicalize_semantic_strategy_actions",
    "semantic_action_signature",
    "validate_semantic_action_postconditions",
]
