from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping

from kaggle_researcher.contracts.final_strategy import (
    ActionProvenance,
    EvidenceCatalogEntry,
    ExperimentArm,
    ExperimentBudget,
    FeatureActionMetadata,
    FeatureExperimentFamily,
    FinalStrategyAction,
    FinalStrategyExperiment,
    FinalStrategyQualityMetrics,
    FinalStrategyResult,
    FinalStrategySection,
    HypothesisToEdaLink,
    REQUIRED_SECTION_IDS,
    SourceToHypothesisLink,
)
from kaggle_researcher.contracts.final_strategy_protocol import (
    PromptFingerprint,
    SelectionStatus,
    StrategyRenderingDraft,
    StrategySelectionDraft,
    StrategySkeleton,
)
from kaggle_researcher.contracts.synthesis_context import FinalSynthesisContext
from kaggle_researcher.reasoning.final_strategy_context import FinalStrategySelectionContext
from kaggle_researcher.reasoning.model_registry import (
    model_registry,
    resolve_model_identity,
    supported_models,
)


BRIDGE_POLICY_VERSION = "2.0"
_COST = {"low": 0, "medium": 1, "high": 2}
_KIND_ORDER = {
    "validation_setup": 0, "safety_setup": 1, "baseline_reproduction": 2,
    "feature_family": 3, "data_quality": 3, "model_comparison": 4,
    "provisional_selection": 5, "oof_generation": 6,
    "threshold_postprocessing": 7, "calibration_postprocessing": 7,
    "final_refit": 8, "submission_integrity": 9,
}
_DEPENDENCY_SENTINELS = {
    "provisional_model_selected",
    "oof_predictions_available",
}


class StrategyBridgeError(ValueError):
    pass


def freeze_strategy_selection(
    draft: StrategySelectionDraft,
    *,
    synthesis_context: FinalSynthesisContext,
    selection_context: FinalStrategySelectionContext,
    selection_status: SelectionStatus,
    selection_prompt_fingerprint: PromptFingerprint,
) -> tuple[StrategySkeleton, dict[str, Any]]:
    """Validate, canonicalize, budget, and freeze a Call 1 selection."""

    allowed = _allowed_catalogs(selection_context)
    _validate_selection_references(draft, allowed)
    limits = selection_context.strategy_limits
    initial_action_count = len(draft.selected_actions)
    if initial_action_count > limits["max_actions"]:
        raise StrategyBridgeError(
            f"selected_actions exceeds max_actions={limits['max_actions']}"
        )
    if [item.section_id for item in draft.section_plan] != REQUIRED_SECTION_IDS:
        raise StrategyBridgeError(
            f"section_plan must use canonical order exactly: {REQUIRED_SECTION_IDS}"
        )
    _validate_client_key_references(draft)

    action_key_map: dict[str, str] = {}
    actions: list[FinalStrategyAction] = []
    seen_action_semantics: set[str] = set()
    action_id_by_semantic: dict[str, str] = {}
    evidence_removed = 0
    for item in draft.selected_actions:
        semantic = _semantic_key(item.action_kind, item.action)
        if semantic in seen_action_semantics:
            action_key_map[item.client_action_key] = action_id_by_semantic[semantic]
            continue
        seen_action_semantics.add(semantic)
        action_id = _stable_id("action", semantic)
        action_id_by_semantic[semantic] = action_id
        action_key_map[item.client_action_key] = action_id
        refs, removed = _minimal_evidence(
            item.primary_evidence_refs,
            item.supporting_evidence_refs,
            item.limitation_evidence_refs,
        )
        evidence_removed += removed
        feature_metadata = (
            FeatureActionMetadata.model_validate(item.feature_metadata)
            if item.feature_metadata else None
        )
        hypotheses = _unique([
            *item.motivating_hypothesis_ids, *item.safety_hypothesis_ids,
            *item.validation_context_ids, *item.rejected_hypothesis_ids,
        ])
        actions.append(FinalStrategyAction(
            action_id=action_id,
            action_kind=_canonical_kind(item.action_kind),
            priority=item.priority,
            confidence=item.confidence,
            action=item.action,
            reason=item.reason,
            evidence_refs=refs,
            primary_evidence_refs=refs[:1],
            limitation_evidence_refs=[
                ref for ref in item.limitation_evidence_refs if ref in refs
            ][:2],
            related_hypothesis_ids=hypotheses,
            motivating_hypothesis_ids=item.motivating_hypothesis_ids,
            safety_hypothesis_ids=item.safety_hypothesis_ids,
            validation_context_ids=item.validation_context_ids,
            rejected_hypothesis_ids=item.rejected_hypothesis_ids,
            source_refs=_validated_action_sources(item.source_refs, hypotheses, allowed),
            eda_result_refs=[ref for ref in refs if ref in allowed["evidence"]],
            safety_constraint_ids=item.safety_constraint_ids,
            validation_requirement_ids=item.validation_requirement_ids,
            experiment_ids=item.approved_experiment_ids,
            validation_strategy=_validation_method(selection_context),
            evidence_origin=_evidence_origin(item.action_kind, bool(item.source_refs)),
            feature_metadata=feature_metadata,
            dependencies=_unique(item.dependencies),
            limitations=item.limitations,
        ))
    actions = [
        item.model_copy(update={
            "dependencies": [action_key_map[value] for value in item.dependencies]
        })
        for item in actions
    ]

    family_key_map: dict[str, str] = {}
    families: list[FeatureExperimentFamily] = []
    schema_columns = _schema_columns(selection_context.schema_summary)
    for item in draft.feature_experiment_families:
        missing = [column for column in item.input_columns if column.casefold() not in schema_columns]
        if missing:
            raise StrategyBridgeError(
                f"feature family {item.client_family_key!r} uses unavailable columns: {missing}"
            )
        family_id = _stable_id(
            "feature_family", _semantic_key(item.client_family_key, *item.input_columns)
        )
        family_key_map[item.client_family_key] = family_id
        refs, removed = _minimal_evidence(
            item.primary_evidence_refs, item.supporting_evidence_refs, []
        )
        evidence_removed += removed
        family = FeatureExperimentFamily(
            family_id=family_id,
            name=item.name,
            priority=item.priority,
            input_columns=item.input_columns,
            hypothesis=item.hypothesis,
            baseline_arm=_compile_arm(item.baseline_arm, family_id),
            candidate_arms=[_compile_arm(arm, family_id) for arm in item.candidate_arms],
            validation_strategy=item.validation_strategy,
            metric=item.metric_name,
            fit_scope=(
                "within_fold" if any(arm.fit_scope == "within_fold" for arm in item.candidate_arms)
                else item.baseline_arm.fit_scope
            ),
            evidence_refs=refs,
            motivating_hypothesis_ids=item.motivating_hypothesis_ids,
            risks=item.risks,
            acceptance_rule=item.acceptance_rule,
            estimated_cost=item.estimated_cost,
            status="planned",
        )
        families.append(family)

    experiment_key_map: dict[str, str] = {}
    experiments: list[FinalStrategyExperiment] = []
    seen_experiment_semantics: set[str] = set()
    experiment_id_by_semantic: dict[str, str] = {}
    warnings: list[str] = []
    self_comparisons_removed = 0
    for item in draft.candidate_experiments:
        baseline = resolve_model_identity(item.model_family_id)
        candidate = resolve_model_identity(item.comparison_model_family_id)
        if item.model_family_id and baseline is None:
            raise StrategyBridgeError(f"unknown model family: {item.model_family_id}")
        if item.comparison_model_family_id and candidate is None:
            raise StrategyBridgeError(
                f"unknown comparison model family: {item.comparison_model_family_id}"
            )
        if baseline and candidate and baseline.canonical_family_id == candidate.canonical_family_id:
            self_comparisons_removed += 1
            warnings.append(
                f"Removed self-comparison {item.client_experiment_key!r}; both aliases resolve to {baseline.canonical_family_id}."
            )
            continue
        canonical_model = candidate or baseline or _default_model(synthesis_context.plan_data.task_type)
        if canonical_model is None:
            raise StrategyBridgeError("no task-compatible model is available")
        semantic = _semantic_key(
            item.experiment_kind, item.name, item.family_key or "",
            canonical_model.canonical_family_id,
        )
        if semantic in seen_experiment_semantics:
            experiment_key_map[item.client_experiment_key] = experiment_id_by_semantic[
                semantic
            ]
            warnings.append(
                f"Merged duplicate experiment {item.client_experiment_key!r}."
            )
            continue
        seen_experiment_semantics.add(semantic)
        experiment_id = _stable_id("experiment", semantic)
        experiment_id_by_semantic[semantic] = experiment_id
        experiment_key_map[item.client_experiment_key] = experiment_id
        refs, removed = _minimal_evidence(
            item.primary_evidence_refs, item.supporting_evidence_refs, []
        )
        evidence_removed += removed
        experiments.append(FinalStrategyExperiment(
            experiment_id=experiment_id,
            priority=item.priority,
            name=item.name,
            hypothesis=item.hypothesis,
            change=item.exact_change,
            feature_inputs=(
                next((family.input_columns for family in families
                      if family.family_id == family_key_map.get(item.family_key or "")), [])
            ),
            model_family=canonical_model.display_name,
            validation_strategy=item.validation_strategy,
            success_metric=item.metric_name,
            acceptance_rule=item.acceptance_rule,
            evidence_refs=refs,
            primary_evidence_refs=refs[:1],
            related_hypothesis_ids=_unique([
                *item.motivating_hypothesis_ids, *item.safety_hypothesis_ids,
                *item.validation_context_ids,
            ]),
            risks=item.risks,
            fit_scope=("oof_only" if _is_threshold(item.experiment_kind) else "within_fold"),
            baseline_canonical_family_id=(baseline.canonical_family_id if baseline else None),
            baseline_implementation_id=(baseline.implementation_id if baseline else None),
            candidate_canonical_family_id=(candidate.canonical_family_id if candidate else canonical_model.canonical_family_id),
            candidate_implementation_id=(candidate.implementation_id if candidate else canonical_model.implementation_id),
            estimated_cost=item.estimated_cost,
            dependencies=_unique(item.dependencies),
        ))

    _validate_required_execution_steps(draft, experiments, selection_context)
    experiments = _canonicalize_dependencies(experiments, experiment_key_map)
    _validate_dependency_graph(experiments)
    experiments.sort(key=_experiment_rank)
    budget = ExperimentBudget(
        max_core_experiments=limits["max_core_experiments"],
        max_first_24h_experiments=limits["max_first_24h_experiments"],
        max_first_48h_experiments=limits["max_first_48h_experiments"],
        max_high_cost_experiments=2,
        estimated_total_cost=sum({"low": 1.0, "medium": 2.0, "high": 4.0}[item.estimated_cost] for item in experiments),
        budget_policy_version=BRIDGE_POLICY_VERSION,
    )
    proposed_core = set(draft.proposed_core_experiment_ids)
    proposed_backlog = set(draft.proposed_backlog_experiment_ids)
    preferred = sorted(
        experiments,
        key=lambda item: (
            experiment_key_map_inv(experiment_key_map, item.experiment_id) not in proposed_core,
            *_experiment_rank(item),
        ),
    )
    client_key_by_id = {
        value: key for key, value in experiment_key_map.items()
    }
    required_core = [item for item in preferred if _is_baseline(item.name)]
    selected_core = [
        item for item in preferred
        if client_key_by_id.get(item.experiment_id) in proposed_core
        and item.priority != "P2"
        and item not in required_core
    ]
    core = [*required_core, *selected_core][: budget.max_core_experiments]
    core_ids = {item.experiment_id for item in core}
    backlog_candidates = sorted(
        (item for item in preferred if item.experiment_id not in core_ids),
        key=lambda item: (
            client_key_by_id.get(item.experiment_id) not in proposed_backlog,
            item.priority != "P2",
            *_experiment_rank(item),
        ),
    )
    backlog = backlog_candidates[: limits["max_backlog_experiments"]]
    core = _force_baseline_first(core)
    core = _force_threshold_late(core)
    core_ids = {item.experiment_id for item in core}
    core = [item.model_copy(update={"status": "required" if _is_baseline(item.name) else "core"}) for item in core]
    backlog = [item.model_copy(update={"status": "backlog"}) for item in backlog if item.experiment_id not in core_ids]

    action_by_key = {item.client_action_key: action_key_map.get(item.client_action_key) for item in draft.selected_actions}
    sections = _compile_sections(
        draft, actions, action_by_key, experiment_key_map, family_key_map,
        core, budget,
    )
    source_links, eda_links = _compile_provenance(
        synthesis_context, actions, families, [*core, *backlog], allowed
    )
    action_provenance = [
        ActionProvenance(
            action_id=action.action_id or "",
            source_refs=action.source_refs,
            hypothesis_ids=action.hypothesis_ids,
            motivating_hypothesis_ids=action.motivating_hypothesis_ids,
            safety_hypothesis_ids=action.safety_hypothesis_ids,
            validation_context_ids=action.validation_context_ids,
            eda_result_refs=action.eda_result_refs,
        ) for action in actions
    ]
    evidence_catalog = _compile_evidence_catalog(
        selection_context,
        _used_evidence(actions, families, [*core, *backlog]),
    )
    dependency_graph = {
        item.experiment_id: list(item.dependencies) for item in [*core, *backlog]
    }
    limitations = _unique([*draft.limitations, *warnings])
    structure = [section.model_dump(mode="json") for section in sections]
    skeleton_material: dict[str, Any] = {
        "contract_family": "strategy_skeleton",
        "schema_version": "2.0",
        "skeleton_schema_version": "2.0",
        "competition_id": synthesis_context.eda_evidence_pack.competition_id,
        "task_type": synthesis_context.plan_data.task_type,
        "metric": {"name": synthesis_context.plan_data.metric},
        "recommended_validation": _validation_method(selection_context),
        "synthesis_selection_status": selection_status,
        "evidence_catalog": {key: value.model_dump(mode="json") for key, value in evidence_catalog.items()},
        "source_to_hypothesis_links": [item.model_dump(mode="json") for item in source_links],
        "hypothesis_to_eda_links": [item.model_dump(mode="json") for item in eda_links],
        "actions": [item.model_dump(mode="json") for item in actions],
        "action_provenance": [item.model_dump(mode="json") for item in action_provenance],
        "feature_experiment_families": [item.model_dump(mode="json") for item in families],
        "core_experiments": [item.model_dump(mode="json") for item in core],
        "experiment_backlog": [item.model_dump(mode="json") for item in backlog],
        "experiment_budget": budget.model_dump(mode="json"),
        "dependency_graph": dependency_graph,
        "section_structure": structure,
        "validation_requirement_ids": sorted({str(value) for action in actions for value in action.validation_requirement_ids}),
        "safety_constraint_ids": sorted({str(value) for action in actions for value in action.safety_constraint_ids}),
        "limitations": limitations,
        "warnings": warnings,
        "selection_prompt_fingerprint": selection_prompt_fingerprint.model_dump(mode="json"),
        "client_key_map": {
            "actions": action_key_map, "families": family_key_map,
            "experiments": experiment_key_map,
        },
    }
    skeleton_hash = _stable_hash(skeleton_material)
    skeleton = StrategySkeleton.model_validate({
        **skeleton_material,
        "skeleton_id": f"skeleton_{skeleton_hash[:16]}",
        "skeleton_hash": skeleton_hash,
    })
    diagnostics = {
        "initial_action_count": initial_action_count,
        "canonical_action_count": len(actions),
        "duplicates_merged": initial_action_count - len(actions),
        "evidence_refs_removed": evidence_removed,
        "evidence_refs_retained": len(_used_evidence(actions, families, [*core, *backlog])),
        "source_links_preserved": len(source_links),
        "hypothesis_roles_reassigned": 0,
        "self_model_comparisons_removed": self_comparisons_removed,
        "feature_experiments_grouped": len(families),
        "candidate_experiment_count": len(experiments),
        "core_experiment_count": len(core),
        "backlog_experiment_count": len(backlog),
        "first_48h_experiment_count": sum(len(block.experiment_ids) for block in sections[-1].time_blocks),
        "dependency_repairs": 0,
        "quality_gate_issues": [],
        "client_key_map": skeleton.client_key_map,
    }
    return skeleton, diagnostics


def skeleton_to_result(
    skeleton: StrategySkeleton,
    *,
    rendering: StrategyRenderingDraft | None,
    rendering_status: str,
    rendering_prompt_fingerprint: PromptFingerprint,
    diagnostics_path: str | None,
    selection_model: str,
    rendering_model: str,
    additional_warnings: Iterable[str] = (),
) -> FinalStrategyResult:
    validate_skeleton_integrity(skeleton)
    payload = {
        "contract_family": "final_strategy", "schema_version": "2.0",
        "competition_id": skeleton.competition_id,
        "synthesis_status": skeleton.synthesis_selection_status,
        "selection_status": skeleton.synthesis_selection_status,
        "rendering_status": rendering_status,
        "llm_output_valid": skeleton.synthesis_selection_status == "llm_success",
        "repair_attempted": skeleton.synthesis_selection_status == "repaired_success",
        "repair_succeeded": skeleton.synthesis_selection_status == "repaired_success",
        "fallback_used": skeleton.synthesis_selection_status == "degraded_fallback",
        "synthesis_diagnostics_path": diagnostics_path,
        "skeleton_id": skeleton.skeleton_id,
        "skeleton_hash": skeleton.skeleton_hash,
        "selection_prompt_fingerprint": skeleton.selection_prompt_fingerprint.model_dump(mode="json"),
        "rendering_prompt_fingerprint": rendering_prompt_fingerprint.model_dump(mode="json"),
        "task_type": skeleton.task_type,
        "metric": skeleton.metric,
        "recommended_validation": skeleton.recommended_validation,
        "actions": deepcopy(skeleton.actions),
        "feature_experiment_families": deepcopy(skeleton.feature_experiment_families),
        "core_experiments": deepcopy(skeleton.core_experiments),
        "experiment_backlog": deepcopy(skeleton.experiment_backlog),
        "experiments": deepcopy([*skeleton.core_experiments, *skeleton.experiment_backlog]),
        "experiment_budget": deepcopy(skeleton.experiment_budget),
        "dependency_graph": deepcopy(skeleton.dependency_graph),
        "sections": deepcopy(skeleton.section_structure),
        "evidence_catalog": deepcopy(skeleton.evidence_catalog),
        "source_to_hypothesis_links": deepcopy(skeleton.source_to_hypothesis_links),
        "hypothesis_to_eda_links": deepcopy(skeleton.hypothesis_to_eda_links),
        "action_provenance": deepcopy(skeleton.action_provenance),
        "selected_validation_requirement_ids": skeleton.validation_requirement_ids,
        "enforced_safety_constraint_ids": skeleton.safety_constraint_ids,
        "limitations": skeleton.limitations,
        "warnings": _unique([*skeleton.warnings, *additional_warnings]),
        "models_used": {
            "selection": selection_model,
            "rendering": rendering_model,
            "selection_prompt_fingerprint": skeleton.selection_prompt_fingerprint.fingerprint,
            "rendering_prompt_fingerprint": rendering_prompt_fingerprint.fingerprint,
        },
        "reference_repairs": [], "acknowledged_risk_ids": [],
    }
    if rendering is not None:
        _merge_rendering(payload, rendering)
    payload["quality_metrics"] = _quality_metrics(payload).model_dump(mode="json")
    payload["diagnostics_summary"] = {
        "selection_status": skeleton.synthesis_selection_status,
        "rendering_status": rendering_status,
        "skeleton_hash_verified": True,
    }
    result = FinalStrategyResult.model_validate(payload)
    validate_skeleton_integrity(skeleton)
    _quality_gate(result)
    return result


def freeze_fallback_result(
    result: FinalStrategyResult,
    *,
    selection_prompt_fingerprint: PromptFingerprint,
    warning: str,
) -> StrategySkeleton:
    """Freeze an already validated deterministic fallback for optional Call 2."""

    material = {
        "contract_family": "strategy_skeleton",
        "schema_version": "2.0",
        "skeleton_schema_version": "2.0",
        "competition_id": result.competition_id,
        "task_type": result.task_type or "unknown",
        "metric": result.metric,
        "recommended_validation": result.recommended_validation or "custom_required",
        "synthesis_selection_status": "degraded_fallback",
        "evidence_catalog": {
            key: value.model_dump(mode="json") for key, value in result.evidence_catalog.items()
        },
        "source_to_hypothesis_links": [item.model_dump(mode="json") for item in result.source_to_hypothesis_links],
        "hypothesis_to_eda_links": [item.model_dump(mode="json") for item in result.hypothesis_to_eda_links],
        "actions": [item.model_dump(mode="json") for item in result.actions],
        "action_provenance": [item.model_dump(mode="json") for item in result.action_provenance],
        "feature_experiment_families": [item.model_dump(mode="json") for item in result.feature_experiment_families],
        "core_experiments": [item.model_dump(mode="json") for item in result.core_experiments],
        "experiment_backlog": [item.model_dump(mode="json") for item in result.experiment_backlog],
        "experiment_budget": result.experiment_budget.model_dump(mode="json"),
        "dependency_graph": result.dependency_graph or {
            item.experiment_id: list(item.dependencies) for item in result.experiments
        },
        "section_structure": [item.model_dump(mode="json") for item in result.sections],
        "validation_requirement_ids": list(map(str, result.selected_validation_requirement_ids)),
        "safety_constraint_ids": list(map(str, result.enforced_safety_constraint_ids)),
        "limitations": list(result.limitations),
        "warnings": _unique([*result.warnings, warning]),
        "selection_prompt_fingerprint": selection_prompt_fingerprint.model_dump(mode="json"),
        "client_key_map": {},
    }
    digest = _stable_hash(material)
    return StrategySkeleton.model_validate({
        **material, "skeleton_id": f"skeleton_{digest[:16]}", "skeleton_hash": digest,
    })


def validate_rendering_draft(
    rendering: StrategyRenderingDraft,
    skeleton: StrategySkeleton,
) -> None:
    validate_skeleton_integrity(skeleton)
    if rendering.skeleton_id != skeleton.skeleton_id or rendering.skeleton_hash != skeleton.skeleton_hash:
        raise StrategyBridgeError("rendering skeleton identity or hash changed")
    expected = {
        "action": [item["action_id"] for item in skeleton.actions],
        "experiment": [item["experiment_id"] for item in [*skeleton.core_experiments, *skeleton.experiment_backlog]],
        "family": [item["family_id"] for item in skeleton.feature_experiment_families],
        "section": [item["section_id"] for item in skeleton.section_structure],
    }
    actual = {
        "action": [item.action_id for item in rendering.action_wording],
        "experiment": [item.experiment_id for item in rendering.experiment_wording],
        "family": [item.family_id for item in rendering.family_wording],
        "section": [item.section_id for item in rendering.section_summaries],
    }
    for label in expected:
        if expected[label] != actual[label]:
            raise StrategyBridgeError(
                f"rendering {label} IDs differ from frozen skeleton: expected={expected[label]}, actual={actual[label]}"
            )
    for text in _rendering_text(rendering):
        if len(text) > 1200:
            raise StrategyBridgeError("rendering text exceeds 1200 characters")
    frozen_text = json.dumps(skeleton.model_dump(mode="json"), ensure_ascii=False).casefold()
    allowed_numbers = set(re.findall(r"(?<![a-z0-9_])\d+(?:\.\d+)?", frozen_text))
    rendered_text = " ".join(_rendering_text(rendering)).casefold()
    introduced_numbers = set(
        re.findall(r"(?<![a-z0-9_])\d+(?:\.\d+)?", rendered_text)
    ) - allowed_numbers
    if introduced_numbers:
        raise StrategyBridgeError(
            f"rendering introduced unsupported numeric claims: {sorted(introduced_numbers)}"
        )
    for identity in model_registry():
        if (
            identity.display_name.casefold() in rendered_text
            and identity.canonical_family_id.casefold() not in frozen_text
        ):
            raise StrategyBridgeError(
                f"rendering introduced an unfrozen model: {identity.display_name}"
            )
    threshold_ids = {
        item["experiment_id"] for item in [*skeleton.core_experiments, *skeleton.experiment_backlog]
        if "threshold" in (item["name"] + " " + item["experiment_id"]).casefold()
    }
    for item in rendering.experiment_wording:
        if item.experiment_id in threshold_ids:
            text = " ".join((item.display_name, item.display_hypothesis, item.display_exact_change, item.display_acceptance_rule)).casefold()
            if "oof" not in text or "test label" not in text:
                raise StrategyBridgeError("threshold rendering must state OOF-only use and exclude test labels")


def validate_skeleton_integrity(skeleton: StrategySkeleton) -> None:
    material = skeleton.model_dump(mode="json")
    skeleton_id = str(material.pop("skeleton_id"))
    skeleton_hash = str(material.pop("skeleton_hash"))
    expected_hash = _stable_hash(material)
    expected_id = f"skeleton_{expected_hash[:16]}"
    if skeleton_hash != expected_hash or skeleton_id != expected_id:
        raise StrategyBridgeError(
            "frozen strategy skeleton content no longer matches its identity and hash"
        )


def _merge_rendering(payload: dict[str, Any], rendering: StrategyRenderingDraft) -> None:
    actions = {item.action_id: item for item in rendering.action_wording}
    for action in payload["actions"]:
        wording = actions[action["action_id"]]
        action["action"] = wording.display_action
        action["reason"] = wording.display_reason
    experiments = {item.experiment_id: item for item in rendering.experiment_wording}
    for experiment in payload["experiments"]:
        wording = experiments[experiment["experiment_id"]]
        experiment.update({
            "name": wording.display_name,
            "hypothesis": wording.display_hypothesis,
            "change": wording.display_exact_change,
            "acceptance_rule": wording.display_acceptance_rule,
            "risks": [wording.display_risk],
        })
    by_id = {item["experiment_id"]: item for item in payload["experiments"]}
    payload["core_experiments"] = [by_id[item["experiment_id"]] for item in payload["core_experiments"]]
    payload["experiment_backlog"] = [by_id[item["experiment_id"]] for item in payload["experiment_backlog"]]
    families = {item.family_id: item for item in rendering.family_wording}
    for family in payload["feature_experiment_families"]:
        wording = families[family["family_id"]]
        family.update({"name": wording.display_name, "hypothesis": wording.display_hypothesis,
                       "acceptance_rule": wording.display_acceptance_rule, "risks": wording.display_risks})
    sections = {item.section_id: item.summary for item in rendering.section_summaries}
    sections["executive_summary"] = rendering.executive_summary
    for section in payload["sections"]:
        section["summary"] = sections[section["section_id"]]
    payload["limitations"] = _unique([*payload["limitations"], *rendering.limitation_wording, rendering.uncertainty_summary])


def _compile_sections(draft: StrategySelectionDraft, actions: list[FinalStrategyAction], action_map: Mapping[str, str | None], experiment_map: Mapping[str, str], family_map: Mapping[str, str], core: list[FinalStrategyExperiment], budget: ExperimentBudget) -> list[FinalStrategySection]:
    by_id = {item.action_id: item for item in actions}
    sections = []
    for plan in draft.section_plan:
        action_ids = [action_map[key] for key in plan.selected_action_keys if action_map.get(key) in by_id]
        selected_actions = [by_id[value] for value in action_ids]
        limitations = [] if action_ids else ["No separate action was selected for this section."]
        section = FinalStrategySection(
            section_id=plan.section_id,
            title=plan.section_id.replace("_", " ").title(),
            summary=plan.summary_intent,
            action_ids=action_ids,
            evidence_refs=_unique(ref for action in selected_actions for ref in action.evidence_refs),
            related_hypothesis_ids=_unique(ref for action in selected_actions for ref in action.hypothesis_ids),
            source_refs=_unique(ref for action in selected_actions for ref in action.source_refs),
            eda_result_refs=_unique(ref for action in selected_actions for ref in action.eda_result_refs),
            availability="available" if action_ids else "limited",
            limitations=limitations,
        )
        sections.append(section)
    anchor = actions[0].action_id if actions else None
    if anchor is None:
        raise StrategyBridgeError("selection must retain at least one action")
    scheduled = [item.experiment_id for item in core][
        : budget.max_first_48h_experiments
    ]
    windows = ["0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours"]
    buckets = [scheduled[:1], scheduled[1:3], scheduled[3:4], scheduled[4:]]
    first = sections[-1].model_dump(mode="json")
    first["time_blocks"] = [
        {"time_window": window, "summary": "Execute the dependency-ordered frozen plan.",
         "action_ids": [anchor] if not bucket else [], "experiment_ids": bucket}
        for window, bucket in zip(windows, buckets)
    ]
    sections[-1] = FinalStrategySection.model_validate(first)
    return sections


def _compile_provenance(
    context: FinalSynthesisContext,
    actions: list[FinalStrategyAction],
    families: list[FeatureExperimentFamily],
    experiments: list[FinalStrategyExperiment],
    allowed: Mapping[str, Any],
) -> tuple[list[SourceToHypothesisLink], list[HypothesisToEdaLink]]:
    source_links = []
    eda_links = []
    used_hypotheses = {
        str(hypothesis)
        for item in [*actions, *families, *experiments]
        for hypothesis in (
            item.hypothesis_ids
            if isinstance(item, FinalStrategyAction)
            else item.motivating_hypothesis_ids
            if isinstance(item, FeatureExperimentFamily)
            else item.related_hypothesis_ids
        )
    }
    for hypothesis_id in sorted(used_hypotheses):
        catalog = allowed["hypothesis_catalog"].get(hypothesis_id, {})
        for source_ref in catalog.get("source_refs") or []:
            if source_ref in allowed["source"]:
                source_links.append(SourceToHypothesisLink(
                    source_ref=source_ref, hypothesis_id=hypothesis_id,
                    relationship="supports", claim_summary=catalog.get("statement") or "Validated research source.",
                ))
        result = next((item for item in context.eda_evidence_pack.hypothesis_results if str(item.hypothesis_id) == hypothesis_id), None)
        if result:
            for ref in result.evidence_refs:
                if str(ref) in allowed["evidence"]:
                    status = str(result.status)
                    if status not in {"confirmed", "partially_confirmed", "rejected", "not_testable", "skipped"}:
                        status = "confirmed"
                    eda_links.append(HypothesisToEdaLink(
                        hypothesis_id=hypothesis_id, eda_result_ref=str(ref),
                        result_status=status, finding_summary=str(result.finding),
                        confidence=str(result.confidence_after_eda),
                    ))
    return source_links, eda_links


def _compile_evidence_catalog(context: FinalStrategySelectionContext, refs: set[str]) -> dict[str, EvidenceCatalogEntry]:
    catalog = {item["evidence_ref"]: item for item in context.evidence_catalog}
    result = {}
    for ref in sorted(refs):
        item = catalog.get(ref)
        if item is None:
            raise StrategyBridgeError(f"used evidence ref is absent from compact catalog: {ref}")
        result[ref] = EvidenceCatalogEntry(
            ref=ref, resolved_value_preview=item["value_preview"],
            value_type=item["value_type"], source_component=ref.split(".", 1)[0].split("[", 1)[0],
            specificity=item["specificity"], available=True,
        )
    return result


def _validate_selection_references(draft: StrategySelectionDraft, allowed: Mapping[str, Any]) -> None:
    issues = []
    for path, item in _iter_grounded_items(draft):
        checks = {
            "primary_evidence_refs": allowed["evidence"], "supporting_evidence_refs": allowed["evidence"],
            "limitation_evidence_refs": allowed["evidence"], "source_refs": allowed["source"],
            "motivating_hypothesis_ids": allowed["hypothesis"], "safety_hypothesis_ids": allowed["hypothesis"],
            "validation_context_ids": allowed["hypothesis"], "rejected_hypothesis_ids": allowed["hypothesis"],
            "safety_constraint_ids": allowed["safety"], "validation_requirement_ids": allowed["validation"],
            "approved_experiment_ids": allowed["approved_experiment"],
        }
        for field, values in checks.items():
            unknown = set(getattr(item, field, []) or []) - set(values)
            if unknown:
                issues.append(f"{path}.{field}: {sorted(unknown)}")
        sources = getattr(item, "source_refs", []) or []
        hypotheses = getattr(item, "motivating_hypothesis_ids", []) or []
        for source in sources:
            if not any(source in allowed["hypothesis_catalog"].get(hypothesis, {}).get("source_refs", []) for hypothesis in hypotheses):
                issues.append(f"{path}.source_refs: {source!r} is not owned by a motivating hypothesis")
    if issues:
        raise StrategyBridgeError("unknown or broken selection references: " + "; ".join(issues))


def _validate_client_key_references(draft: StrategySelectionDraft) -> None:
    action_keys = {item.client_action_key for item in draft.selected_actions}
    family_keys = {
        item.client_family_key for item in draft.feature_experiment_families
    }
    experiment_keys = {
        item.client_experiment_key for item in draft.candidate_experiments
    }
    issues: list[str] = []
    for item in draft.selected_actions:
        unknown = set(item.dependencies) - action_keys
        if unknown:
            issues.append(
                f"action {item.client_action_key!r} dependencies={sorted(unknown)}"
            )
    for family in draft.feature_experiment_families:
        arm_keys = {
            family.baseline_arm.client_arm_key,
            *(item.client_arm_key for item in family.candidate_arms),
        }
        for arm in [family.baseline_arm, *family.candidate_arms]:
            unknown = set(arm.dependencies) - arm_keys - _DEPENDENCY_SENTINELS
            if unknown:
                issues.append(
                    f"family {family.client_family_key!r} arm "
                    f"{arm.client_arm_key!r} dependencies={sorted(unknown)}"
                )
    for item in draft.candidate_experiments:
        if item.family_key is not None and item.family_key not in family_keys:
            issues.append(
                f"experiment {item.client_experiment_key!r} family_key="
                f"{item.family_key!r}"
            )
        unknown = set(item.dependencies) - experiment_keys - _DEPENDENCY_SENTINELS
        if unknown:
            issues.append(
                f"experiment {item.client_experiment_key!r} "
                f"dependencies={sorted(unknown)}"
            )
    for section in draft.section_plan:
        unknown_actions = set(section.selected_action_keys) - action_keys
        unknown_families = set(section.selected_family_keys) - family_keys
        unknown_experiments = set(section.selected_experiment_keys) - experiment_keys
        if unknown_actions or unknown_families or unknown_experiments:
            issues.append(
                f"section {section.section_id!r} has unknown client keys: "
                f"actions={sorted(unknown_actions)}, "
                f"families={sorted(unknown_families)}, "
                f"experiments={sorted(unknown_experiments)}"
            )
    if issues:
        raise StrategyBridgeError(
            "unknown selection client-key references: " + "; ".join(issues)
        )


def _validate_required_execution_steps(draft: StrategySelectionDraft, experiments: list[FinalStrategyExperiment], context: FinalStrategySelectionContext) -> None:
    baseline = context.baseline_summary
    completed = str(baseline.get("status") or "").casefold() in {"completed", "complete", "success", "successful", "succeeded"}
    if completed and not any(_is_baseline(item.name) for item in experiments):
        raise StrategyBridgeError("completed baseline exists but baseline reproduction experiment is missing")
    requires_threshold = context.metric_contract.get("requires_threshold") is True or context.metric_contract.get("threshold_search_needed") is True
    threshold_drafts = [item for item in draft.candidate_experiments if _is_threshold(item.experiment_kind)]
    if threshold_drafts and not requires_threshold:
        raise StrategyBridgeError("threshold experiment is not allowed for this metric")
    for item in threshold_drafts:
        required = {"provisional_model_selected", "oof_predictions_available"}
        if not required <= set(item.dependencies):
            raise StrategyBridgeError("threshold experiment must depend on provisional_model_selected and oof_predictions_available")


def _canonicalize_dependencies(experiments: list[FinalStrategyExperiment], key_map: Mapping[str, str]) -> list[FinalStrategyExperiment]:
    return [item.model_copy(update={"dependencies": _unique(key_map.get(dep, dep) for dep in item.dependencies)}) for item in experiments]


def _validate_dependency_graph(experiments: list[FinalStrategyExperiment]) -> None:
    ids = {item.experiment_id for item in experiments}
    graph = {item.experiment_id: [dep for dep in item.dependencies if dep in ids] for item in experiments}
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(node: str) -> None:
        if node in visiting:
            raise StrategyBridgeError("experiment dependency cycle detected")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph[node]: visit(dep)
        visiting.remove(node); visited.add(node)
    for node in sorted(graph): visit(node)


def _quality_gate(result: FinalStrategyResult) -> None:
    if result.skeleton_hash is None or len(result.skeleton_hash) != 64:
        raise StrategyBridgeError("canonical result lacks frozen skeleton hash")
    core = {item.experiment_id for item in result.core_experiments}
    backlog = {item.experiment_id for item in result.experiment_backlog}
    if core & backlog:
        raise StrategyBridgeError("core and backlog overlap")
    if len(core) > result.experiment_budget.max_core_experiments:
        raise StrategyBridgeError("core budget exceeded")
    if result.quality_metrics.first_48h_experiment_count > result.experiment_budget.max_first_48h_experiments:
        raise StrategyBridgeError("first-48-hour budget exceeded")


def _quality_metrics(payload: Mapping[str, Any]) -> FinalStrategyQualityMetrics:
    actions = payload["actions"]
    refs = [len(item.get("evidence_refs") or []) for item in actions]
    first = next(item for item in payload["sections"] if item["section_id"] == "first_48_hours")
    role_counts = Counter(field for item in actions for field in (
        "motivating_hypothesis_ids", "safety_hypothesis_ids", "validation_context_ids", "rejected_hypothesis_ids"
    ) for _ in item.get(field) or [])
    return FinalStrategyQualityMetrics(
        action_count=len(actions), feature_family_count=len(payload["feature_experiment_families"]),
        core_experiment_count=len(payload["core_experiments"]), backlog_experiment_count=len(payload["experiment_backlog"]),
        average_evidence_refs_per_action=round(sum(refs) / len(refs), 3) if refs else 0.0,
        max_evidence_refs_per_action=max(refs, default=0), actions_exceeding_evidence_limits=sum(value > 6 for value in refs),
        actions_with_source_refs=sum(bool(item.get("source_refs")) for item in actions),
        actions_without_source_refs=sum(not item.get("source_refs") for item in actions),
        hypothesis_role_counts=dict(role_counts), first_48h_experiment_count=sum(len(item.get("experiment_ids") or []) for item in first.get("time_blocks") or []),
    )


def _allowed_catalogs(context: FinalStrategySelectionContext) -> dict[str, Any]:
    hypotheses = {item["hypothesis_id"]: item for item in context.hypothesis_catalog}
    return {
        "evidence": {item["evidence_ref"] for item in context.evidence_catalog},
        "source": {item["source_ref"] for item in context.source_catalog},
        "hypothesis": set(hypotheses), "hypothesis_catalog": hypotheses,
        "model": {item["canonical_family_id"] for item in context.model_catalog},
        "safety": {item.get("safety_constraint_id") for item in context.safety_constraint_catalog},
        "validation": {item.get("validation_requirement_id") for item in context.validation_requirement_catalog},
        "approved_experiment": {
            item["experiment_id"] for item in context.approved_experiment_catalog
        },
    }


def _iter_grounded_items(draft: StrategySelectionDraft):
    for index, item in enumerate(draft.selected_actions): yield f"selected_actions.{index}", item
    for index, item in enumerate(draft.feature_experiment_families): yield f"feature_experiment_families.{index}", item
    for index, item in enumerate(draft.candidate_experiments): yield f"candidate_experiments.{index}", item


def _compile_arm(arm: Any, family_id: str) -> ExperimentArm:
    return ExperimentArm(
        arm_id=_stable_id("arm", _semantic_key(family_id, arm.client_arm_key)),
        name=arm.name, exact_change=arm.exact_change,
        generated_features=arm.generated_features, fit_scope=arm.fit_scope,
        leakage_risk=arm.leakage_risk,
        dependencies=[
            dependency
            if dependency in _DEPENDENCY_SENTINELS
            else _stable_id("arm", _semantic_key(family_id, dependency))
            for dependency in arm.dependencies
        ],
    )


def _minimal_evidence(primary: Iterable[str], supporting: Iterable[str], limitations: Iterable[str]) -> tuple[list[str], int]:
    before = _unique([*primary, *supporting, *limitations])
    selected = _unique([*list(primary)[:1], *list(supporting)[:3], *list(limitations)[:2]])
    refined = []
    for ref in selected:
        if "." not in ref and any(other.startswith(ref + ".") for other in selected):
            continue
        refined.append(ref)
    return refined, len(before) - len(refined)


def _validated_action_sources(sources: Iterable[str], hypotheses: Iterable[str], allowed: Mapping[str, Any]) -> list[str]:
    return _unique(source for source in sources if source in allowed["source"] and any(source in allowed["hypothesis_catalog"].get(hypothesis, {}).get("source_refs", []) for hypothesis in hypotheses))


def _compile_evidence_origin(kind: str, has_source: bool) -> str:
    return _evidence_origin(kind, has_source)


def _evidence_origin(kind: str, has_source: bool) -> str:
    if "safety" in kind or "leakage" in kind: return "Safety-warning"
    if has_source: return "Source-supported"
    if "hypothesis" in kind or "experiment" in kind or "feature" in kind: return "Hypothesis-to-test"
    return "EDA-confirmed"


def _canonical_kind(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    aliases = {"baseline": "baseline_reproduction", "threshold": "threshold_postprocessing", "validation": "validation_setup"}
    return aliases.get(key, key or "general")


def _validation_method(context: FinalStrategySelectionContext) -> str:
    primary = context.validation_contract.get("primary_validation") or context.validation_contract.get("recommended_validation_candidate") or {}
    method = primary.get("method") if isinstance(primary, Mapping) else primary
    method = str(method or "stratified_kfold")
    allowed = {"stratified_kfold", "kfold", "group_kfold", "stratified_group_kfold", "temporal_holdout", "temporal_cv", "ranking_group_cv", "custom_required"}
    if method not in allowed: raise StrategyBridgeError(f"unsupported primary validation method: {method}")
    return method


def _schema_columns(schema: Mapping[str, Any]) -> set[str]:
    columns = set()
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if "name" in value: columns.add(str(value["name"]).casefold())
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(schema)
    return columns


def _used_evidence(actions: Iterable[FinalStrategyAction], families: Iterable[FeatureExperimentFamily], experiments: Iterable[FinalStrategyExperiment]) -> set[str]:
    return {str(ref) for item in [*actions, *families, *experiments] for ref in item.evidence_refs}


def _default_model(task_type: str):
    values = supported_models(task_type)
    return values[0] if values else None


def _experiment_rank(item: FinalStrategyExperiment) -> tuple[int, int, int, str]:
    kind = next((name for name in _KIND_ORDER if name in (item.name + " " + item.change).casefold().replace(" ", "_")), "feature_family")
    return (_KIND_ORDER.get(kind, 50), {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[item.priority], _COST[item.estimated_cost], item.experiment_id)


def _force_baseline_first(items: list[FinalStrategyExperiment]) -> list[FinalStrategyExperiment]:
    baseline = [item for item in items if _is_baseline(item.name)]
    return [*baseline, *(item for item in items if item not in baseline)]


def _force_threshold_late(items: list[FinalStrategyExperiment]) -> list[FinalStrategyExperiment]:
    threshold = [item for item in items if _is_threshold(item.name)]
    submission = [item for item in items if "submission" in item.name.casefold()]
    middle = [item for item in items if item not in threshold and item not in submission]
    return [*middle, *threshold, *submission]


def _is_baseline(value: str) -> bool: return "baseline" in value.casefold() and ("repro" in value.casefold() or "recorded" in value.casefold())
def _is_threshold(value: str) -> bool: return "threshold" in value.casefold()
def _semantic_key(*values: str) -> str: return "|".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip() for value in values)
def _stable_id(prefix: str, key: str) -> str: return f"{prefix}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"
def _stable_hash(value: Any) -> str: return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
def _unique(values: Iterable[Any]) -> list[Any]: return list(dict.fromkeys(value for value in values if value not in (None, "")))
def experiment_key_map_inv(mapping: Mapping[str, str], value: str) -> str: return next((key for key, mapped in mapping.items() if mapped == value), value)
def _rendering_text(value: StrategyRenderingDraft) -> list[str]:
    result = [value.executive_summary, value.uncertainty_summary, *value.limitation_wording]
    for item in value.action_wording: result.extend([item.display_action, item.display_reason])
    for item in value.experiment_wording: result.extend([item.display_name, item.display_hypothesis, item.display_exact_change, item.display_acceptance_rule, item.display_risk])
    for item in value.family_wording: result.extend([item.display_name, item.display_hypothesis, item.display_acceptance_rule, *item.display_risks])
    result.extend(item.summary for item in value.section_summaries)
    return result


__all__ = [
    "BRIDGE_POLICY_VERSION", "StrategyBridgeError", "freeze_strategy_selection",
    "freeze_fallback_result", "skeleton_to_result", "validate_rendering_draft",
    "validate_skeleton_integrity",
]
