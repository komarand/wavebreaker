from __future__ import annotations

import json
import os
import re
from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Mapping

from kaggle_researcher.contracts.evidence import (
    EvidencePathResolutionError,
    resolve_evidence_ref,
)
from kaggle_researcher.contracts.final_strategy import (
    ActionProvenance,
    EvidenceCatalogEntry,
    ExperimentArm,
    ExperimentBudget,
    FeatureExperimentFamily,
    FinalStrategyExperiment,
    FinalStrategyQualityMetrics,
    FinalStrategyResult,
)
from kaggle_researcher.contracts.final_strategy_evidence import bounded_evidence_preview
from kaggle_researcher.reasoning.model_registry import (
    distinct_candidate,
    is_valid_model_comparison,
    resolve_model_identity,
    supported_models,
)


_SUCCESS = {"completed", "complete", "success", "successful", "succeeded"}
_COST_VALUE = {"low": 1.0, "medium": 2.0, "high": 4.0}
_GROUPED_MARKERS = {
    "missingness_age": ("age_imputation", "missing_indicator_age"),
    "relationship_family": ("family_size", "is_alone"),
    "cabin_representation": ("cabin",),
    "name_representation": ("title", "name"),
    "ticket_representation": ("ticket",),
    "fare_representation": ("fare",),
}


def compact_final_strategy(
    result: FinalStrategyResult,
    *,
    evidence_pack: Mapping[str, Any],
    source_ids: Iterable[str] = (),
    budget: ExperimentBudget | None = None,
) -> FinalStrategyResult:
    """Apply the production v2 strategy policy once, deterministically."""

    payload = deepcopy(result.model_dump(mode="json"))
    payload["schema_version"] = "2.0"
    configured_budget = budget or _budget_from_environment()
    category_by_hypothesis = _hypothesis_categories(evidence_pack, result)
    source_links = {
        (str(link.source_ref), str(link.hypothesis_id))
        for link in result.source_to_hypothesis_links
    }
    allowed_sources = set(source_ids)
    unsupported_removed = 0
    root_specific = 0
    overflow_count = 0

    compact_actions: list[dict[str, Any]] = []
    for raw in payload.get("actions") or []:
        if raw.get("experiment_id"):
            # Structured experiments are emitted once in core/backlog, not duplicated
            # as verbose action records.
            continue
        action = dict(raw)
        kind = _action_kind(action)
        if kind == "threshold_postprocessing":
            # The threshold recommendation is represented once as the dependency-
            # ordered OOF postprocessing experiment near the end of the plan.
            continue
        action["action_kind"] = kind
        selected, removed, roots = _select_action_evidence(action, evidence_pack, kind)
        unsupported_removed += removed
        root_specific += roots
        action["evidence_refs"] = selected
        action["primary_evidence_refs"] = selected[:1]
        action["limitation_evidence_refs"] = [
            ref for ref in action.get("limitation_evidence_refs") or [] if ref in selected
        ][:2]
        action["eda_result_refs"] = [
            ref for ref in action.get("eda_result_refs") or [] if ref in selected
        ]
        if len(selected) > 4 and not action.get("evidence_overflow_reason"):
            overflow_count += 1
        _assign_hypothesis_roles(action, kind, category_by_hypothesis)
        selected_hypotheses = set(action["hypothesis_ids"])
        action["source_refs"] = sorted({
            ref
            for ref in action.get("source_refs") or []
            if ref in allowed_sources
            and any((ref, hypothesis_id) in source_links for hypothesis_id in selected_hypotheses)
        })
        compact_actions.append(action)
    payload["actions"] = compact_actions

    families = _build_feature_families(
        evidence_pack,
        validation=str(result.recommended_validation or ""),
        metric=_metric_name(result.metric, evidence_pack),
        category_by_hypothesis=category_by_hypothesis,
    )
    grouped_markers = tuple(
        marker for family in families for marker in _GROUPED_MARKERS.get(family.family_id, ())
    )
    experiments, self_comparisons_removed = _compact_experiments(
        result,
        evidence_pack=evidence_pack,
        families=families,
        grouped_markers=grouped_markers,
        category_by_hypothesis=category_by_hypothesis,
    )
    core, backlog = _apply_budget(experiments, configured_budget)
    payload["feature_experiment_families"] = [item.model_dump(mode="json") for item in families]
    payload["core_experiments"] = [item.model_dump(mode="json") for item in core]
    payload["experiment_backlog"] = [item.model_dump(mode="json") for item in backlog]
    payload["experiments"] = [
        item.model_dump(mode="json") for item in [*core, *backlog]
    ]
    payload["experiment_budget"] = configured_budget.model_copy(update={
        "estimated_total_cost": sum(
            _COST_VALUE[item.estimated_cost] for item in [*core, *backlog]
        ),
    }).model_dump(mode="json")

    _clean_sections(payload)
    first_48_count = _schedule_first_48_hours(payload, core, configured_budget)
    _write_executive_summary(payload, evidence_pack, len(core), len(backlog))
    _refresh_provenance(payload)

    catalog, duplicate_bytes = _build_evidence_catalog(payload, evidence_pack)
    payload["evidence_catalog"] = {
        ref: entry.model_dump(mode="json") for ref, entry in catalog.items()
    }
    refs_per_action = [len(action.get("evidence_refs") or []) for action in compact_actions]
    role_counts = Counter(
        field
        for action in compact_actions
        for field in (
            "motivating_hypothesis_ids",
            "safety_hypothesis_ids",
            "validation_context_ids",
            "rejected_hypothesis_ids",
        )
        for _ in action.get(field) or []
    )
    quality = FinalStrategyQualityMetrics(
        action_count=len(compact_actions),
        feature_family_count=len(families),
        core_experiment_count=len(core),
        backlog_experiment_count=len(backlog),
        average_evidence_refs_per_action=(
            round(sum(refs_per_action) / len(refs_per_action), 3) if refs_per_action else 0.0
        ),
        max_evidence_refs_per_action=max(refs_per_action, default=0),
        root_refs_for_specific_claims=root_specific,
        unsupported_evidence_refs_removed=unsupported_removed,
        actions_exceeding_evidence_limits=overflow_count,
        unresolved_refs=sum(not entry.available for entry in catalog.values()),
        actions_with_source_refs=sum(bool(action.get("source_refs")) for action in compact_actions),
        actions_without_source_refs=sum(not action.get("source_refs") for action in compact_actions),
        hypothesis_role_counts=dict(sorted(role_counts.items())),
        duplicate_preview_bytes_avoided=duplicate_bytes,
        model_self_comparisons_removed=self_comparisons_removed,
        first_48h_experiment_count=first_48_count,
    )
    payload["quality_metrics"] = quality.model_dump(mode="json")
    payload["diagnostics_summary"] = {
        "quality_gate": "passed",
        "quality_metrics": quality.model_dump(mode="json"),
        "provenance": {
            "source_hypotheses_total": len(result.source_to_hypothesis_links),
            "hypotheses_with_source_refs": len({
                str(link.hypothesis_id) for link in result.source_to_hypothesis_links
            }),
            "actions_with_source_provenance": quality.actions_with_source_refs,
            "eda_only_actions": quality.actions_without_source_refs,
            "missing_source_warnings": sum(
                not hypothesis_id_has_source
                for hypothesis_id_has_source in _hypothesis_source_flags(result)
            ),
        },
    }
    _append_provenance_limitation(payload, result)
    compacted = FinalStrategyResult.model_validate(payload)
    validate_compacted_strategy(compacted)
    return compacted


def validate_compacted_strategy(result: FinalStrategyResult) -> None:
    """Final deterministic quality gate beyond the Pydantic shape contract."""

    catalog = set(result.evidence_catalog)
    unresolved = sorted({
        ref for action in result.actions for ref in action.evidence_refs if ref not in catalog
    })
    if unresolved:
        raise ValueError(f"Unresolved compact strategy evidence refs: {unresolved}")
    if result.quality_metrics.root_refs_for_specific_claims:
        raise ValueError("Specific actions retain root-level evidence refs")
    if result.quality_metrics.actions_exceeding_evidence_limits:
        raise ValueError("Actions exceed the evidence-reference policy")
    core_ids = [item.experiment_id for item in result.core_experiments]
    if len(core_ids) > result.experiment_budget.max_core_experiments:
        raise ValueError("Core experiment budget exceeded")
    if result.quality_metrics.first_48h_experiment_count > result.experiment_budget.max_first_48h_experiments:
        raise ValueError("First-48-hours experiment budget exceeded")
    baseline_index = next((i for i, item in enumerate(result.core_experiments) if _is_baseline(item)), None)
    threshold_index = next((i for i, item in enumerate(result.core_experiments) if _is_threshold(item)), None)
    if baseline_index is not None and baseline_index != 0:
        raise ValueError("Baseline reproduction is not the first modeling experiment")
    if threshold_index is not None and baseline_index is not None and threshold_index <= baseline_index:
        raise ValueError("Threshold postprocessing precedes baseline reproduction")
    family_markers = tuple(
        marker
        for family in result.feature_experiment_families
        for marker in _GROUPED_MARKERS.get(family.family_id, ())
    )
    redundant = [
        item.experiment_id
        for item in result.experiments
        if not item.experiment_id.startswith("family_")
        and any(marker in item.experiment_id.casefold() for marker in family_markers)
    ]
    if redundant:
        raise ValueError(f"Ungrouped feature experiments remain: {redundant}")


def _select_action_evidence(
    action: Mapping[str, Any], evidence_pack: Mapping[str, Any], kind: str
) -> tuple[list[str], int, int]:
    refs = list(dict.fromkeys(str(ref) for ref in action.get("evidence_refs") or []))
    required = {
        "baseline_reproduction": ("baseline_evidence.status",),
        "threshold_postprocessing": ("metric_evidence.requires_threshold",),
        "primary_id_safety": ("inferred_schema.primary_id_column",),
        "validation_policy": ("validation_evidence.primary_validation",),
        "drift_risk": (
            _first_resolving(evidence_pack, (
                "drift_evidence.feature_drift_severity",
                "drift_evidence.overall_severity",
                "drift_evidence.severity",
            )),
        ),
    }.get(kind, ())
    for ref in required:
        if ref and _resolves(evidence_pack, ref) and ref not in refs:
            refs.insert(0, ref)
    text = " ".join((str(action.get("action") or ""), str(action.get("reason") or ""))).casefold()
    inputs = [str(value).casefold() for value in (action.get("feature_metadata") or {}).get("input_columns") or []]
    scored: list[tuple[int, int, str]] = []
    for index, ref in enumerate(refs):
        if not _resolves(evidence_pack, ref) and ref != "final_synthesizer.repaired":
            continue
        value = _resolve_optional(evidence_pack, ref)
        rendered = json.dumps(value, ensure_ascii=False, default=str).casefold()
        score = 100 if ref in required else 0
        score += 55 if any(column in ref.casefold() or column in rendered for column in inputs) else 0
        score += 25 if _specificity(ref, value) in {"leaf", "item"} else 0
        score += 15 if any(token in ref.casefold() for token in _action_tokens(text)) else 0
        score -= 40 if _specificity(ref, value) == "root" and kind != "broad_inspection" else 0
        scored.append((-score, index, ref))
    selected = [item[2] for item in sorted(scored)[:4]]
    selected.sort(key=lambda ref: (ref not in required, refs.index(ref)))
    if not selected and refs:
        selected = [refs[0]]
    roots = sum(
        _specificity(ref, _resolve_optional(evidence_pack, ref)) == "root"
        for ref in selected
        if kind != "broad_inspection"
    )
    # A root is retained only if no concrete child from the same component resolves.
    if roots:
        refined = []
        for ref in selected:
            if _specificity(ref, _resolve_optional(evidence_pack, ref)) != "root":
                refined.append(ref)
                continue
            child = _best_child_ref(evidence_pack, ref, text, inputs)
            if child:
                refined.append(child)
            elif kind == "broad_inspection":
                refined.append(ref)
        selected = list(dict.fromkeys(refined))
        roots = sum(
            _specificity(ref, _resolve_optional(evidence_pack, ref)) == "root"
            for ref in selected
            if kind != "broad_inspection"
        )
    return selected, max(0, len(refs) - len(selected)), roots


def _assign_hypothesis_roles(
    action: dict[str, Any], kind: str, categories: Mapping[str, str]
) -> None:
    existing = list(dict.fromkeys([
        *(action.get("hypothesis_ids") or []),
        *(action.get("related_hypothesis_ids") or []),
    ]))
    desired = {
        "baseline_reproduction": ({"baseline"}, set(), {"validation"}),
        "threshold_postprocessing": ({"metric"}, set(), {"validation"}),
        "primary_id_safety": (set(), {"schema", "leakage"}, set()),
        "drift_risk": ({"drift", "leaderboard"}, set(), set()),
        "validation_policy": (set(), set(), {"validation"}),
        "feature_family": ({"feature", "schema", "missingness", "relationship"}, {"leakage"}, {"validation"}),
    }.get(kind, (set(), set(), set()))
    motivating = _ids_in_categories(existing, categories, desired[0])
    safety = _ids_in_categories(existing, categories, desired[1])
    validation = _ids_in_categories(existing, categories, desired[2])
    if not motivating and not safety and not validation and existing:
        motivating = existing[:1]
    action["motivating_hypothesis_ids"] = motivating
    action["safety_hypothesis_ids"] = [item for item in safety if item not in motivating]
    action["validation_context_ids"] = [
        item for item in validation if item not in motivating and item not in safety
    ]
    action["rejected_hypothesis_ids"] = [
        item for item in action.get("rejected_hypothesis_ids") or [] if item in existing
    ]
    union = list(dict.fromkeys([
        *action["motivating_hypothesis_ids"],
        *action["safety_hypothesis_ids"],
        *action["validation_context_ids"],
        *action["rejected_hypothesis_ids"],
    ]))
    action["related_hypothesis_ids"] = union
    action["hypothesis_ids"] = union


def _build_feature_families(
    evidence_pack: Mapping[str, Any],
    *,
    validation: str,
    metric: str,
    category_by_hypothesis: Mapping[str, str],
) -> list[FeatureExperimentFamily]:
    if not validation:
        return []
    columns = _column_refs(evidence_pack)
    feature_ids = _ids_in_categories(
        list(category_by_hypothesis), category_by_hypothesis,
        {"feature", "schema", "missingness", "relationship"},
    )
    fallback_ids = list(category_by_hypothesis)[:1]
    motivating = feature_ids or fallback_ids
    families: list[FeatureExperimentFamily] = []

    def add(
        family_id: str,
        name: str,
        inputs: list[str],
        arms: list[tuple[str, str, str, list[str], str, str]],
        evidence_refs: list[str],
        *,
        cost: str = "low",
    ) -> None:
        if not motivating or not all(column.casefold() in columns for column in inputs):
            return
        refs = [ref for ref in dict.fromkeys(evidence_refs) if _resolves(evidence_pack, ref)][:4]
        if not refs:
            return
        baseline = ExperimentArm(
            arm_id=f"{family_id}_baseline",
            name="Current baseline representation",
            exact_change="Keep the recorded baseline handling unchanged.",
            generated_features=[],
            fit_scope="within_fold",
            leakage_risk="low",
        )
        candidate_arms = [
            ExperimentArm(
                arm_id=arm_id,
                name=arm_name,
                exact_change=change,
                generated_features=features,
                fit_scope=fit_scope,
                leakage_risk=risk,
                dependencies=["baseline_reproduced", "folds_locked"],
            )
            for arm_id, arm_name, change, features, fit_scope, risk in arms
        ]
        families.append(FeatureExperimentFamily(
            family_id=family_id,
            name=name,
            priority="P1",
            input_columns=inputs,
            hypothesis=f"One controlled {name.casefold()} arm improves paired OOF {metric} without leakage.",
            baseline_arm=baseline,
            candidate_arms=candidate_arms,
            validation_strategy=validation,
            metric=metric,
            fit_scope="within_fold" if any(arm.fit_scope == "within_fold" for arm in candidate_arms) else "per_row",
            evidence_refs=refs,
            motivating_hypothesis_ids=motivating[:2],
            risks=["Arms may be redundant or unstable across folds."],
            acceptance_rule=f"Adopt at most one arm after paired OOF {metric} improves in the registered direction and is fold-stable.",
            estimated_cost=cost,
            status="planned",
        ))

    age_missing = _column_diagnostic_ref(evidence_pack, "Age", "missingness_diagnostics")
    if age_missing:
        add(
            "missingness_age", "Age missingness and imputation", ["Age"],
            [
                ("age_imputation", "Fold-fitted imputation", "Fit the imputer on each training fold and transform validation/test.", ["Age_imputed"], "within_fold", "low"),
                ("age_imputation_indicator", "Imputation plus missing indicator", "Add a row-wise Age-missing flag to fold-fitted imputation.", ["Age_imputed", "Age_missing"], "within_fold", "low"),
            ],
            [columns.get("age", ""), age_missing],
        )
    if "sibsp" in columns and "parch" in columns:
        add(
            "relationship_family", "Family and relationship structure", ["SibSp", "Parch"],
            [
                ("family_size", "Family size", "Add 1 + SibSp + Parch.", ["family_size"], "per_row", "low"),
                ("family_size_is_alone", "Family size plus is-alone", "Add family_size and an is_alone indicator.", ["family_size", "is_alone"], "per_row", "low"),
            ],
            [columns["sibsp"], columns["parch"]],
        )
    if "cabin" in columns:
        cabin_diag = _all_column_diagnostic_refs(evidence_pack, "Cabin")
        add(
            "cabin_representation", "Cabin representation", ["Cabin"],
            [
                ("cabin_missing", "Missingness only", "Add only Cabin missingness.", ["cabin_missing"], "per_row", "low"),
                ("cabin_deck", "Missingness plus deck", "Extract deck with an unknown bucket and retain missingness.", ["cabin_missing", "cabin_deck"], "per_row", "low"),
                ("cabin_frequency", "Fold frequency", "Fit Cabin frequency within each training fold.", ["cabin_frequency"], "within_fold", "medium"),
                ("cabin_shape", "Structural text shape", "Add length/digit/punctuation summaries without token identity.", ["cabin_text_shape"], "per_row", "low"),
            ], [columns["cabin"], *cabin_diag], cost="medium",
        )
    if "name" in columns:
        add(
            "name_representation", "Name representation", ["Name"],
            [
                ("name_title", "Title", "Extract a normalized title with rare-title bucketing.", ["name_title"], "per_row", "low"),
                ("name_shape", "Structural text shape", "Add structural text summaries without name identity.", ["name_text_shape"], "per_row", "low"),
                ("name_frequency", "Fold frequency", "Fit exact-name frequency within each training fold.", ["name_frequency"], "within_fold", "medium"),
            ], [columns["name"], *_all_column_diagnostic_refs(evidence_pack, "Name")], cost="medium",
        )
    if "ticket" in columns:
        add(
            "ticket_representation", "Ticket representation", ["Ticket"],
            [
                ("ticket_group", "Ticket group size", "Add ticket group size without using target values.", ["ticket_group_size"], "within_fold", "medium"),
                ("ticket_frequency", "Fold frequency", "Fit Ticket frequency within each training fold.", ["ticket_frequency"], "within_fold", "medium"),
                ("ticket_shape", "Structural text shape", "Add length/digit/punctuation summaries.", ["ticket_text_shape"], "per_row", "low"),
            ], [columns["ticket"], *_all_column_diagnostic_refs(evidence_pack, "Ticket")], cost="medium",
        )
    if "fare" in columns:
        add(
            "fare_representation", "Fare representation", ["Fare"],
            [
                ("fare_log", "log1p Fare", "Add log1p(max(Fare, 0)).", ["fare_log1p"], "per_row", "low"),
                ("fare_group_normalized", "Within-group normalization", "Fit Fare normalization within each training fold and supported group.", ["fare_group_normalized"], "within_fold", "medium"),
            ], [columns["fare"], *_all_column_diagnostic_refs(evidence_pack, "Fare")],
        )
    return families


def _compact_experiments(
    result: FinalStrategyResult,
    *,
    evidence_pack: Mapping[str, Any],
    families: list[FeatureExperimentFamily],
    grouped_markers: tuple[str, ...],
    category_by_hypothesis: Mapping[str, str],
) -> tuple[list[FinalStrategyExperiment], int]:
    task_type = result.task_type or str((evidence_pack.get("metric_evidence") or {}).get("task_type") or "")
    baseline_raw = str((evidence_pack.get("baseline_evidence") or {}).get("model_type") or "")
    baseline = resolve_model_identity(baseline_raw)
    if baseline is None or task_type.casefold() not in baseline.task_types:
        baseline = next(iter(supported_models(task_type)), None)
    candidates: list[FinalStrategyExperiment] = []
    self_removed = 0
    for experiment in result.experiments:
        text = f"{experiment.experiment_id} {experiment.name} {experiment.change}".casefold()
        identity_text = f"{experiment.experiment_id} {experiment.name}".casefold()
        if any(marker in text for marker in grouped_markers):
            continue
        if _is_threshold(experiment) and not _threshold_required(evidence_pack):
            continue
        data = experiment.model_dump(mode="json")
        data["primary_evidence_refs"] = list(experiment.evidence_refs[:1])
        data["dependencies"] = _experiment_dependencies(experiment)
        data["estimated_cost"] = _estimated_cost(experiment)
        if baseline:
            data["model_family"] = baseline.display_name
            data["baseline_canonical_family_id"] = baseline.canonical_family_id
            data["baseline_implementation_id"] = baseline.implementation_id
        if "model" in identity_text and "comparison" in identity_text:
            candidate = resolve_model_identity(experiment.model_family)
            if candidate is None or not is_valid_model_comparison(baseline, candidate):
                self_removed += 1
                continue
            data["candidate_canonical_family_id"] = candidate.canonical_family_id
            data["candidate_implementation_id"] = candidate.implementation_id
        candidates.append(FinalStrategyExperiment.model_validate(data))

    hypothesis_ids = _ids_in_categories(
        list(category_by_hypothesis), category_by_hypothesis,
        {"feature", "schema", "baseline", "validation"},
    ) or list(category_by_hypothesis)[:1]
    baseline_ids = _ids_in_categories(
        list(category_by_hypothesis), category_by_hypothesis,
        {"baseline", "validation"},
    ) or hypothesis_ids
    baseline_evidence = evidence_pack.get("baseline_evidence") or {}
    if (
        baseline
        and baseline_ids
        and str(baseline_evidence.get("status") or "").casefold() in _SUCCESS
        and not any(_is_baseline(item) for item in candidates)
    ):
        refs = [ref for ref in (
            "baseline_evidence.status",
            "baseline_evidence.model_type",
            "baseline_evidence.metric_value",
            "validation_evidence.primary_validation",
        ) if _resolves(evidence_pack, ref)]
        candidates.append(FinalStrategyExperiment(
            experiment_id="baseline_reproduction",
            priority="P0",
            name="Reproduce the recorded baseline",
            hypothesis="The recorded baseline can be reproduced under the locked folds and preprocessing boundaries.",
            change="Re-run the recorded estimator, features, preprocessing, folds, and metric without optimization.",
            model_family=baseline.display_name,
            validation_strategy=str(result.recommended_validation),
            success_metric=_metric_name(result.metric, evidence_pack),
            acceptance_rule="Require the recorded OOF result within expected numeric tolerance and explain any fold-level discrepancy.",
            evidence_refs=refs,
            related_hypothesis_ids=baseline_ids[:2],
            risks=["Environment or preprocessing drift can prevent exact reproduction."],
            fit_scope="within_fold",
            baseline_canonical_family_id=baseline.canonical_family_id,
            baseline_implementation_id=baseline.implementation_id,
            status="required",
            estimated_cost="low",
            dependencies=["folds_locked", "schema_roles_locked"],
        ))
    for family in families:
        candidates.append(FinalStrategyExperiment(
            experiment_id=f"family_{family.family_id}",
            priority=family.priority,
            name=family.name,
            hypothesis=family.hypothesis,
            change="Run the baseline arm and candidate arms as one controlled multi-arm family.",
            feature_inputs=family.input_columns,
            model_family=baseline.display_name if baseline else "registered_baseline",
            validation_strategy=family.validation_strategy,
            success_metric=family.metric,
            acceptance_rule=family.acceptance_rule,
            evidence_refs=family.evidence_refs,
            related_hypothesis_ids=family.motivating_hypothesis_ids,
            risks=family.risks,
            fit_scope=family.fit_scope,
            baseline_canonical_family_id=baseline.canonical_family_id if baseline else None,
            baseline_implementation_id=baseline.implementation_id if baseline else None,
            estimated_cost=family.estimated_cost,
            dependencies=["baseline_reproduced", "folds_locked"],
        ))

    if baseline and hypothesis_ids and not any("model" in item.experiment_id.casefold() for item in candidates):
        candidate = distinct_candidate(baseline, task_type)
        if is_valid_model_comparison(baseline, candidate):
            validation_ref = "validation_evidence.primary_validation"
            evidence_refs = [ref for ref in (
                "baseline_evidence.model_type", validation_ref,
            ) if _resolves(evidence_pack, ref)]
            if evidence_refs:
                candidates.append(FinalStrategyExperiment(
                    experiment_id="model_family_comparison",
                    priority="P2",
                    name=f"{baseline.display_name} versus {candidate.display_name}",
                    hypothesis="A distinct supported model family may improve paired OOF performance.",
                    change="Change only the estimator family; preserve folds, features, preprocessing boundaries, and metric.",
                    model_family=candidate.display_name,
                    validation_strategy=str(result.recommended_validation),
                    success_metric=_metric_name(result.metric, evidence_pack),
                    acceptance_rule="Adopt only if paired OOF performance improves and remains stable across folds.",
                    evidence_refs=evidence_refs,
                    related_hypothesis_ids=hypothesis_ids[:2],
                    risks=["A different estimator may add variance without stable benefit."],
                    fit_scope="within_fold",
                    baseline_canonical_family_id=baseline.canonical_family_id,
                    baseline_implementation_id=baseline.implementation_id,
                    candidate_canonical_family_id=candidate.canonical_family_id,
                    candidate_implementation_id=candidate.implementation_id,
                    estimated_cost="medium",
                    dependencies=["baseline_reproduced", "folds_locked"],
                ))
        else:
            limitation = "No distinct supported model family is available for a meaningful comparison."
            if limitation not in result.limitations:
                result.limitations.append(limitation)

    if _threshold_required(evidence_pack) and hypothesis_ids and not any(_is_threshold(item) for item in candidates):
        refs = [ref for ref in (
            "metric_evidence.requires_threshold", "validation_evidence.primary_validation",
        ) if _resolves(evidence_pack, ref)]
        candidates.append(FinalStrategyExperiment(
            experiment_id="threshold_oof_postprocessing",
            priority="P2",
            name="OOF-only threshold postprocessing",
            hypothesis="An OOF-selected threshold may improve the threshold-dependent metric.",
            change="Compare the default threshold with candidates selected only from OOF predictions; report fold/seed stability.",
            model_family=baseline.display_name if baseline else "provisional_best_model",
            validation_strategy=str(result.recommended_validation),
            success_metric=_metric_name(result.metric, evidence_pack),
            acceptance_rule="Adopt only if OOF metric improves over the default threshold without test labels or in-sample predictions.",
            evidence_refs=refs,
            related_hypothesis_ids=hypothesis_ids[:2],
            risks=["Threshold gains may be fold-unstable or overfit."],
            fit_scope="oof_only",
            baseline_canonical_family_id=baseline.canonical_family_id if baseline else None,
            baseline_implementation_id=baseline.implementation_id if baseline else None,
            estimated_cost="low",
            dependencies=["baseline_reproduced", "folds_locked", "provisional_model_selected", "oof_predictions_available"],
        ))

    if hypothesis_ids and not any("submission" in item.experiment_id.casefold() for item in candidates):
        submission_refs = [ref for ref in (
            "inferred_schema.primary_id_column",
            "inferred_schema.prediction_column",
            "inferred_schema.sample_submission_table",
        ) if _resolves(evidence_pack, ref)]
        if len(submission_refs) >= 2:
            candidates.append(FinalStrategyExperiment(
                experiment_id="submission_integrity_check",
                priority="P0",
                name="Submission schema and row-integrity check",
                hypothesis="The frozen predictions can be exported without changing row identity or output schema.",
                change="Verify row count/order, primary IDs, prediction column, finite values, and sample-submission compatibility after all modeling decisions are frozen.",
                model_family=baseline.display_name if baseline else "provisional_best_model",
                validation_strategy=str(result.recommended_validation),
                success_metric=_metric_name(result.metric, evidence_pack),
                acceptance_rule="Export only when every submission integrity assertion passes.",
                evidence_refs=submission_refs,
                related_hypothesis_ids=hypothesis_ids[:2],
                risks=["Row-order or schema mismatch can invalidate an otherwise sound model."],
                fit_scope="not_applicable",
                baseline_canonical_family_id=baseline.canonical_family_id if baseline else None,
                baseline_implementation_id=baseline.implementation_id if baseline else None,
                estimated_cost="low",
                dependencies=[
                    "baseline_reproduced", "provisional_model_selected",
                    "postprocessing_frozen",
                ],
            ))

    # Keep one canonical experiment per semantic ID.
    unique: dict[str, FinalStrategyExperiment] = {}
    for item in candidates:
        unique.setdefault(item.experiment_id, item)
    return list(unique.values()), self_removed


def _apply_budget(
    experiments: list[FinalStrategyExperiment], budget: ExperimentBudget
) -> tuple[list[FinalStrategyExperiment], list[FinalStrategyExperiment]]:
    ranked = sorted(experiments, key=_experiment_rank)
    pinned_ids = {
        item.experiment_id
        for item in ranked
        if _is_baseline(item)
        or _is_threshold(item)
        or "submission" in item.experiment_id.casefold()
        or (
            item.candidate_canonical_family_id is not None
            and item.candidate_canonical_family_id != item.baseline_canonical_family_id
        )
    }
    fill_limit = max(0, budget.max_core_experiments - len(pinned_ids))
    fill_ids = {
        item.experiment_id
        for item in [
            candidate for candidate in ranked
            if candidate.experiment_id not in pinned_ids
        ][:fill_limit]
    }
    selected_ids = pinned_ids | fill_ids
    core: list[FinalStrategyExperiment] = []
    backlog: list[FinalStrategyExperiment] = []
    high_cost = 0
    for item in ranked:
        can_select = item.experiment_id in selected_ids and len(core) < budget.max_core_experiments
        if item.estimated_cost == "high":
            can_select = can_select and high_cost < budget.max_high_cost_experiments
        if can_select:
            status = "required" if _is_baseline(item) else "core"
            selected = item.model_copy(update={"status": status})
            core.append(selected)
            high_cost += item.estimated_cost == "high"
        else:
            backlog.append(item.model_copy(update={"status": "backlog"}))
    # Threshold is downstream and submission integrity (if present) is last.
    core.sort(key=_experiment_rank)
    backlog.sort(key=_experiment_rank)
    return core, backlog


def _schedule_first_48_hours(
    payload: dict[str, Any],
    core: list[FinalStrategyExperiment],
    budget: ExperimentBudget,
) -> int:
    section = next(
        (item for item in payload.get("sections") or [] if item.get("section_id") == "first_48_hours"),
        None,
    )
    if section is None:
        return 0
    action_ids = [
        action["action_id"] for action in payload.get("actions") or [] if action.get("action_id")
    ]
    anchor = action_ids[:1]
    scheduled = core[: budget.max_first_48h_experiments]
    baseline = [item.experiment_id for item in scheduled if _is_baseline(item)]
    remaining = [item.experiment_id for item in scheduled if not _is_baseline(item)]
    first_24 = remaining[: max(0, budget.max_first_24h_experiments - len(baseline))]
    later = remaining[len(first_24):]
    section["time_blocks"] = [
        {
            "time_window": "0-4_hours",
            "summary": "Lock metric, folds, schema roles, and safety checks.",
            "action_ids": anchor,
            "experiment_ids": [],
        },
        {
            "time_window": "4-12_hours",
            "summary": "Reproduce the recorded baseline under the locked validation policy.",
            "action_ids": [] if baseline else anchor,
            "experiment_ids": baseline,
        },
        {
            "time_window": "12-24_hours",
            "summary": "Run the highest-value stable ablation and low-cost feature families.",
            "action_ids": [] if first_24 else anchor,
            "experiment_ids": first_24,
        },
        {
            "time_window": "24-48_hours",
            "summary": "Finish selected comparisons, then perform OOF-only postprocessing and submission checks.",
            "action_ids": [] if later else anchor,
            "experiment_ids": later,
        },
    ]
    section["summary"] = (
        f"Execute {len(scheduled)} first-48-hour experiments dependency-first; threshold tuning, if present, is OOF-only and downstream."
    )
    return len({item.experiment_id for item in scheduled})


def _write_executive_summary(
    payload: dict[str, Any], evidence_pack: Mapping[str, Any], core: int, backlog: int
) -> None:
    section = next(
        (item for item in payload.get("sections") or [] if item.get("section_id") == "executive_summary"),
        None,
    )
    if section is None:
        return
    baseline = evidence_pack.get("baseline_evidence") or {}
    baseline_done = str(baseline.get("status") or "").casefold() in _SUCCESS
    baseline_value = baseline.get("metric_value")
    next_step = (
        "Reproduce the recorded baseline under the locked validation policy."
        if baseline_done else
        "Establish a leakage-safe baseline under the locked validation policy."
    )
    status = str(payload.get("synthesis_status") or "unknown")
    section["summary"] = (
        f"Synthesis status: {status}. Task: {payload.get('task_type') or 'unknown'}; "
        f"metric: {_metric_name(payload.get('metric') or {}, evidence_pack)}; primary validation: "
        f"{payload.get('recommended_validation') or 'not_available'}. Baseline result: "
        f"{baseline_value if baseline_value is not None else 'not_available'}. First next step: "
        f"{next_step} Core experiments: {core}; backlog: {backlog}. Threshold selection is "
        "downstream OOF-only postprocessing, never the first modeling action."
    )


def _clean_sections(payload: dict[str, Any]) -> None:
    actions = {
        action["action_id"]: action
        for action in payload.get("actions") or []
        if action.get("action_id")
    }
    for section in payload.get("sections") or []:
        section["action_ids"] = [
            action_id for action_id in section.get("action_ids") or [] if action_id in actions
        ]
        owned = [actions[action_id] for action_id in section["action_ids"]]
        if owned:
            section["evidence_refs"] = list(dict.fromkeys(
                ref for action in owned for ref in action.get("evidence_refs") or []
            ))
            section["related_hypothesis_ids"] = list(dict.fromkeys(
                ref for action in owned for ref in action.get("hypothesis_ids") or []
            ))
            section["source_refs"] = list(dict.fromkeys(
                ref for action in owned for ref in action.get("source_refs") or []
            ))
            section["eda_result_refs"] = list(dict.fromkeys(
                ref for action in owned for ref in action.get("eda_result_refs") or []
            ))


def _refresh_provenance(payload: dict[str, Any]) -> None:
    payload["action_provenance"] = [
        ActionProvenance(
            action_id=action["action_id"],
            source_refs=action.get("source_refs") or [],
            hypothesis_ids=action.get("hypothesis_ids") or [],
            motivating_hypothesis_ids=action.get("motivating_hypothesis_ids") or [],
            safety_hypothesis_ids=action.get("safety_hypothesis_ids") or [],
            validation_context_ids=action.get("validation_context_ids") or [],
            eda_result_refs=action.get("eda_result_refs") or [],
        ).model_dump(mode="json")
        for action in payload.get("actions") or []
        if action.get("action_id")
    ]


def _build_evidence_catalog(
    payload: Mapping[str, Any], evidence_pack: Mapping[str, Any]
) -> tuple[dict[str, EvidenceCatalogEntry], int]:
    refs: list[str] = []
    for action in payload.get("actions") or []:
        refs.extend(action.get("evidence_refs") or [])
    for key in ("experiments", "core_experiments", "experiment_backlog", "feature_experiment_families"):
        for item in payload.get(key) or []:
            refs.extend(item.get("evidence_refs") or [])
    for section in payload.get("sections") or []:
        refs.extend(section.get("evidence_refs") or [])
    counts = Counter(refs)
    catalog: dict[str, EvidenceCatalogEntry] = {}
    avoided = 0
    for ref in sorted(counts):
        try:
            value = resolve_evidence_ref(evidence_pack, ref)
            available = True
            warnings: list[str] = []
        except EvidencePathResolutionError:
            if ref == "final_synthesizer.repaired":
                value = "Deterministic synthesis repair marker."
                available = True
                warnings = ["Synthetic contract marker; not an EDA measurement."]
            else:
                value = None
                available = False
                warnings = ["Reference did not resolve in the validated EDA evidence pack."]
        preview = bounded_evidence_preview(value, max_chars=320)
        encoded = len(json.dumps(preview, ensure_ascii=False, default=str).encode("utf-8"))
        avoided += max(0, counts[ref] - 1) * encoded
        catalog[ref] = EvidenceCatalogEntry(
            ref=ref,
            resolved_value_preview=preview,
            value_type=type(value).__name__,
            source_component=ref.split(".", 1)[0].split("[", 1)[0],
            specificity=_specificity(ref, value),
            available=available,
            warnings=warnings,
        )
    return catalog, avoided


def _action_kind(action: Mapping[str, Any]) -> str:
    text = f"{action.get('action', '')} {action.get('reason', '')}".casefold()
    if "baseline" in text and any(token in text for token in ("reproduce", "rerun", "re-run")):
        return "baseline_reproduction"
    if "threshold" in text:
        return "threshold_postprocessing"
    if "primary id" in text or "primary identifier" in text:
        return "primary_id_safety"
    if "drift" in text:
        return "drift_risk"
    if "validation" in text and any(token in text for token in ("kfold", "fold", "split")):
        return "validation_policy"
    if action.get("feature_metadata") or any(token in text for token in (
        "feature", "imput", "family_size", "is_alone", "cabin", "ticket", "fare", "name",
    )):
        return "feature_family"
    if any(token in text for token in ("inspect", "audit", "review", "preserve")):
        return "broad_inspection"
    return "general"


def _hypothesis_categories(
    evidence_pack: Mapping[str, Any], result: FinalStrategyResult
) -> dict[str, str]:
    categories = {
        str(item.get("hypothesis_id")): str(item.get("category") or "unknown").casefold()
        for item in evidence_pack.get("hypothesis_results") or []
        if isinstance(item, Mapping) and item.get("hypothesis_id")
    }
    for action in result.actions:
        for hypothesis_id in action.hypothesis_ids:
            categories.setdefault(str(hypothesis_id), _category_from_id(str(hypothesis_id)))
    return categories


def _category_from_id(value: str) -> str:
    prefix = value.casefold().split("_", 1)[0]
    return {
        "val": "validation", "leak": "leakage", "rel": "relationship",
        "feat": "feature", "base": "baseline",
    }.get(prefix, prefix)


def _ids_in_categories(
    ids: list[str], categories: Mapping[str, str], wanted: set[str]
) -> list[str]:
    return [item for item in ids if categories.get(str(item), _category_from_id(str(item))) in wanted]


def _column_refs(evidence_pack: Mapping[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    schema = evidence_pack.get("inferred_schema") or {}
    for table_index, table in enumerate(schema.get("tables") or []):
        for column_index, column in enumerate((table or {}).get("columns") or []):
            name = str((column or {}).get("name") or "").strip()
            if name:
                found.setdefault(
                    name.casefold(),
                    f"inferred_schema.tables[{table_index}].columns[{column_index}].name",
                )
    for index, name in enumerate((evidence_pack.get("baseline_evidence") or {}).get("feature_columns") or []):
        found.setdefault(str(name).casefold(), f"baseline_evidence.feature_columns[{index}]")
    return found


def _column_diagnostic_ref(
    evidence_pack: Mapping[str, Any], column: str, family: str
) -> str | None:
    diagnostics = (evidence_pack.get("feature_diagnostics") or {}).get(family) or {}
    for collection in ("recommended_indicators", "columns", "high_cardinality_candidates"):
        for index, item in enumerate(diagnostics.get(collection) or []):
            if str((item or {}).get("column") or "").casefold() == column.casefold():
                return f"feature_diagnostics.{family}.{collection}[{index}]"
    return None


def _all_column_diagnostic_refs(
    evidence_pack: Mapping[str, Any], column: str
) -> list[str]:
    refs = []
    for family in (
        "missingness_diagnostics", "categorical_feature_diagnostics",
        "text_feature_diagnostics", "numeric_feature_diagnostics",
    ):
        ref = _column_diagnostic_ref(evidence_pack, column, family)
        if ref:
            refs.append(ref)
    return refs[:3]


def _experiment_dependencies(item: FinalStrategyExperiment) -> list[str]:
    if _is_baseline(item):
        return ["folds_locked", "schema_roles_locked"]
    if _is_threshold(item):
        return ["baseline_reproduced", "folds_locked", "provisional_model_selected", "oof_predictions_available"]
    return list(item.dependencies or ["baseline_reproduced", "folds_locked"])


def _estimated_cost(item: FinalStrategyExperiment) -> str:
    text = f"{item.name} {item.change}".casefold()
    if any(token in text for token in ("multi-seed", "nested", "blend", "ensemble")):
        return "high"
    if any(token in text for token in ("model", "frequency", "text", "group")):
        return "medium"
    return "low"


def _experiment_rank(item: FinalStrategyExperiment) -> tuple[int, int, str]:
    text = f"{item.experiment_id} {item.name}".casefold()
    if _is_baseline(item):
        stage = 0
    elif "ablation" in text and item.priority in {"P0", "P1"}:
        stage = 1
    elif item.experiment_id.startswith("family_missingness"):
        stage = 2
    elif item.experiment_id.startswith("family_relationship"):
        stage = 3
    elif item.experiment_id.startswith("family_cabin"):
        stage = 4
    elif item.experiment_id.startswith("family_"):
        stage = 5
    elif "model" in text:
        stage = 6
    elif _is_threshold(item):
        stage = 8
    elif "submission" in text:
        stage = 9
    else:
        stage = 6
    return stage, {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[item.priority], item.experiment_id


def _is_baseline(item: FinalStrategyExperiment) -> bool:
    text = f"{item.experiment_id} {item.name} {item.change}".casefold()
    return "baseline" in text and any(token in text for token in ("reproduce", "rerun", "re-run"))


def _is_threshold(item: FinalStrategyExperiment) -> bool:
    return "threshold" in f"{item.experiment_id} {item.name} {item.change}".casefold()


def _threshold_required(evidence_pack: Mapping[str, Any]) -> bool:
    metric = evidence_pack.get("metric_evidence") or {}
    return metric.get("requires_threshold") is True or metric.get("threshold_search_needed") is True


def _metric_name(metric: Mapping[str, Any], evidence_pack: Mapping[str, Any]) -> str:
    evidence = evidence_pack.get("metric_evidence") or {}
    return str(metric.get("name") or metric.get("metric_name") or evidence.get("metric_name") or "unknown")


def _specificity(ref: str, value: Any) -> str:
    if "[" in ref:
        return "item" if isinstance(value, Mapping) else "leaf"
    if "." not in ref:
        return "root"
    return "object" if isinstance(value, (Mapping, list, tuple)) else "leaf"


def _resolves(evidence_pack: Mapping[str, Any], ref: str) -> bool:
    try:
        resolve_evidence_ref(evidence_pack, ref)
    except EvidencePathResolutionError:
        return False
    return True


def _resolve_optional(evidence_pack: Mapping[str, Any], ref: str) -> Any:
    try:
        return resolve_evidence_ref(evidence_pack, ref)
    except EvidencePathResolutionError:
        return None


def _first_resolving(evidence_pack: Mapping[str, Any], refs: Iterable[str]) -> str | None:
    return next((ref for ref in refs if _resolves(evidence_pack, ref)), None)


def _action_tokens(text: str) -> set[str]:
    stop = {"the", "and", "with", "from", "only", "under", "this", "that", "use", "run"}
    return {
        token for token in re.findall(r"[a-z][a-z0-9_]+", text)
        if len(token) > 3 and token not in stop
    }


def _best_child_ref(
    evidence_pack: Mapping[str, Any], root: str, text: str, inputs: list[str]
) -> str | None:
    value = _resolve_optional(evidence_pack, root)
    if isinstance(value, Mapping):
        candidates = [f"{root}.{key}" for key in sorted(value)]
    elif isinstance(value, list):
        candidates = [f"{root}[{index}]" for index in range(min(len(value), 20))]
    else:
        return None
    scored = []
    for candidate in candidates:
        resolved = _resolve_optional(evidence_pack, candidate)
        rendered = json.dumps(resolved, ensure_ascii=False, default=str).casefold()
        score = sum(column in candidate.casefold() or column in rendered for column in inputs) * 10
        score += sum(
            token in candidate.casefold() or token in rendered
            for token in _action_tokens(text)
        )
        scored.append((-score, candidate))
    return sorted(scored)[0][1] if scored else None


def _hypothesis_source_flags(result: FinalStrategyResult) -> list[bool]:
    linked = {str(link.hypothesis_id) for link in result.source_to_hypothesis_links}
    hypotheses = {
        str(hypothesis_id) for action in result.actions for hypothesis_id in action.hypothesis_ids
    }
    return [hypothesis_id in linked for hypothesis_id in sorted(hypotheses)]


def _append_provenance_limitation(
    payload: dict[str, Any], result: FinalStrategyResult
) -> None:
    if result.source_to_hypothesis_links:
        return
    limitation = (
        "Research hypotheses contained no validated source refs; no arbitrary retrieved documents were attached to strategy actions."
    )
    limitations = list(payload.get("limitations") or [])
    if limitation not in limitations:
        limitations.append(limitation)
    payload["limitations"] = limitations


def _budget_from_environment() -> ExperimentBudget:
    def value(name: str, default: int, *, minimum: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        parsed = int(raw)
        if parsed < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return parsed

    return ExperimentBudget(
        max_core_experiments=value("FINAL_STRATEGY_MAX_CORE_EXPERIMENTS", 8),
        max_first_24h_experiments=value("FINAL_STRATEGY_MAX_FIRST_24H_EXPERIMENTS", 4),
        max_first_48h_experiments=value("FINAL_STRATEGY_MAX_FIRST_48H_EXPERIMENTS", 8),
        max_high_cost_experiments=value(
            "FINAL_STRATEGY_MAX_HIGH_COST_EXPERIMENTS", 2, minimum=0
        ),
    )


__all__ = ["compact_final_strategy", "validate_compacted_strategy"]
