from __future__ import annotations

import json
import os
from typing import Any, Mapping

from pydantic import Field

from kaggle_researcher.contracts.base import ContractModel
from kaggle_researcher.contracts.evidence import EvidencePathResolutionError, resolve_evidence_ref
from kaggle_researcher.contracts.final_strategy import REQUIRED_SECTION_IDS
from kaggle_researcher.contracts.registries import ContractRegistries
from kaggle_researcher.contracts.synthesis_context import FinalSynthesisContext
from kaggle_researcher.reasoning.model_registry import supported_models


CONTEXT_POLICY_VERSION = "2.0"


class FinalStrategySelectionContext(ContractModel):
    model_config = {**ContractModel.model_config, "protected_namespaces": ()}

    context_policy_version: str = CONTEXT_POLICY_VERSION
    competition_id: str
    task_type: str
    metric_contract: dict[str, Any]
    validation_contract: dict[str, Any]
    schema_summary: dict[str, Any]
    baseline_summary: dict[str, Any]
    ablation_summary: dict[str, Any]
    drift_summary: dict[str, Any]
    leakage_summary: list[dict[str, Any]]
    feature_diagnostic_summary: dict[str, Any]
    hypothesis_catalog: list[dict[str, Any]]
    source_catalog: list[dict[str, Any]]
    evidence_catalog: list[dict[str, Any]]
    evidence_manifest_metadata: dict[str, Any] = Field(default_factory=dict)
    model_catalog: list[dict[str, Any]]
    safety_constraint_catalog: list[dict[str, Any]]
    validation_requirement_catalog: list[dict[str, Any]]
    approved_experiment_catalog: list[dict[str, Any]] = Field(default_factory=list)
    required_section_ids: list[str]
    strategy_limits: dict[str, int]
    unavailable_capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    omitted_catalog_counts: dict[str, int] = Field(default_factory=dict)
    context_character_count: int = 0
    context_character_budget: int


def build_final_strategy_selection_context(
    context: FinalSynthesisContext,
    registries: ContractRegistries,
    *,
    max_chars: int | None = None,
) -> FinalStrategySelectionContext:
    """Build a deterministic, bounded catalog for strategy selection."""

    budget = max_chars or _env_int("FINAL_SYNTHESIS_CONTEXT_MAX_CHARS", 60000)
    eda = context.eda_evidence_pack.model_dump(mode="json")
    hypotheses = []
    priority_by_id: dict[str, str] = {}
    for identifier in sorted(registries.hypotheses.by_id, key=str):
        raw = registries.hypotheses.by_id[identifier].model_dump(mode="json")
        hypothesis_id = str(identifier)
        priority_by_id[hypothesis_id] = str(raw.get("priority") or "P2")
        result = next(
            (
                item
                for item in eda.get("hypothesis_results") or []
                if str(item.get("hypothesis_id")) == hypothesis_id
            ),
            {},
        )
        hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "category": raw.get("category") or result.get("category") or "unknown",
            "statement": raw.get("claim") or raw.get("hypothesis") or raw.get("finding") or "",
            "priority": raw.get("priority") or "P2",
            "source_refs": sorted(set(map(str, raw.get("source_refs") or []))),
            "testability": raw.get("testability") or raw.get("test_method") or "bounded_by_eda",
            "eda_result_status": result.get("status") or "not_tested",
            "relevant_evidence_refs": sorted(set(map(str, result.get("evidence_refs") or [])))[:6],
        })
    hypotheses.sort(key=lambda item: (_priority_rank(item["priority"]), item["hypothesis_id"]))

    sources = []
    for document in sorted(context.retrieved_documents, key=lambda item: str(item.id)):
        metadata = document.metadata or {}
        sources.append({
            "source_ref": str(document.id),
            "source_type": str(document.source),
            "title": str(document.title)[:240],
            "claim_summary": str(
                metadata.get("claim_summary")
                or metadata.get("summary")
                or document.content[:360]
            ),
        })

    required_evidence = _required_evidence_refs(eda)
    evidence = []
    for ref in sorted(map(str, context.allowed_eda_result_refs)):
        entry = context.lookup_evidence_ref(ref)
        if entry is None or entry.canonical_path is None:
            continue
        try:
            value = resolve_evidence_ref(eda, entry.canonical_path)
        except EvidencePathResolutionError:
            continue
        evidence.append({
            "evidence_ref": ref,
            "value_preview": _bounded_value(value),
            "value_type": entry.value_type or type(value).__name__,
            "specificity": _specificity(ref, value),
            "semantic_tags": _semantic_tags(ref),
            "required": ref in required_evidence,
        })
    evidence.sort(key=lambda item: (
        not item["required"],
        item["specificity"] == "root",
        item["evidence_ref"],
    ))

    models = [
        {
            "canonical_family_id": item.canonical_family_id,
            "implementation_id": item.implementation_id,
            "display_name": item.display_name,
            "available": item.available,
            "capabilities": dict(item.capabilities),
        }
        for item in supported_models(context.plan_data.task_type)
    ]
    safety = [
        item.model_dump(mode="json")
        for _, item in sorted(registries.safety_constraints.by_id.items(), key=lambda pair: str(pair[0]))
    ]
    requirements = [
        item.model_dump(mode="json")
        for _, item in sorted(registries.validation_requirements.by_id.items(), key=lambda pair: str(pair[0]))
    ]
    approved_experiments = [
        {
            "experiment_id": str(identifier),
            "experiment": str(registries.experiments.by_id[identifier].experiment)[:500],
            "why": str(registries.experiments.by_id[identifier].why)[:500],
            "cost": str(registries.experiments.by_id[identifier].cost),
            "source_hypothesis_ids": [
                str(value)
                for value in registries.experiments.experiment_to_hypotheses.get(
                    identifier, ()
                )
            ],
            "evidence_ids": [
                str(value)
                for value in registries.experiments.by_id[identifier].evidence_ids[:8]
            ],
        }
        for identifier in sorted(registries.experiments.approved_ids, key=str)[:100]
    ]
    payload: dict[str, Any] = {
        "competition_id": context.eda_evidence_pack.competition_id,
        "task_type": context.plan_data.task_type,
        "metric_contract": _bounded_mapping(eda.get("metric_evidence") or {"metric_name": context.plan_data.metric}),
        "validation_contract": _bounded_mapping(eda.get("validation_evidence") or {}),
        "schema_summary": _schema_summary(eda.get("inferred_schema") or {}),
        "baseline_summary": _bounded_mapping(eda.get("baseline_evidence") or {}),
        "ablation_summary": _bounded_mapping(eda.get("baseline_ablation_evidence") or {}),
        "drift_summary": _bounded_mapping(eda.get("drift_evidence") or {}),
        "leakage_summary": [_bounded_mapping(item) for item in (eda.get("leakage_evidence") or [])[:30]],
        "feature_diagnostic_summary": _bounded_mapping(eda.get("feature_diagnostics") or {}),
        "hypothesis_catalog": hypotheses,
        "source_catalog": sources,
        "evidence_catalog": evidence,
        "evidence_manifest_metadata": context.reference_prompt_payload()[
            "evidence_manifest_metadata"
        ],
        "model_catalog": models,
        "safety_constraint_catalog": safety,
        "validation_requirement_catalog": requirements,
        "approved_experiment_catalog": approved_experiments,
        "required_section_ids": list(REQUIRED_SECTION_IDS),
        "strategy_limits": {
            "max_actions": _env_int("FINAL_SYNTHESIS_MAX_ACTIONS", 15),
            "max_core_experiments": _env_int("FINAL_SYNTHESIS_MAX_CORE_EXPERIMENTS", 8),
            "max_backlog_experiments": _env_int("FINAL_SYNTHESIS_MAX_BACKLOG_EXPERIMENTS", 12),
            "max_first_24h_experiments": _env_int("FINAL_SYNTHESIS_MAX_FIRST_24H_EXPERIMENTS", 4),
            "max_first_48h_experiments": _env_int("FINAL_SYNTHESIS_MAX_FIRST_48H_EXPERIMENTS", 8),
        },
        "unavailable_capabilities": [
            message for message in context.optional_stage_failure_messages if message
        ],
        "limitations": list(dict.fromkeys(context.limitations))[:30],
        "omitted_catalog_counts": {},
        "context_character_budget": budget,
    }
    _fit_catalogs(payload, budget, required_evidence)
    payload["context_character_count"] = len(_stable_json(payload))
    return FinalStrategySelectionContext.model_validate(payload)


def _fit_catalogs(payload: dict[str, Any], budget: int, required_evidence: set[str]) -> None:
    omitted = {
        "hypothesis_catalog": 0,
        "source_catalog": 0,
        "evidence_catalog": 0,
        "approved_experiment_catalog": 0,
    }
    while len(_stable_json(payload)) > budget:
        evidence = payload["evidence_catalog"]
        removable_index = next(
            (index for index in range(len(evidence) - 1, -1, -1)
             if evidence[index]["evidence_ref"] not in required_evidence),
            None,
        )
        if removable_index is not None:
            evidence.pop(removable_index)
            omitted["evidence_catalog"] += 1
            continue
        hypotheses = payload["hypothesis_catalog"]
        if hypotheses and _priority_rank(hypotheses[-1].get("priority")) >= 2:
            hypotheses.pop()
            omitted["hypothesis_catalog"] += 1
            continue
        if payload["source_catalog"]:
            payload["source_catalog"].pop()
            omitted["source_catalog"] += 1
            continue
        if payload["approved_experiment_catalog"]:
            payload["approved_experiment_catalog"].pop()
            omitted["approved_experiment_catalog"] += 1
            continue
        break
    payload["omitted_catalog_counts"] = omitted


def _required_evidence_refs(eda: Mapping[str, Any]) -> set[str]:
    refs = {"validation_evidence.primary_validation"}
    if eda.get("baseline_evidence"):
        refs.add("baseline_evidence.status")
    if (eda.get("metric_evidence") or {}).get("requires_threshold") is not None:
        refs.add("metric_evidence.requires_threshold")
    if (eda.get("inferred_schema") or {}).get("primary_id_column"):
        refs.add("inferred_schema.primary_id_column")
    for collection in (
        eda.get("safety_constraints") or [],
        eda.get("validation_requirements") or [],
    ):
        for item in collection:
            if isinstance(item, Mapping):
                refs.update(map(str, item.get("evidence_refs") or []))
    return refs


def _schema_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "train_base_table", "test_base_table", "target_column", "prediction_column",
        "primary_id_column", "sample_submission_table", "global_roles", "tables",
    }
    return {key: _bounded_value(value[key], limit=4000) for key in sorted(value) if key in keep}


def _bounded_mapping(value: Mapping[str, Any], limit: int = 7000) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(value):
        candidate = {**result, str(key): _bounded_value(value[key])}
        if len(_stable_json(candidate)) > limit:
            break
        result = candidate
    return result


def _bounded_value(value: Any, limit: int = 600) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, Mapping):
        result = {str(key): _bounded_value(item, max(80, limit // 4)) for key, item in list(value.items())[:12]}
        rendered = _stable_json(result)
        return result if len(rendered) <= limit else rendered[:limit]
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, max(80, limit // 4)) for item in value[:8]]
    return value


def _semantic_tags(ref: str) -> list[str]:
    tokens = ref.replace("[", ".").replace("]", "").replace("_", ".").split(".")
    return sorted({token.casefold() for token in tokens if len(token) > 2})[:8]


def _specificity(ref: str, value: Any) -> str:
    if "[" in ref:
        return "item" if isinstance(value, Mapping) else "leaf"
    if "." not in ref:
        return "root"
    return "object" if isinstance(value, (Mapping, list)) else "leaf"


def _priority_rank(value: Any) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(str(value), 4)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = ["CONTEXT_POLICY_VERSION", "FinalStrategySelectionContext", "build_final_strategy_selection_context"]
