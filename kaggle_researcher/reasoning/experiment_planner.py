from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.contracts.normalization import EXPERIMENT_EVIDENCE_ID_ALIASES
from kaggle_researcher.reasoning.common import (
    CANONICAL_REASONING_EVIDENCE_IDS,
    format_retrieved_documents,
    known_evidence_ids,
)
from kaggle_researcher.reasoning.prompts import SYSTEM_RULES
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeakageRiskResult,
    MetricResult,
    RetrievedDocument,
    ValidationResult,
)


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
EVIDENCE_ID_ALIASES = EXPERIMENT_EVIDENCE_ID_ALIASES


async def plan_experiments(
    validation_result: ValidationResult,
    leakage_result: LeakageRiskResult,
    metric_result: MetricResult,
    retrieved_documents: list[RetrievedDocument] | None = None,
    client: DeepSeekClient | None = None,
    model: str = "deepseek-v4-pro",
    *,
    chunks: list[RetrievedDocument] | None = None,
) -> list[ExperimentItem]:
    if client is None:
        raise ValueError("client is required")
    docs = retrieved_documents if retrieved_documents is not None else chunks or []
    allowed_evidence_ids = known_experiment_evidence_ids(docs)
    response = await client.chat_json(
        model=model,
        system_prompt=f"{SYSTEM_RULES}\n\n"
        + (
            "You are the Experiment Planner. Build a prioritized experiment queue with ROI logic. "
            "Return JSON only as an object with key 'experiments'. Each item must match "
            "ExperimentItem schema and include priority, experiment, why, cost, expected_gain, "
            "risk, and evidence_ids. Priorities must be P0, P1, P2, or P3. P0 should include "
            "honest validation and a baseline if not already covered. Do not present EDA, "
            "adversarial validation, leakage checks, or leakage detection as already executed. "
            "Use planned/action language such as 'run', 'check', 'inspect', or 'evaluate'. "
            "Sort conceptually by ROI and urgency; final output will be sorted P0 to P3."
            " Every evidence_ids entry must exactly match one value from allowed_evidence_ids. "
            "Do not invent aliases, section names, abbreviations, or semantic shorthand. "
            "Do not use validation_policy unless it is explicitly present in allowed_evidence_ids. "
            "When no valid supporting evidence exists, return an empty evidence_ids list."
        ),
        user_prompt=json.dumps(
            {
                "validation_result": validation_result.model_dump(mode="json"),
                "leakage_result": leakage_result.model_dump(mode="json"),
                "metric_result": metric_result.model_dump(mode="json"),
                "retrieved_documents": format_retrieved_documents(docs),
                "allowed_evidence_ids": allowed_evidence_ids,
                "expected_schema": {
                    "experiments": [ExperimentItem.model_json_schema()],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        timeout=120,
    )
    raw_items = response.get("experiments", response if isinstance(response, list) else [])
    experiments, replacements = _parse_and_normalize_experiments(raw_items)
    unknown_ids = _unknown_evidence_ids(experiments, allowed_evidence_ids)
    if unknown_ids:
        repaired = await _repair_evidence_ids_once(
            client=client,
            model=model,
            experiments=experiments,
            invalid_evidence_ids=unknown_ids,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        experiments, repair_replacements = _parse_and_normalize_experiments(repaired)
        replacements.extend(repair_replacements)
        unknown_ids = _unknown_evidence_ids(experiments, allowed_evidence_ids)
    if unknown_ids:
        raise ValueError(_unknown_evidence_message(unknown_ids, allowed_evidence_ids, replacements, stage="final"))

    experiments = _ensure_required_p0_experiments(experiments)
    experiments = _assign_and_validate_experiment_ids(experiments)
    return sorted(experiments, key=lambda item: PRIORITY_ORDER[item.priority])


def known_experiment_evidence_ids(docs: Sequence[RetrievedDocument]) -> list[str]:
    """Return the single canonical evidence registry used by prompt and validation."""
    return known_evidence_ids(docs, additional_ids=CANONICAL_REASONING_EVIDENCE_IDS)


def normalize_evidence_ids(
    evidence_ids: Sequence[str],
    *,
    aliases: Mapping[str, str] = EVIDENCE_ID_ALIASES,
) -> list[str]:
    normalized: list[str] = []
    for evidence_id in evidence_ids:
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("Experiment evidence_ids must contain non-empty strings.")
        canonical = aliases.get(evidence_id, evidence_id)
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _parse_and_normalize_experiments(raw_items: object) -> tuple[list[ExperimentItem], list[tuple[str, str]]]:
    if not isinstance(raw_items, list):
        raise ValueError("Experiment planner response must contain an experiments list.")
    experiments: list[ExperimentItem] = []
    replacements: list[tuple[str, str]] = []
    for item in raw_items:
        experiment = ExperimentItem.model_validate(item)
        original = list(experiment.evidence_ids)
        normalized = normalize_evidence_ids(original)
        replacements.extend((value, EVIDENCE_ID_ALIASES[value]) for value in original if value in EVIDENCE_ID_ALIASES and value != EVIDENCE_ID_ALIASES[value])
        experiments.append(experiment.model_copy(update={"evidence_ids": normalized}))
    return experiments, replacements


def _assign_and_validate_experiment_ids(experiments: Sequence[ExperimentItem]) -> list[ExperimentItem]:
    assigned: list[ExperimentItem] = []
    seen: set[str] = set()
    for experiment in experiments:
        experiment_id = experiment.experiment_id or _stable_experiment_id(experiment.experiment)
        if experiment_id in seen:
            raise ValueError(f"Experiment plan contains duplicate experiment_id: {experiment_id!r}.")
        seen.add(experiment_id)
        assigned.append(experiment.model_copy(update={"experiment_id": experiment_id}))
    return assigned


def _stable_experiment_id(text: str) -> str:
    normalized = " ".join(text.lower().split()).encode("utf-8")
    return f"exp_{hashlib.sha256(normalized).hexdigest()[:12]}"


def _unknown_evidence_ids(experiments: Sequence[ExperimentItem], allowed_evidence_ids: Sequence[str]) -> list[str]:
    allowed = set(allowed_evidence_ids)
    return sorted({evidence_id for experiment in experiments for evidence_id in experiment.evidence_ids if evidence_id not in allowed})


async def _repair_evidence_ids_once(
    *,
    client: DeepSeekClient,
    model: str,
    experiments: Sequence[ExperimentItem],
    invalid_evidence_ids: Sequence[str],
    allowed_evidence_ids: Sequence[str],
) -> object:
    response = await client.chat_json(
        model=model,
        system_prompt=f"{SYSTEM_RULES}\n\nRepair only ExperimentItem evidence_ids. Replace or remove invalid evidence IDs. Use only exact IDs from allowed_evidence_ids. Do not change experiment intent unless required to remove unsupported grounding. Do not add new experiments. Return JSON only with key 'experiments'.",
        user_prompt=json.dumps({"invalid_evidence_ids": list(invalid_evidence_ids), "allowed_evidence_ids": list(allowed_evidence_ids), "experiments": [item.model_dump(mode="json") for item in experiments]}, ensure_ascii=False, indent=2),
        timeout=120,
    )
    return response.get("experiments", response if isinstance(response, list) else []) if isinstance(response, (dict, list)) else []


def _unknown_evidence_message(unknown: Sequence[str], allowed: Sequence[str], replacements: Sequence[tuple[str, str]], *, stage: str) -> str:
    alias_text = ", ".join(f"{source}->{target}" for source, target in replacements) or "none"
    allowed_sample = ", ".join(list(allowed)[:12])
    return f"ExperimentItem contains unknown evidence_ids after alias normalization at {stage} stage: {list(unknown)}. Allowed IDs: [{allowed_sample}]. Alias replacements: {alias_text}."


def _ensure_required_p0_experiments(experiments: list[ExperimentItem]) -> list[ExperimentItem]:
    updated = list(experiments)
    p0_text = " ".join(
        f"{item.experiment} {item.why} {item.risk}".lower()
        for item in updated
        if item.priority == "P0"
    )
    if not any(token in p0_text for token in ("validation", "cv", "holdout", "split")):
        updated.append(
            ExperimentItem(
                priority="P0",
                experiment="Establish honest validation before model iteration",
                why="A reliable validation protocol is required before trusting experiment gains.",
                cost="low",
                expected_gain="high",
                risk="Without honest validation, later experiments may optimize noise or public leaderboard artifacts.",
                evidence_ids=[],
            )
        )
    if "baseline" not in p0_text:
        updated.append(
            ExperimentItem(
                priority="P0",
                experiment="Train a simple baseline and evaluate on honest validation",
                why="A baseline anchors ROI estimates for later, more expensive experiments.",
                cost="low",
                expected_gain="medium",
                risk="Skipping the baseline makes it harder to separate real improvements from pipeline variance.",
                evidence_ids=[],
            )
        )
    return updated
