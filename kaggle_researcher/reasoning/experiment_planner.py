from __future__ import annotations

import json

from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.reasoning.common import format_retrieved_documents, validate_evidence_ids
from kaggle_researcher.reasoning.prompts import SYSTEM_RULES
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeakageRiskResult,
    MetricResult,
    RetrievedDocument,
    ValidationResult,
)


PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


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
        ),
        user_prompt=json.dumps(
            {
                "validation_result": validation_result.model_dump(mode="json"),
                "leakage_result": leakage_result.model_dump(mode="json"),
                "metric_result": metric_result.model_dump(mode="json"),
                "retrieved_documents": format_retrieved_documents(docs),
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
    experiments = [ExperimentItem.model_validate(item) for item in raw_items]
    for experiment in experiments:
        unknown_ids = validate_evidence_ids(experiment, docs)
        if unknown_ids:
            raise ValueError(f"ExperimentItem contains unknown evidence_ids: {unknown_ids}")

    experiments = _ensure_required_p0_experiments(experiments)
    return sorted(experiments, key=lambda item: PRIORITY_ORDER[item.priority])


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
