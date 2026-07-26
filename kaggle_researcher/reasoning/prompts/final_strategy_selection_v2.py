from __future__ import annotations

import json
from typing import Any

from kaggle_researcher.contracts.final_strategy_protocol import (
    PromptFingerprint,
    StrategySelectionDraft,
)
from kaggle_researcher.reasoning.final_strategy_context import (
    CONTEXT_POLICY_VERSION,
    FinalStrategySelectionContext,
)
from kaggle_researcher.reasoning.prompts.final_strategy_common import prompt_fingerprint


FINAL_STRATEGY_SELECTION_PROMPT_VERSION = "2.0"
SELECTION_USER_TEMPLATE = "strategy_selection_context + output_schema + exact allowed ID catalogs"
FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT = """You are the Strategy Selection component of an automated Kaggle research system.
You do not write the final report. Select the smallest safe, evidence-grounded strategy from the validated catalogs.
Return exactly one JSON object matching StrategySelectionDraft. Do not output Markdown, code fences, commentary, or chain-of-thought.
Always include contract_family="strategy_selection_draft" and schema_version="2.0".
Use only exact IDs from the allowed catalogs. Never invent, rewrite, abbreviate, or approximate an ID. Do not generate final stable action, family, or experiment IDs; use unique client keys.
Respect strategy_limits. Select at most 15 actions, 8 core experiments, 12 backlog experiments, and 8 first-48-hour experiments unless the supplied limits are stricter.
Use one precise primary evidence ref where possible, no more than three supporting refs, and no more than two limitation refs. Do not reproduce all EDA evidence or attach unrelated broad evidence roots.
Preserve source -> hypothesis -> EDA -> strategy provenance. Use source refs only when the hypothesis catalog explicitly owns them. EDA-only and deterministic safety actions need no source.
Attach approved_experiment_ids to actions only from the approved experiment catalog; never restore rejected or unknown experiment IDs.
Assign hypotheses by semantic role. Do not attach one broad hypothesis union to every action.
Group related feature representations into controlled experiment families, and only when the actual input columns exist. Do not hard-code competition-specific features.
Use canonical available model_family_ids only. Never compare aliases of the same canonical family or use task-incompatible models.
Keep the selected metric, validation, schema roles, safety constraints, and validation requirements immutable.
When completed baseline evidence exists, baseline reproduction is the first modeling experiment. Threshold selection is downstream OOF-only postprocessing after provisional model selection and OOF prediction generation; never use test labels.
Rank contract steps, baseline, stable low-cost feature families, distinct model comparisons, and OOF postprocessing. Put lower-priority supported work, including remaining P2 ideas, in backlog.
Use every required section ID exactly once in section_plan. summary_intent is a plan, not polished prose.
Before returning, silently self-check all IDs, budgets, dependencies, evidence minimality, section coverage, baseline order, and OOF threshold constraints.
Do not predict score gains or add fields absent from the schema. Return JSON only."""


def build_selection_prompt(
    context: FinalStrategySelectionContext,
) -> tuple[str, str, PromptFingerprint]:
    payload = context.model_dump(mode="json")
    user_payload: dict[str, Any] = {
        "instruction": "Return one StrategySelectionDraft JSON object only.",
        "selection_context": payload,
        "allowed_source_refs": [item["source_ref"] for item in payload["source_catalog"]],
        "allowed_hypothesis_ids": [item["hypothesis_id"] for item in payload["hypothesis_catalog"]],
        "allowed_evidence_refs": [item["evidence_ref"] for item in payload["evidence_catalog"]],
        "allowed_model_family_ids": [item["canonical_family_id"] for item in payload["model_catalog"]],
        "allowed_safety_constraint_ids": [
            item.get("safety_constraint_id") for item in payload["safety_constraint_catalog"]
        ],
        "allowed_validation_requirement_ids": [
            item.get("validation_requirement_id")
            for item in payload["validation_requirement_catalog"]
        ],
        "allowed_approved_experiment_ids": [
            item["experiment_id"] for item in payload["approved_experiment_catalog"]
        ],
        "required_section_ids": payload["required_section_ids"],
        "output_schema": StrategySelectionDraft.model_json_schema(),
    }
    fingerprint = prompt_fingerprint(
        prompt_name="final_strategy_selection",
        prompt_version=FINAL_STRATEGY_SELECTION_PROMPT_VERSION,
        system_prompt=FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT,
        user_template=SELECTION_USER_TEMPLATE,
        context_policy_version=CONTEXT_POLICY_VERSION,
    )
    return (
        FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT,
        json.dumps(user_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        fingerprint,
    )


def build_selection_repair_prompt(
    *, invalid_draft: dict[str, Any], issues: list[dict[str, Any]],
    context: FinalStrategySelectionContext,
) -> str:
    payload = context.model_dump(mode="json")
    return json.dumps({
        "instruction": (
            "Correct only the listed validation errors. Preserve supported choices, remove unknown IDs and forbidden fields, "
            "repair client-key references, fill required fields without adding unsupported content, and return JSON only."
        ),
        "invalid_draft": invalid_draft,
        "validation_issues": issues,
        "allowed_catalogs": {
            "source_refs": [item["source_ref"] for item in payload["source_catalog"]],
            "hypothesis_ids": [item["hypothesis_id"] for item in payload["hypothesis_catalog"]],
            "evidence_refs": [item["evidence_ref"] for item in payload["evidence_catalog"]],
            "model_family_ids": [item["canonical_family_id"] for item in payload["model_catalog"]],
            "safety_constraint_ids": [item.get("safety_constraint_id") for item in payload["safety_constraint_catalog"]],
            "validation_requirement_ids": [item.get("validation_requirement_id") for item in payload["validation_requirement_catalog"]],
            "approved_experiment_ids": [
                item["experiment_id"] for item in payload["approved_experiment_catalog"]
            ],
            "required_section_ids": payload["required_section_ids"],
        },
        "output_schema": StrategySelectionDraft.model_json_schema(),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "FINAL_STRATEGY_SELECTION_PROMPT_VERSION",
    "FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT",
    "build_selection_prompt", "build_selection_repair_prompt",
]
