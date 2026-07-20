from __future__ import annotations

import json
from typing import Any

from kaggle_researcher.contracts.final_strategy_protocol import (
    PromptFingerprint,
    StrategyRenderingDraft,
    StrategySkeleton,
)
from kaggle_researcher.reasoning.prompts.final_strategy_common import prompt_fingerprint


FINAL_STRATEGY_RENDERING_PROMPT_VERSION = "2.0"
RENDERING_USER_TEMPLATE = "frozen_strategy_skeleton + approved evidence previews + wording-only schema"
FINAL_STRATEGY_RENDERING_SYSTEM_PROMPT = """You are the Final Strategy Rendering component of an automated Kaggle research system.
A validated strategy skeleton is frozen. You do not make strategic decisions. Improve only clarity, concision, and readability.
Return exactly one JSON object matching StrategyRenderingDraft. Do not output Markdown, code fences, commentary, or chain-of-thought.
The skeleton ID and hash, all action/experiment/family/section IDs, evidence and source refs, hypotheses, safety and validation IDs, kinds, priorities, confidence, dependencies, model identities, validation, metric, columns, fit scopes, leakage risks, budget assignment, first-48-hour membership, and section structure are immutable.
Return wording records for exactly the existing IDs. Do not introduce new claims, features, models, experiments, risks, expected gains, evidence, or identifiers.
Write concise professional language with concrete verbs, inputs, validation boundaries, testable acceptance rules, and honest uncertainty. Do not promise leaderboard gains.
The executive summary must state synthesis status, task type, metric, validation, available baseline, highest-priority safety issue, next modeling step, and core/backlog counts. Baseline reproduction remains the first modeling step.
Describe related feature arms as one family. Threshold tuning remains downstream, uses OOF predictions only after provisional selection, compares with the default threshold, and never uses test labels.
Use only supplied evidence previews. Return JSON only."""


def build_rendering_prompt(
    skeleton: StrategySkeleton,
    *,
    max_chars: int = 50000,
) -> tuple[str, str, PromptFingerprint]:
    immutable = skeleton.model_dump(mode="json", exclude={"evidence_catalog"})
    previews = {
        ref: value for ref, value in list(skeleton.evidence_catalog.items())[:80]
    }
    payload = {
        "instruction": "Return wording only; the frozen structure is immutable.",
        "skeleton_id": skeleton.skeleton_id,
        "skeleton_hash": skeleton.skeleton_hash,
        "immutable_strategy_payload": immutable,
        "renderable_evidence_previews": previews,
        "required_section_order": [item["section_id"] for item in skeleton.section_structure],
        "style_constraints": {"concise": True, "no_score_promises": True, "max_field_chars": 1200},
        "rendering_output_schema": StrategyRenderingDraft.model_json_schema(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > max_chars:
        payload["renderable_evidence_previews"] = {}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = prompt_fingerprint(
        prompt_name="final_strategy_rendering",
        prompt_version=FINAL_STRATEGY_RENDERING_PROMPT_VERSION,
        system_prompt=FINAL_STRATEGY_RENDERING_SYSTEM_PROMPT,
        user_template=RENDERING_USER_TEMPLATE,
    )
    return FINAL_STRATEGY_RENDERING_SYSTEM_PROMPT, encoded, fingerprint


def build_rendering_repair_prompt(
    *, invalid_draft: dict[str, Any], issues: list[dict[str, Any]], skeleton: StrategySkeleton,
) -> str:
    return json.dumps({
        "instruction": "Correct only the listed wording/ID errors. Do not alter or add strategy structure. Return JSON only.",
        "invalid_rendering": invalid_draft,
        "validation_issues": issues,
        "required_identity": {"skeleton_id": skeleton.skeleton_id, "skeleton_hash": skeleton.skeleton_hash},
        "allowed_ids": {
            "actions": [item["action_id"] for item in skeleton.actions],
            "experiments": [item["experiment_id"] for item in [*skeleton.core_experiments, *skeleton.experiment_backlog]],
            "families": [item["family_id"] for item in skeleton.feature_experiment_families],
            "sections": [item["section_id"] for item in skeleton.section_structure],
        },
        "output_schema": StrategyRenderingDraft.model_json_schema(),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "FINAL_STRATEGY_RENDERING_PROMPT_VERSION",
    "FINAL_STRATEGY_RENDERING_SYSTEM_PROMPT",
    "build_rendering_prompt", "build_rendering_repair_prompt",
]
