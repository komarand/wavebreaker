from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from kaggle_researcher.brief_context import PackedBriefContext, pack_brief_context
from kaggle_researcher.brief_prompts import BRIEF_SYSTEM_PROMPT
from kaggle_researcher.brief_schemas import CompetitionBrief
from kaggle_researcher.clients.deepseek_client import DeepSeekClient
from kaggle_researcher.config import Settings
from kaggle_researcher.facts.models import CompetitionFacts


class BriefGenerationError(RuntimeError):
    """Raised when the model twice returns an invalid CompetitionBrief payload."""


async def generate_brief(
    facts: CompetitionFacts,
    settings: Settings,
) -> CompetitionBrief:
    packed_context = pack_brief_context(facts, settings.max_context_tokens)
    client = DeepSeekClient(api_key=settings.deepseek_api_key)
    user_prompt = _initial_user_prompt(packed_context)

    payload = await client.chat_json(
        model=settings.deepseek_v4_pro,
        system_prompt=BRIEF_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    try:
        return CompetitionBrief.model_validate(payload)
    except ValidationError as first_error:
        retry_payload = await client.chat_json(
            model=settings.deepseek_v4_pro,
            system_prompt=BRIEF_SYSTEM_PROMPT,
            user_prompt=_retry_user_prompt(
                packed_context=packed_context,
                invalid_payload=payload,
                validation_error=first_error,
            ),
        )
        try:
            return CompetitionBrief.model_validate(retry_payload)
        except ValidationError as retry_error:
            raise BriefGenerationError(
                "DeepSeek returned an invalid CompetitionBrief after one schema retry. "
                f"First validation error: {first_error}. "
                f"Retry validation error: {retry_error}."
            ) from retry_error


def _initial_user_prompt(packed_context: PackedBriefContext) -> str:
    included_ids = json.dumps(
        packed_context.stats.included_source_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    limitations = json.dumps(
        packed_context.stats.limitations,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Create the CompetitionBrief from the packed evidence below.\n"
        "Only the IDs in AVAILABLE_SOURCE_IDS may appear in Claim.source_ids.\n\n"
        f"<AVAILABLE_SOURCE_IDS>\n{included_ids}\n</AVAILABLE_SOURCE_IDS>\n\n"
        f"<CONTEXT_PACKING_LIMITATIONS>\n{limitations}\n"
        "</CONTEXT_PACKING_LIMITATIONS>\n\n"
        f"<PACKED_BRIEF_CONTEXT>\n{packed_context.text}\n"
        "</PACKED_BRIEF_CONTEXT>"
    )


def _retry_user_prompt(
    *,
    packed_context: PackedBriefContext,
    invalid_payload: dict[str, Any],
    validation_error: ValidationError,
) -> str:
    invalid_json = json.dumps(
        invalid_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{_initial_user_prompt(packed_context)}\n\n"
        "The previous JSON response failed CompetitionBrief schema validation. "
        "Return a complete replacement JSON object. Do not explain the correction.\n\n"
        f"<INVALID_RESPONSE>\n{invalid_json}\n</INVALID_RESPONSE>\n\n"
        f"<SCHEMA_VALIDATION_ERROR>\n{validation_error}\n"
        "</SCHEMA_VALIDATION_ERROR>"
    )
