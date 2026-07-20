from __future__ import annotations

import hashlib
import json

from kaggle_researcher.contracts.final_strategy_protocol import PromptFingerprint


def prompt_fingerprint(
    *,
    prompt_name: str,
    prompt_version: str,
    system_prompt: str,
    user_template: str,
    output_schema_version: str = "2.0",
    context_policy_version: str = "2.0",
) -> PromptFingerprint:
    system_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    template_hash = hashlib.sha256(user_template.encode("utf-8")).hexdigest()
    material = json.dumps(
        {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "system_prompt_hash": system_hash,
            "user_template_hash": template_hash,
            "output_schema_version": output_schema_version,
            "context_policy_version": context_policy_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return PromptFingerprint(
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        system_prompt_hash=system_hash,
        user_template_hash=template_hash,
        output_schema_version=output_schema_version,
        context_policy_version=context_policy_version,
        fingerprint=hashlib.sha256(material.encode("utf-8")).hexdigest(),
    )


__all__ = ["prompt_fingerprint"]
