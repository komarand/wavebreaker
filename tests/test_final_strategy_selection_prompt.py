from __future__ import annotations

from kaggle_researcher.contracts.final_strategy_protocol import StrategySelectionDraft
from kaggle_researcher.reasoning.prompts.final_strategy_selection_v2 import (
    FINAL_STRATEGY_SELECTION_PROMPT_VERSION,
    FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT,
)
from kaggle_researcher.reasoning.prompts.final_strategy_common import prompt_fingerprint


def test_selection_prompt_is_versioned_compact_and_schema_aligned() -> None:
    schema = StrategySelectionDraft.model_json_schema()
    rendered = str(schema)
    assert FINAL_STRATEGY_SELECTION_PROMPT_VERSION == "2.0"
    assert "do not write the final report" in FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT.casefold()
    assert "smallest safe" in FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT.casefold()
    assert "source -> hypothesis -> eda -> strategy" in FINAL_STRATEGY_SELECTION_SYSTEM_PROMPT.casefold()
    assert "client_action_key" in rendered
    assert "support_refs" not in rendered
    assert "evidence_origin" not in rendered
    assert "narrative" not in rendered
    assert "action_id" not in schema["$defs"]["SelectedActionDraft"]["properties"]


def test_prompt_fingerprint_is_deterministic_and_secret_independent(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first-secret")
    first = prompt_fingerprint(
        prompt_name="selection", prompt_version="2.0", system_prompt="system",
        user_template="template",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "second-secret")
    second = prompt_fingerprint(
        prompt_name="selection", prompt_version="2.0", system_prompt="system",
        user_template="template",
    )
    assert first == second
    assert "secret" not in first.model_dump_json()
