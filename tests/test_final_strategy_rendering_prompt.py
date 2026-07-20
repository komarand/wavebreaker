from __future__ import annotations

from kaggle_researcher.contracts.final_strategy_protocol import StrategyRenderingDraft
from kaggle_researcher.reasoning.prompts.final_strategy_rendering_v2 import (
    FINAL_STRATEGY_RENDERING_PROMPT_VERSION,
    FINAL_STRATEGY_RENDERING_SYSTEM_PROMPT,
)


def test_rendering_prompt_is_wording_only_and_hash_bound() -> None:
    properties = StrategyRenderingDraft.model_json_schema()["properties"]
    assert FINAL_STRATEGY_RENDERING_PROMPT_VERSION == "2.0"
    assert "do not make strategic decisions" in FINAL_STRATEGY_RENDERING_SYSTEM_PROMPT.casefold()
    assert {"skeleton_id", "skeleton_hash", "action_wording", "experiment_wording"} <= set(properties)
    assert not {"actions", "experiments", "evidence_refs", "dependencies"} & set(properties)
