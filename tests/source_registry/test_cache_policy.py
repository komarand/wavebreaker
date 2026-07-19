from types import SimpleNamespace

from kaggle_researcher.source_registry.cache_policy import decide_cache_action
from kaggle_researcher.source_registry.schemas import CachePolicy


def test_cache_decisions_are_deterministic() -> None:
    record = SimpleNamespace(input_hash="input", processor_fingerprint="fp", artifact_id=None)
    policy = CachePolicy()
    assert decide_cache_action("summary", record, "input", "fp", policy).decision == "cache_hit"
    assert decide_cache_action("summary", record, "other", "fp", policy).decision == "input_changed"
    assert decide_cache_action("summary", record, "input", "other", policy).decision == "processor_changed"
    assert decide_cache_action("summary", record, "input", "fp", CachePolicy(rebuild_artifacts={"summaries"})).decision == "forced_rebuild"
