from kaggle_researcher.source_registry.fingerprints import (
    build_embedding_fingerprint,
    build_parser_fingerprint,
    build_summary_fingerprint,
)


def test_fingerprints_are_stable_secret_free_and_stage_scoped() -> None:
    first = build_parser_fingerprint(extra={"b": 2, "a": 1}, api_key="one")
    second = build_parser_fingerprint(extra={"a": 1, "b": 2}, api_key="two")
    assert first.fingerprint == second.fingerprint
    assert "api_key" not in first.configuration
    assert build_summary_fingerprint(model="m", prompt="a").fingerprint != build_summary_fingerprint(model="m", prompt="b").fingerprint
    assert build_embedding_fingerprint(model="a", dimension=2).fingerprint != build_embedding_fingerprint(model="b", dimension=2).fingerprint
