import json

from kaggle_researcher.source_registry.hashing import compute_content_hashes


def test_line_endings_normalize_but_raw_hashes_differ() -> None:
    crlf = compute_content_hashes("A\r\nB\r\n")
    lf = compute_content_hashes("A\nB\n")
    assert crlf.raw_hash != lf.raw_hash
    assert crlf.normalized_hash == lf.normalized_hash
    assert compute_content_hashes("Code").normalized_hash != compute_content_hashes("code").normalized_hash


def test_notebook_output_ignoring_is_explicit_and_source_changes_remain_visible() -> None:
    first = {"cells": [{"cell_type": "code", "source": ["x=1"], "execution_count": 1, "outputs": [{"text": "a"}]}]}
    second = {"cells": [{"cell_type": "code", "source": ["x=1"], "execution_count": 2, "outputs": [{"text": "b"}]}]}
    changed = {"cells": [{"cell_type": "code", "source": ["x=2"], "execution_count": 2, "outputs": []}]}
    kwargs = {"content_type": "notebook", "policy_version": "notebook-v1", "ignore_notebook_outputs": True}
    assert compute_content_hashes(json.dumps(first), **kwargs).normalized_hash == compute_content_hashes(json.dumps(second), **kwargs).normalized_hash
    assert compute_content_hashes(json.dumps(first), **kwargs).normalized_hash != compute_content_hashes(json.dumps(changed), **kwargs).normalized_hash
