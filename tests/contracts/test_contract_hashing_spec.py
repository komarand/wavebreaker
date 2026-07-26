from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import pytest

from kaggle_researcher.contracts.hashing import canonical_contract_hash

pytestmark = pytest.mark.contract


def _future_hash(value: object) -> str:
    return canonical_contract_hash(value)


def test_canonical_hash_ignores_dictionary_insertion_order() -> None:
    assert _future_hash({"a": 1, "b": 2}) == _future_hash({"b": 2, "a": 1})


def test_canonical_hash_serializes_enums_stably() -> None:
    class Status(str, Enum):
        READY = "ready"
    assert _future_hash({"status": Status.READY}) == _future_hash({"status": "ready"})


def test_canonical_hash_normalizes_equivalent_aware_datetimes() -> None:
    utc = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)
    plus_three = datetime.fromisoformat("2026-07-20T12:00:00+03:00")
    assert _future_hash({"at": utc}) == _future_hash({"at": plus_three})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_canonical_hash_rejects_non_finite_numbers(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _future_hash({"value": invalid})


def test_canonical_hash_changes_for_meaningful_scalar_and_ordered_list_changes() -> None:
    original = {"score": 0.61, "steps": ["baseline", "features"]}
    assert _future_hash(original) != _future_hash({**original, "score": 0.62})
    assert _future_hash(original) != _future_hash({**original, "steps": list(reversed(original["steps"]))})


def test_canonical_hash_is_repeatable_sha256_not_python_hash() -> None:
    value = {"nested": [1, 2, 3]}
    first = _future_hash(value)
    assert first == _future_hash(value)
    assert len(first) == 64
    assert int(first, 16) >= 0
    assert first != str(hash(repr(value)))


def test_canonical_hash_includes_none_and_normalizes_sets_deterministically() -> None:
    assert _future_hash({"optional": None}) != _future_hash({})
    assert _future_hash({"ids": {"b", "a"}}) == _future_hash({"ids": {"a", "b"}})
