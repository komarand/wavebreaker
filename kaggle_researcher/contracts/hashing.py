from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math
from collections.abc import Mapping, Set
from typing import Any

from pydantic import BaseModel

from kaggle_researcher.contracts.errors import (
    ContractCanonicalizationError,
    ContractIssue,
)


CANONICAL_HASH_POLICY_VERSION = "1.0"


def canonical_contract_bytes(
    model_or_data: Any,
    *,
    policy_version: str,
) -> bytes:
    """Return deterministic UTF-8 JSON bytes under an explicit hash policy.

    Policy 1.0 includes ``None`` fields, sorts mapping keys, preserves list/tuple
    order, sorts sets by their canonical JSON encoding, normalizes aware
    datetimes to UTC, and rejects unsupported or non-finite values.
    """
    if policy_version != CANONICAL_HASH_POLICY_VERSION:
        raise ContractCanonicalizationError(
            f"Unsupported canonical hash policy version: {policy_version!r}",
            issues=(ContractIssue(
                "policy_version", policy_version, CANONICAL_HASH_POLICY_VERSION,
                "unsupported canonical hash policy",
            ),),
            contract="canonical_hash_policy",
        )
    value = (
        model_or_data.model_dump(mode="python", exclude_none=False)
        if isinstance(model_or_data, BaseModel)
        else model_or_data
    )
    normalized = _canonicalize(value, path="$", policy_version=policy_version)
    try:
        rendered = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractCanonicalizationError(
            "Canonical JSON serialization failed",
            contract="canonical_hash_policy",
        ) from exc
    return rendered.encode("utf-8")


def sha256_contract(
    model_or_data: Any,
    *,
    policy_version: str = CANONICAL_HASH_POLICY_VERSION,
) -> str:
    return hashlib.sha256(
        canonical_contract_bytes(model_or_data, policy_version=policy_version)
    ).hexdigest()


# Compatibility name used by the Prompt 0 specification tests.
canonical_contract_hash = sha256_contract


def _canonicalize(value: Any, *, path: str, policy_version: str) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(
            value.model_dump(mode="python", exclude_none=False),
            path=path,
            policy_version=policy_version,
        )
    if isinstance(value, Enum):
        return _canonicalize(value.value, path=path, policy_version=policy_version)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, value, "finite JSON number", "non-finite numbers are forbidden")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            _fail(path, value, "timezone-aware datetime", "naive datetime is ambiguous")
        utc = value.astimezone(timezone.utc)
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, key, "string mapping key", "non-string keys are forbidden")
            normalized[key] = _canonicalize(
                item, path=f"{path}.{key}", policy_version=policy_version
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, path=f"{path}[{index}]", policy_version=policy_version)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        items = [
            _canonicalize(item, path=f"{path}{{item}}", policy_version=policy_version)
            for item in value
        ]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
            ),
        )
    _fail(path, value, "canonical JSON value", f"unsupported type {type(value).__name__}")


def _fail(path: str, value: Any, expected: str, reason: str) -> None:
    raise ContractCanonicalizationError(
        "Contract canonicalization failed",
        issues=(ContractIssue(path, _safe_value(value), expected, reason),),
        contract="canonical_hash_policy",
    )


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


__all__ = [
    "CANONICAL_HASH_POLICY_VERSION",
    "canonical_contract_bytes",
    "canonical_contract_hash",
    "sha256_contract",
]
