from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_HTTP_STATUS_PATTERN = re.compile(r"(?<!\d)(401|403|404|429|5\d{2})(?!\d)")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class UnpackedListResponse:
    items: list[Any]
    next_page_token: str | None
    wrapped: bool


def create_kaggle_api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def unpack_list_response(response: Any, collection_name: str) -> UnpackedListResponse:
    """Normalize Kaggle 1.x lists and Kaggle 2.x response envelopes."""
    if response is None:
        return UnpackedListResponse([], None, False)
    if isinstance(response, (list, tuple)):
        return UnpackedListResponse(list(response), None, False)

    raw_items = _response_field(response, collection_name)
    if raw_items is _MISSING:
        return UnpackedListResponse([], None, False)

    if raw_items is None:
        items: list[Any] = []
    elif isinstance(raw_items, (list, tuple)):
        items = list(raw_items)
    else:
        try:
            items = list(raw_items)
        except TypeError:
            items = []

    raw_token = _response_field(response, "next_page_token", "nextPageToken")
    token = None
    if raw_token is not _MISSING and raw_token is not None:
        normalized_token = str(raw_token).strip()
        token = normalized_token or None
    return UnpackedListResponse(items, token, True)


def extract_http_status(exc: BaseException) -> int | None:
    candidates = (
        _safe_attribute(exc, "status"),
        _safe_attribute(_safe_attribute(exc, "http_resp"), "status"),
        _safe_attribute(_safe_attribute(exc, "response"), "status_code"),
        _safe_attribute(_safe_attribute(exc, "response"), "status"),
    )
    for candidate in candidates:
        status = _coerce_http_status(candidate)
        if status is not None:
            return status

    match = _HTTP_STATUS_PATTERN.search(str(exc))
    return int(match.group(1)) if match is not None else None


def is_forbidden(exc: BaseException) -> bool:
    return extract_http_status(exc) == 403


def _safe_attribute(value: Any, name: str) -> Any:
    if value is None:
        return None
    try:
        return getattr(value, name)
    except Exception:
        return None


def _response_field(response: Any, *names: str) -> Any:
    if isinstance(response, dict):
        normalized = {
            _normalize_response_key(str(key)): value for key, value in response.items()
        }
        for name in names:
            key = _normalize_response_key(name)
            if key in normalized:
                return normalized[key]
        return _MISSING

    for name in names:
        try:
            return getattr(response, name)
        except (AttributeError, TypeError):
            continue
        except Exception:
            continue
    return _MISSING


def _normalize_response_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _coerce_http_status(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None
