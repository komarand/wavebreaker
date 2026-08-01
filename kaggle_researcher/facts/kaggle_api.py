from __future__ import annotations

import re
from typing import Any


_HTTP_STATUS_PATTERN = re.compile(r"(?<!\d)(401|403|404|429|5\d{2})(?!\d)")


def create_kaggle_api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


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


def _coerce_http_status(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None
