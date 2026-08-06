from __future__ import annotations

import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_HTTP_STATUS_PATTERN = re.compile(
    r"\b(?:http(?:error)?(?:\s+status)?|status(?:\s+code)?|failed\s+with|returned)"
    r"\D{0,12}(4\d{2}|5\d{2})\b",
    re.IGNORECASE,
)
_MISSING = object()
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {"ConnectTimeout", "ConnectionError", "ReadTimeout", "Timeout"}
)
_ATTEMPT_ATTRIBUTE = "_kaggle_request_attempt"
_MAX_ATTEMPTS_ATTRIBUTE = "_kaggle_request_max_attempts"


@dataclass(frozen=True, slots=True)
class UnpackedListResponse:
    items: list[Any]
    next_page_token: str | None
    wrapped: bool


class KaggleRequestPolicy:
    """Bound retries and pace Kaggle calls without coupling to a client SDK."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        min_interval_seconds: float = 0.5,
        jitter_fraction: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if (
            min(
                base_delay_seconds,
                max_delay_seconds,
                min_interval_seconds,
                jitter_fraction,
            )
            < 0
        ):
            raise ValueError("request policy delays cannot be negative")
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.min_interval_seconds = min_interval_seconds
        self.jitter_fraction = jitter_fraction
        self._sleep = sleep
        self._monotonic = monotonic
        self._random_value = random_value
        self._lock = threading.Lock()
        self._last_started_at: float | None = None

    def call(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            self._wait_for_slot()
            try:
                return operation()
            except Exception as exc:
                if attempt >= self.max_attempts or not is_retryable_kaggle_error(exc):
                    _annotate_attempt(exc, attempt, self.max_attempts)
                    raise
                self._sleep(self._retry_delay(exc, attempt))
        raise AssertionError("unreachable retry loop")

    def _wait_for_slot(self) -> None:
        with self._lock:
            now = self._monotonic()
            if self._last_started_at is not None:
                wait_seconds = self._last_started_at + self.min_interval_seconds - now
                if wait_seconds > 0:
                    self._sleep(wait_seconds)
                    now = self._monotonic()
            self._last_started_at = now

    def _retry_delay(self, exc: BaseException, attempt: int) -> float:
        retry_after = extract_retry_after(exc)
        if retry_after is not None:
            return min(retry_after, self.max_delay_seconds)
        exponential = self.base_delay_seconds * (2 ** (attempt - 1))
        jitter = exponential * self.jitter_fraction * self._random_value()
        return min(exponential + jitter, self.max_delay_seconds)


GLOBAL_KAGGLE_POLICY = KaggleRequestPolicy(
    max_attempts=6,
    min_interval_seconds=1.5,
    jitter_fraction=0.2,
)


def create_kaggle_api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def unpack_list_response(response: Any, collection_name: str) -> UnpackedListResponse:
    """Normalize Kaggle 1.x lists and Kaggle 2.x response envelopes."""
    if response is None:
        return UnpackedListResponse([], None, False)
    if isinstance(response, list | tuple):
        return UnpackedListResponse(list(response), None, False)

    raw_items = _response_field(response, collection_name)
    if raw_items is _MISSING:
        return UnpackedListResponse([], None, False)

    if raw_items is None:
        items: list[Any] = []
    elif isinstance(raw_items, list | tuple):
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
        _safe_attribute(exc, "code"),
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


def extract_retry_after(exc: BaseException) -> float | None:
    response = _safe_attribute(exc, "response")
    http_response = _safe_attribute(exc, "http_resp")
    for candidate in (
        _safe_attribute(response, "headers"),
        _safe_attribute(http_response, "headers"),
        _safe_attribute(exc, "headers"),
    ):
        if not isinstance(candidate, Mapping):
            continue
        raw_value = next(
            (value for key, value in candidate.items() if str(key).lower() == "retry-after"),
            None,
        )
        try:
            seconds = float(raw_value)
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return seconds
    return None


def extract_request_attempt(exc: BaseException) -> tuple[int | None, int | None]:
    attempt = _positive_integer(_safe_attribute(exc, _ATTEMPT_ATTRIBUTE))
    max_attempts = _positive_integer(_safe_attribute(exc, _MAX_ATTEMPTS_ATTRIBUTE))
    return attempt, max_attempts


def is_retryable_kaggle_error(exc: BaseException) -> bool:
    status = extract_http_status(exc)
    if status is not None:
        return status in _RETRYABLE_HTTP_STATUSES
    return isinstance(exc, TimeoutError | ConnectionError) or (
        type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES
    )


def is_forbidden(exc: BaseException) -> bool:
    return extract_http_status(exc) == 403


def _annotate_attempt(exc: BaseException, attempt: int, max_attempts: int) -> None:
    try:
        setattr(exc, _ATTEMPT_ATTRIBUTE, attempt)
        setattr(exc, _MAX_ATTEMPTS_ATTRIBUTE, max_attempts)
    except Exception:
        return


def _safe_attribute(value: Any, name: str) -> Any:
    if value is None:
        return None
    try:
        return getattr(value, name)
    except Exception:
        return None


def _response_field(response: Any, *names: str) -> Any:
    if isinstance(response, dict):
        normalized = {_normalize_response_key(str(key)): value for key, value in response.items()}
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


def _positive_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
