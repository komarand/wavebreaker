from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    UnpackedListResponse,
    create_kaggle_api,
    extract_http_status,
    extract_request_attempt,
    unpack_list_response,
)
from kaggle_researcher.facts.kaggle_api import (
    KaggleRequestPolicy as KaggleRequestPolicy,
)

_REF_NAMES = ("ref", "kernelRef", "kernel_ref", "id")
_NOTEBOOK_REQUEST_POLICY = GLOBAL_KAGGLE_POLICY
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NotebookPullResult:
    path: Path | None
    http_status: int | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    error_type: str | None = None

    def failure_detail(self) -> str | None:
        if self.error_type is None:
            return None
        status = str(self.http_status) if self.http_status is not None else "unknown"
        attempt = str(self.attempt) if self.attempt is not None else "?"
        maximum = str(self.max_attempts) if self.max_attempts is not None else "?"
        return f"HTTP {status}, attempt {attempt}/{maximum}"


def list_competition_notebooks(
    slug: str,
    max_notebooks: int,
    api: Any | None = None,
) -> list[dict[str, Any]]:
    if max_notebooks <= 0:
        return []

    if api is None:
        api = create_kaggle_api()
    page_size = min(max_notebooks, 100)
    page = 1
    page_token: str | None = None
    seen_tokens: set[str] = set()
    notebooks_by_ref: dict[str, dict[str, Any]] = {}

    while len(notebooks_by_ref) < max_notebooks:
        unpacked = _NOTEBOOK_REQUEST_POLICY.call(
            lambda current_page=page, current_token=page_token: _list_kernels_page(
                api,
                slug=slug,
                page=current_page,
                page_size=page_size,
                page_token=current_token,
            )
        )
        kernels = unpacked.items
        for kernel in kernels:
            normalized = _normalize_kernel(kernel)
            if normalized is None:
                continue
            ref = normalized["ref"]
            existing = notebooks_by_ref.get(ref)
            if existing is None or _votes_sort_key(normalized) > _votes_sort_key(existing):
                notebooks_by_ref[ref] = normalized

        if len(notebooks_by_ref) >= max_notebooks:
            break

        next_page_token = unpacked.next_page_token
        if next_page_token:
            if next_page_token in seen_tokens or not kernels:
                break
            seen_tokens.add(next_page_token)
            page_token = next_page_token
            page += 1
            continue

        if unpacked.wrapped or len(kernels) < page_size:
            break
        page += 1

    return sorted(
        notebooks_by_ref.values(),
        key=lambda notebook: (-_votes_sort_key(notebook), notebook["ref"]),
    )[:max_notebooks]


def _list_kernels_page(
    api: Any,
    *,
    slug: str,
    page: int,
    page_size: int,
    page_token: str | None,
) -> UnpackedListResponse:
    common_kwargs = {
        "competition": slug,
        "page_size": page_size,
        "sort_by": "voteCount",
        "language": "python",
        "kernel_type": "notebook",
    }
    list_with_response = getattr(api, "kernels_list_with_response", None)
    if callable(list_with_response):
        response = list_with_response(
            **common_kwargs,
            page_token=page_token,
        )
        return unpack_list_response(response, "kernels")

    if page_token is None:
        response = api.kernels_list(**common_kwargs, page=page)
    else:
        try:
            response = api.kernels_list(**common_kwargs, page_token=page_token)
        except TypeError:
            response = api.kernels_list(**common_kwargs, page=page)
    return unpack_list_response(response, "kernels")


def pull_notebook(
    kernel_ref: str,
    dest: Path,
    api: Any | None = None,
) -> Path | None:
    return pull_notebook_with_diagnostics(kernel_ref, dest, api).path


def pull_notebook_with_diagnostics(
    kernel_ref: str,
    dest: Path,
    api: Any | None = None,
) -> NotebookPullResult:
    if not kernel_ref:
        return NotebookPullResult(path=None)

    try:
        dest.mkdir(parents=True, exist_ok=True)
        before = _notebook_file_state(dest)
        if api is None:
            api = create_kaggle_api()
        _NOTEBOOK_REQUEST_POLICY.call(lambda: _pull_kernel(api, kernel_ref, dest))
        after = _notebook_file_state(dest)
    except Exception as exc:
        attempt, max_attempts = extract_request_attempt(exc)
        result = NotebookPullResult(
            path=None,
            http_status=extract_http_status(exc),
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=type(exc).__name__,
        )
        logger.warning(
            "Failed to pull Kaggle notebook %s (%s, %s)",
            kernel_ref,
            type(exc).__name__,
            result.failure_detail(),
        )
        return result

    changed_paths = sorted(
        path for path, state in after.items() if path not in before or before[path] != state
    )
    return NotebookPullResult(path=changed_paths[0] if changed_paths else None)


def _normalize_kernel(kernel: Any) -> dict[str, Any] | None:
    ref = _get_kernel_value(kernel, *_REF_NAMES)
    if not isinstance(ref, str) or not ref.strip():
        logger.warning("Skipping Kaggle notebook entry without a ref")
        return None
    ref = ref.strip()

    raw_title = _get_kernel_value(kernel, "title", "kernelTitle", "kernel_title")
    title = raw_title.strip() if isinstance(raw_title, str) else ""

    return {
        "ref": ref,
        "title": title or ref,
        "author": _normalize_author(
            _get_kernel_value(
                kernel,
                "author",
                "authorName",
                "author_name",
                "ownerName",
                "owner_name",
            )
        ),
        "votes": _parse_votes(
            _get_kernel_value(
                kernel,
                "totalVotes",
                "total_votes",
                "votes",
                "voteCount",
                "vote_count",
            )
        ),
        "public_score": _parse_public_score(
            _get_kernel_value(
                kernel,
                "publicScore",
                "public_score",
            )
        ),
        "last_run": _get_kernel_value(
            kernel,
            "lastRunTime",
            "last_run_time",
            "lastRun",
            "last_run",
        ),
    }


def _normalize_author(author: Any) -> str | None:
    if isinstance(author, str):
        return author or None
    value = _get_kernel_value(
        author,
        "name",
        "username",
        "userName",
        "user_name",
        "displayName",
        "display_name",
    )
    return value if isinstance(value, str) and value else None


def _parse_public_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _parse_votes(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        votes = int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0
    return votes if votes >= 0 else 0


def _votes_sort_key(notebook: dict[str, Any]) -> int:
    votes = notebook.get("votes")
    return votes if isinstance(votes, int) else 0


def _pull_kernel(api: Any, kernel_ref: str, dest: Path) -> None:
    try:
        api.kernels_pull(
            kernel_ref,
            path=str(dest),
            metadata=True,
            quiet=True,
        )
    except TypeError:
        try:
            api.kernels_pull(
                kernel_ref,
                path=str(dest),
                metadata=True,
            )
        except TypeError:
            api.kernels_pull(
                kernel_ref,
                path=str(dest),
            )


def _notebook_file_state(dest: Path) -> dict[Path, tuple[int, int]]:
    state: dict[Path, tuple[int, int]] = {}
    for path in dest.rglob("*.ipynb"):
        if not path.is_file():
            continue
        stat = path.stat()
        state[path] = (stat.st_size, stat.st_mtime_ns)
    return state


def _get_kernel_value(kernel: Any, *names: str) -> Any:
    if kernel is None:
        return None
    if isinstance(kernel, dict):
        normalized_values = {_normalize_key(str(key)): value for key, value in kernel.items()}
        for name in names:
            value = normalized_values.get(_normalize_key(name))
            if value is not None and value != "":
                return value
        return None

    for name in names:
        try:
            value = getattr(kernel, name)
        except Exception:
            continue
        if value is not None and value != "":
            return value

    normalized_names = {_normalize_key(name) for name in names}
    for attribute_name in dir(kernel):
        if _normalize_key(attribute_name) not in normalized_names:
            continue
        try:
            value = getattr(kernel, attribute_name)
        except Exception:
            continue
        if value is not None and value != "":
            return value
    return None


def _normalize_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())
