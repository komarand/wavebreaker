from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from kaggle_researcher.agents import kaggle_agent


_REF_NAMES = ("ref", "kernelRef", "kernel_ref", "id")


def list_competition_notebooks(slug: str, max_notebooks: int) -> list[dict[str, Any]]:
    if max_notebooks <= 0:
        return []

    api = kaggle_agent.KaggleApi()
    api.authenticate()
    page_size = min(max_notebooks, 100)
    page = 1
    notebooks_by_ref: dict[str, dict[str, Any]] = {}

    while len(notebooks_by_ref) < max_notebooks:
        kernels = list(
            api.kernels_list(
                competition=slug,
                page=page,
                page_size=page_size,
                sort_by="voteCount",
                language="python",
            )
            or []
        )
        for kernel in kernels:
            normalized = _normalize_kernel(kernel)
            if normalized is None:
                continue
            ref = normalized["ref"]
            existing = notebooks_by_ref.get(ref)
            if existing is None or _votes_sort_key(normalized) > _votes_sort_key(existing):
                notebooks_by_ref[ref] = normalized

        if len(kernels) < page_size or len(notebooks_by_ref) >= max_notebooks:
            break
        page += 1

    return sorted(
        notebooks_by_ref.values(),
        key=lambda notebook: (-_votes_sort_key(notebook), notebook["ref"]),
    )[:max_notebooks]


def pull_notebook(kernel_ref: str, dest: Path) -> Path | None:
    if not kernel_ref:
        return None

    try:
        dest.mkdir(parents=True, exist_ok=True)
        before = _notebook_file_state(dest)
        api = kaggle_agent.KaggleApi()
        api.authenticate()
        _pull_kernel(api, kernel_ref, dest)
        after = _notebook_file_state(dest)
    except Exception:
        return None

    changed_paths = sorted(
        path
        for path, state in after.items()
        if path not in before or before[path] != state
    )
    return changed_paths[0] if changed_paths else None


def _normalize_kernel(kernel: Any) -> dict[str, Any] | None:
    ref = _get_kernel_value(kernel, *_REF_NAMES)
    if not isinstance(ref, str) or not ref:
        return None

    return {
        "ref": ref,
        "title": _get_kernel_value(kernel, "title", "kernelTitle", "kernel_title"),
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
        "totalVotes": _parse_votes(
            _get_kernel_value(
                kernel,
                "totalVotes",
                "total_votes",
                "votes",
                "voteCount",
                "vote_count",
            )
        ),
        "publicScore": _parse_public_score(
            _get_kernel_value(
                kernel,
                "publicScore",
                "public_score",
            )
        ),
        "lastRunTime": _get_kernel_value(
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


def _parse_votes(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        votes = int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return votes if votes >= 0 else None


def _votes_sort_key(notebook: dict[str, Any]) -> int:
    votes = notebook.get("totalVotes")
    return votes if isinstance(votes, int) else -1


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
        normalized_values = {
            _normalize_key(str(key)): value for key, value in kernel.items()
        }
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
