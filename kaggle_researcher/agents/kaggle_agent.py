from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kaggle_researcher.logging_utils import get_logger
from kaggle_researcher.schemas import SourceDocument


logger = get_logger(__name__)


def search_notebooks(queries: list[str], max_notebooks: int) -> list[dict[str, Any]]:
    if max_notebooks <= 0:
        return []

    notebooks_by_ref: dict[str, dict[str, Any]] = {}

    for query in queries:
        rows = _run_kaggle_kernel_search(query=query, max_notebooks=max_notebooks)
        for row in rows:
            normalized = _normalize_search_row(row)
            kernel_ref = normalized.get("kernel_ref")
            if not kernel_ref:
                continue

            existing = notebooks_by_ref.get(kernel_ref)
            if existing is None or normalized["total_votes"] > existing.get("total_votes", 0):
                notebooks_by_ref[kernel_ref] = normalized

    return sorted(
        notebooks_by_ref.values(),
        key=lambda item: item.get("total_votes", 0),
        reverse=True,
    )[:max_notebooks]


def get_notebook_content(kernel_ref: str, max_chars: int = 8000) -> str:
    if not kernel_ref:
        logger.warning("Cannot download Kaggle notebook without kernel_ref")
        return ""

    with tempfile.TemporaryDirectory() as temp_dir:
        result = subprocess.run(
            ["kaggle", "kernels", "pull", kernel_ref, "-p", temp_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("Failed to download Kaggle notebook %s: %s", kernel_ref, result.stderr.strip())
            return ""

        notebook_paths = sorted(Path(temp_dir).rglob("*.ipynb"))
        if not notebook_paths:
            logger.warning("Downloaded Kaggle notebook %s did not contain an .ipynb file", kernel_ref)
            return ""

        try:
            content = _extract_notebook_text(notebook_paths[0])
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse Kaggle notebook %s: %s", kernel_ref, exc)
            return ""

    return content[:max_chars]


def build_kaggle_documents(
    raw_results: list[dict[str, Any]],
    competition_id: str,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    seen_refs: set[str] = set()

    for raw_result in sorted(
        raw_results,
        key=lambda item: _parse_votes(item.get("total_votes", item.get("totalVotes", 0))),
        reverse=True,
    ):
        kernel_ref = _get_first(raw_result, "kernel_ref", "ref", "kernelRef")
        if not kernel_ref or kernel_ref in seen_refs:
            continue
        seen_refs.add(kernel_ref)

        content = str(raw_result.get("content") or "")
        if not content:
            content = get_notebook_content(kernel_ref)

        title = str(_get_first(raw_result, "title", "kernelTitle") or kernel_ref)
        url = _get_first(raw_result, "url", "kernelUrl") or f"https://www.kaggle.com/code/{kernel_ref}"
        total_votes = _parse_votes(_get_first(raw_result, "total_votes", "totalVotes", "votes") or 0)

        documents.append(
            SourceDocument(
                id=_document_id(competition_id=competition_id, kernel_ref=kernel_ref),
                competition_id=competition_id,
                source="kaggle",
                title=title,
                url=str(url),
                content=content,
                metadata={
                    "kernel_ref": kernel_ref,
                    "total_votes": total_votes,
                },
            )
        )

    return documents


def _run_kaggle_kernel_search(query: str, max_notebooks: int) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "kaggle",
            "kernels",
            "list",
            "--search",
            query,
            "--sort-by",
            "voteCount",
            "--page-size",
            str(max_notebooks),
            "--csv",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.warning("Kaggle notebook search failed for query %r: %s", query, result.stderr.strip())
        return []

    return list(csv.DictReader(result.stdout.splitlines()))


def _normalize_search_row(row: dict[str, Any]) -> dict[str, Any]:
    kernel_ref = _get_first(row, "ref", "kernel_ref", "kernelRef", "kernel")
    total_votes = _parse_votes(_get_first(row, "totalVotes", "total_votes", "votes", "voteCount") or 0)
    title = _get_first(row, "title", "kernelTitle") or kernel_ref or "Untitled Kaggle notebook"
    url = _get_first(row, "url", "kernelUrl") or (
        f"https://www.kaggle.com/code/{kernel_ref}" if kernel_ref else None
    )

    return {
        "kernel_ref": kernel_ref,
        "ref": kernel_ref,
        "title": title,
        "url": url,
        "total_votes": total_votes,
        "source": "kaggle",
    }


def _extract_notebook_text(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts: list[str] = []

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        source = _source_to_text(cell.get("source", ""))

        if cell_type == "markdown":
            if source.strip():
                parts.append(source.strip())
        elif cell_type == "code":
            code_snippet = source[:500]
            if code_snippet.strip():
                parts.append(code_snippet)

    return "\n\n".join(parts)


def _source_to_text(source: Any) -> str:
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _get_first(data: dict[str, Any], *keys: str) -> Any:
    lowered = {_normalize_key(key): value for key, value in data.items()}
    for key in keys:
        normalized_key = _normalize_key(key)
        if normalized_key in lowered and lowered[normalized_key] not in {None, ""}:
            return lowered[normalized_key]
    return None


def _normalize_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _parse_votes(value: Any) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return 0


def _document_id(competition_id: str, kernel_ref: str) -> str:
    digest = hashlib.sha1(f"{competition_id}:{kernel_ref}".encode("utf-8")).hexdigest()[:16]
    return f"kaggle-{digest}"
