from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from kaggle_researcher.logging_utils import get_logger
from kaggle_researcher.schemas import SourceDocument
from kaggle_researcher.source_registry.fingerprints import build_parser_fingerprint


logger = get_logger(__name__)
MAX_NOTEBOOKS = 20
CODE_CELL_MAX_CHARS = 800
NOTEBOOK_TEXT_EXTRACTOR_VERSION = "1.0"


def search_notebooks(
    queries: list[str],
    competition_id: str | None = None,
    max_notebooks: int = MAX_NOTEBOOKS,
) -> list[dict[str, Any]]:
    if max_notebooks <= 0:
        return []

    api = KaggleApi()
    api.authenticate()
    notebooks_by_ref: dict[str, dict[str, Any]] = {}

    if competition_id:
        for kernel in api.kernels_list(
            competition=competition_id,
            page_size=max_notebooks,
            sort_by="voteCount",
            language="python",
        ):
            _add_kernel_result(notebooks_by_ref, kernel, competition_id=competition_id)

    if not notebooks_by_ref:
        for query in queries:
            for kernel in api.kernels_list(
                search=query,
                page_size=max_notebooks,
                sort_by="voteCount",
                language="python",
            ):
                _add_kernel_result(notebooks_by_ref, kernel, competition_id=competition_id)

    return sorted(
        notebooks_by_ref.values(),
        key=lambda item: item.get("total_votes", 0),
        reverse=True,
    )[:max_notebooks]


def get_notebook_content(kernel_ref: str, max_chars: int = 8000) -> str:
    if not kernel_ref:
        raise RuntimeError("Cannot download Kaggle notebook without kernel_ref")

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory(prefix="kaggle_kernel_") as temp_dir:
        temp_path = Path(temp_dir)
        _pull_kernel(api=api, kernel_ref=kernel_ref, temp_path=temp_path)

        notebook_paths = sorted(temp_path.rglob("*.ipynb"))
        if notebook_paths:
            return _extract_notebook_text(notebook_paths[0])[:max_chars]

        python_paths = sorted(temp_path.rglob("*.py"))
        if python_paths:
            return python_paths[0].read_text(encoding="utf-8", errors="replace")[:max_chars]

        downloaded_files = sorted(
            str(path.relative_to(temp_path))
            for path in temp_path.rglob("*")
            if path.is_file()
        )
        raise RuntimeError(
            f"Kaggle pull for {kernel_ref} did not contain .ipynb or .py files. "
            f"Downloaded files: {downloaded_files}"
        )


def _pull_kernel(api: KaggleApi, kernel_ref: str, temp_path: Path) -> None:
    try:
        try:
            api.kernels_pull(kernel_ref, path=str(temp_path), metadata=True, quiet=True)
        except TypeError:
            try:
                api.kernels_pull(kernel_ref, path=str(temp_path), metadata=True)
            except TypeError:
                api.kernels_pull(kernel_ref, path=str(temp_path))
    except Exception as exc:
        raise RuntimeError(f"Kaggle pull failed for {kernel_ref}: {exc}") from exc


def _add_kernel_result(
    notebooks_by_ref: dict[str, dict[str, Any]],
    kernel: Any,
    competition_id: str | None,
) -> None:
    normalized = _normalize_kernel(kernel, competition_id=competition_id)
    kernel_ref = normalized.get("id")
    if not kernel_ref:
        return

    existing = notebooks_by_ref.get(kernel_ref)
    if existing is None or normalized["total_votes"] > existing.get("total_votes", 0):
        notebooks_by_ref[kernel_ref] = normalized


def _normalize_kernel(kernel: Any, competition_id: str | None) -> dict[str, Any]:
    kernel_ref = _get_kernel_value(kernel, "ref", "kernel_ref", "kernelRef", "id")
    title = _get_kernel_value(kernel, "title", "kernelTitle") or kernel_ref or "Untitled Kaggle notebook"
    votes = _parse_votes(
        _get_kernel_value(kernel, "total_votes", "totalVotes", "votes", "voteCount", "totalVoteCount")
    )
    url = _get_kernel_value(kernel, "url", "kernelUrl") or (
        f"https://www.kaggle.com/code/{kernel_ref}" if kernel_ref else None
    )
    raw_metadata = _get_kernel_value(kernel, "metadata")
    version = _get_kernel_value(kernel, "version", "versionNumber", "currentVersionNumber")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata.update({
        "ref": kernel_ref,
        "competition_id": competition_id,
        "source_revision": str(version) if version is not None else None,
        "revision_is_reliable": version is not None,
    })

    return {
        "id": kernel_ref,
        "kernel_ref": kernel_ref,
        "ref": kernel_ref,
        "title": title,
        "url": url,
        "total_votes": votes,
        "source": "kaggle",
        "metadata": metadata,
    }


def _get_kernel_value(kernel: Any, *names: str) -> Any:
    if isinstance(kernel, dict):
        return _get_first(kernel, *names)

    for name in names:
        if hasattr(kernel, name):
            value = getattr(kernel, name)
            if _has_value(value):
                return value

    normalized_names = {_normalize_key(name) for name in names}
    for name in dir(kernel):
        try:
            value = getattr(kernel, name)
        except Exception:
            continue
        if _normalize_key(name) in normalized_names and _has_value(value):
            return value

    return None


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
        kernel_ref = _get_first(raw_result, "id", "kernel_ref", "ref", "kernelRef")
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
                    **dict(raw_result.get("metadata") or {}),
                    "kernel_ref": kernel_ref,
                    "total_votes": total_votes,
                    "parser_fingerprint": kaggle_notebook_parser_fingerprint(),
                },
            )
        )

    return documents


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
            code_snippet = source[:CODE_CELL_MAX_CHARS]
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
        if normalized_key in lowered and _has_value(lowered[normalized_key]):
            return lowered[normalized_key]
    return None


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


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


def kaggle_notebook_parser_fingerprint() -> str:
    return build_parser_fingerprint(
        processor_name="kaggle_notebook_text_extractor",
        processor_version=NOTEBOOK_TEXT_EXTRACTOR_VERSION,
        max_chars=8000,
        cell_source_limit=CODE_CELL_MAX_CHARS,
    ).fingerprint
