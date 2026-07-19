from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from kaggle_researcher.source_registry.schemas import ContentHashes


TEXT_NORMALIZATION_POLICY_VERSION = "text-v1"
NOTEBOOK_NORMALIZATION_POLICY_VERSION = "notebook-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def normalize_text_for_hashing(text: str, policy_version: str = TEXT_NORMALIZATION_POLICY_VERSION) -> str:
    if policy_version != TEXT_NORMALIZATION_POLICY_VERSION:
        raise ValueError(f"Unsupported text normalization policy: {policy_version}")
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalize_notebook_for_hashing(
    notebook: str | bytes | dict[str, Any],
    *,
    ignore_outputs: bool = False,
    policy_version: str = NOTEBOOK_NORMALIZATION_POLICY_VERSION,
) -> str:
    if policy_version != NOTEBOOK_NORMALIZATION_POLICY_VERSION:
        raise ValueError(f"Unsupported notebook normalization policy: {policy_version}")
    if isinstance(notebook, bytes):
        notebook = notebook.decode("utf-8-sig")
    value = json.loads(notebook) if isinstance(notebook, str) else notebook
    if not isinstance(value, dict):
        raise ValueError("Notebook content must be a JSON object")
    normalized = _normalize_notebook_value(value, ignore_outputs=ignore_outputs)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_content_hashes(
    raw_content: str | bytes,
    *,
    content_type: str = "text",
    policy_version: str = TEXT_NORMALIZATION_POLICY_VERSION,
    ignore_notebook_outputs: bool = False,
) -> ContentHashes:
    raw_bytes = raw_content.encode("utf-8") if isinstance(raw_content, str) else bytes(raw_content)
    if content_type in {"notebook", "application/x-ipynb+json"}:
        normalized = normalize_notebook_for_hashing(
            raw_content,
            ignore_outputs=ignore_notebook_outputs,
            policy_version=policy_version,
        )
        normalized_bytes = normalized.encode("utf-8")
    elif content_type == "binary" or (isinstance(raw_content, bytes) and policy_version == "binary-v1"):
        if policy_version != "binary-v1":
            raise ValueError(f"Unsupported binary normalization policy: {policy_version}")
        normalized_bytes = raw_bytes
    else:
        text = raw_content if isinstance(raw_content, str) else raw_bytes.decode("utf-8-sig", errors="replace")
        normalized_bytes = normalize_text_for_hashing(text, policy_version).encode("utf-8")
    return ContentHashes(
        raw_hash=sha256_bytes(raw_bytes),
        normalized_hash=sha256_bytes(normalized_bytes),
        normalization_policy_version=policy_version,
        normalized_size=len(normalized_bytes),
    )


def _normalize_notebook_value(value: dict[str, Any], *, ignore_outputs: bool) -> dict[str, Any]:
    result = dict(value)
    cells: list[Any] = []
    for raw_cell in value.get("cells", []):
        if not isinstance(raw_cell, dict):
            cells.append(raw_cell)
            continue
        cell = dict(raw_cell)
        if ignore_outputs:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
            metadata = cell.get("metadata")
            if isinstance(metadata, dict):
                cell["metadata"] = {
                    key: item for key, item in metadata.items()
                    if key not in {"execution", "collapsed", "scrolled"}
                }
        cells.append(cell)
    result["cells"] = cells
    if ignore_outputs and isinstance(result.get("metadata"), dict):
        result["metadata"] = {
            key: item for key, item in result["metadata"].items()
            if key not in {"widgets", "execution"}
        }
    return result
