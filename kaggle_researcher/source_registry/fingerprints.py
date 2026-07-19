from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from kaggle_researcher.source_registry.hashing import sha256_text
from kaggle_researcher.source_registry.schemas import ProcessorFingerprint


_SECRET_PARTS = ("api_key", "apikey", "token", "secret", "password", "credential", "authorization")


def canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_processor_fingerprint(
    processor_name: str,
    processor_version: str,
    configuration: Mapping[str, Any] | None = None,
    dependency_versions: Mapping[str, Any] | None = None,
) -> ProcessorFingerprint:
    name = processor_name.strip()
    version = processor_version.strip()
    if not name:
        raise ValueError("processor_name must not be empty")
    if not version:
        raise ValueError("processor_version must not be empty")
    safe_configuration = _strip_secrets(dict(configuration or {}))
    payload = {
        "configuration": safe_configuration,
        "processor_name": name,
        "processor_version": version,
    }
    if dependency_versions is not None:
        payload["dependency_versions"] = _strip_secrets(dict(dependency_versions))
    return ProcessorFingerprint(
        processor_name=name,
        processor_version=version,
        configuration=safe_configuration,
        fingerprint=sha256_text(canonical_json(payload)),
    )


def build_parser_fingerprint(
    *,
    processor_name: str = "pdf_parser",
    processor_version: str = "1.0",
    max_chars: int = 8000,
    page_selection_policy_version: str = "first3-last2-tables-v1",
    table_extraction_policy_version: str = "pdfplumber-v1",
    text_normalization_policy: str = "text-v1",
    dependency_versions: Mapping[str, Any] | None = None,
    **configuration: Any,
) -> ProcessorFingerprint:
    return build_processor_fingerprint(
        processor_name,
        processor_version,
        {
            "max_chars": max_chars,
            "page_selection_policy_version": page_selection_policy_version,
            "table_extraction_policy_version": table_extraction_policy_version,
            "text_normalization_policy": text_normalization_policy,
            **configuration,
        },
        dependency_versions or {},
    )


def build_summary_fingerprint(
    *,
    provider: str = "deepseek",
    model: str,
    prompt_template_version: str = "1.0",
    prompt: str = "",
    model_revision: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    input_selection_policy: str = "parsed-or-raw-v1",
    response_parsing_version: str = "text-v1",
    **configuration: Any,
) -> ProcessorFingerprint:
    return build_processor_fingerprint(
        "source_summarizer",
        "1.0",
        {
            "provider": provider,
            "model": model,
            "model_revision": model_revision,
            "prompt_template_version": prompt_template_version,
            "prompt_hash": sha256_text(prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "input_selection_policy": input_selection_policy,
            "response_parsing_version": response_parsing_version,
            **configuration,
        },
        {},
    )


def build_embedding_fingerprint(
    *,
    model: str,
    dimension: int,
    provider: str = "sentence_transformers",
    model_revision: str | None = None,
    normalize_embeddings: bool = True,
    truncation_policy: str = "model-default-v1",
    max_input_size: int | None = None,
    text_preprocessing_policy: str = "identity-v1",
    input_format_version: str = "text-v1",
    **configuration: Any,
) -> ProcessorFingerprint:
    return build_processor_fingerprint(
        "source_embedder",
        "1.0",
        {
            "provider": provider,
            "model": model,
            "model_revision": model_revision,
            "dimension": dimension,
            "normalize_embeddings": normalize_embeddings,
            "truncation_policy": truncation_policy,
            "max_input_size": max_input_size,
            "text_preprocessing_policy": text_preprocessing_policy,
            "input_format_version": input_format_version,
            **configuration,
        },
        {},
    )


def build_static_analysis_fingerprint(
    *,
    analyzer_name: str,
    analyzer_version: str,
    enabled_detectors: list[str] | tuple[str, ...] = (),
    rules_version: str = "1.0",
    truncation_policy: str = "bounded-v1",
    **configuration: Any,
) -> ProcessorFingerprint:
    return build_processor_fingerprint(
        analyzer_name,
        analyzer_version,
        {
            "enabled_detectors": sorted(enabled_detectors),
            "rules_version": rules_version,
            "truncation_policy": truncation_policy,
            **configuration,
        },
        {},
    )


def build_search_request_fingerprint(
    *,
    provider: str,
    normalized_query: str,
    result_limit: int,
    sort_mode: str | None = None,
    filters: Mapping[str, Any] | None = None,
    request_policy_version: str = "1.0",
    **configuration: Any,
) -> ProcessorFingerprint:
    return build_processor_fingerprint(
        f"{provider}_search",
        request_policy_version,
        {
            "provider": provider,
            "normalized_query": normalized_query,
            "result_limit": result_limit,
            "sort_mode": sort_mode,
            "filters": dict(filters or {}),
            **configuration,
        },
        {},
    )


def _strip_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_secrets(item)
            for key, item in value.items()
            if not any(part in str(key).lower().replace("-", "_") for part in _SECRET_PARTS)
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_strip_secrets(item) for item in value]
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(canonical_items, key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
