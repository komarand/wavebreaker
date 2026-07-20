from __future__ import annotations

import json
from typing import Any

from kaggle_researcher.source_registry.hashing import sha256_text
from kaggle_researcher.source_registry.schemas import ProcessorFingerprint


def build_processor_fingerprint(
    processor_name: str,
    processor_version: str,
    configuration: dict[str, Any] | None = None,
) -> ProcessorFingerprint:
    """Build a deterministic fingerprint for a versioned processing step."""
    name = processor_name.strip()
    version = processor_version.strip()
    if not name:
        raise ValueError("processor_name must not be empty")
    if not version:
        raise ValueError("processor_version must not be empty")
    config = configuration or {}
    canonical = json.dumps(
        {"configuration": config, "processor_name": name, "processor_version": version},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return ProcessorFingerprint(
        processor_name=name,
        processor_version=version,
        configuration=config,
        fingerprint=sha256_text(canonical),
    )
