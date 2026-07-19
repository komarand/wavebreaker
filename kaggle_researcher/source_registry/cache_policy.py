from __future__ import annotations

from typing import Any

from kaggle_researcher.source_registry.schemas import CacheDecision, CachePolicy


def decide_cache_action(
    stage: str,
    existing_record: Any | None,
    current_input_hash: str,
    current_processor_fingerprint: str,
    policy: CachePolicy,
    *,
    source_id: str | None = None,
    version_id: Any | None = None,
) -> CacheDecision:
    if policy.rebuilds(stage):
        return CacheDecision(
            stage=stage,
            decision="forced_rebuild",
            source_id=source_id,
            version_id=version_id,
            artifact_id=getattr(existing_record, "artifact_id", None),
            embedding_id=getattr(existing_record, "embedding_id", None),
            reason="Cache reuse was bypassed by the current rebuild policy.",
            previous_fingerprint=_record_fingerprint(existing_record),
            current_fingerprint=current_processor_fingerprint,
        )
    if existing_record is None:
        return CacheDecision(
            stage=stage,
            decision="cache_miss",
            source_id=source_id,
            version_id=version_id,
            reason="No compatible cache record exists.",
            current_fingerprint=current_processor_fingerprint,
        )
    previous_fingerprint = _record_fingerprint(existing_record)
    if not previous_fingerprint or previous_fingerprint.startswith("legacy:unknown"):
        return CacheDecision(
            stage=stage,
            decision="processor_changed",
            source_id=source_id,
            version_id=version_id,
            reason="The cached record has no compatible processor fingerprint.",
            previous_fingerprint=previous_fingerprint,
            current_fingerprint=current_processor_fingerprint,
        )
    if previous_fingerprint != current_processor_fingerprint:
        return CacheDecision(
            stage=stage,
            decision="processor_changed",
            source_id=source_id,
            version_id=version_id,
            reason="Processor fingerprint changed.",
            previous_fingerprint=previous_fingerprint,
            current_fingerprint=current_processor_fingerprint,
        )
    if getattr(existing_record, "input_hash", None) != current_input_hash:
        return CacheDecision(
            stage=stage,
            decision="input_changed",
            source_id=source_id,
            version_id=version_id,
            reason="Processor input hash changed.",
            previous_fingerprint=previous_fingerprint,
            current_fingerprint=current_processor_fingerprint,
        )
    return CacheDecision(
        stage=stage,
        decision="cache_hit",
        source_id=source_id,
        version_id=version_id,
        artifact_id=getattr(existing_record, "artifact_id", None),
        embedding_id=getattr(existing_record, "embedding_id", None),
        reason="Version, input hash, and processor fingerprint match.",
        previous_fingerprint=previous_fingerprint,
        current_fingerprint=current_processor_fingerprint,
    )


def _record_fingerprint(record: Any | None) -> str | None:
    if record is None:
        return None
    return getattr(record, "processor_fingerprint", None) or getattr(record, "embedding_fingerprint", None)
