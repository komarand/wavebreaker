from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kaggle_researcher.source_registry.schemas import CachePolicy, CacheRunTelemetry


logger = logging.getLogger(__name__)


def build_cache_report(telemetry: CacheRunTelemetry, policy: CachePolicy) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": telemetry.run_id,
        "competition_id": telemetry.competition_id,
        "policy": {
            "source_refresh_mode": policy.source_refresh_mode.value,
            "rebuild_artifacts": sorted(policy.rebuild_artifacts),
        },
        "search": {
            "hits": telemetry.search_cache_hits,
            "misses": telemetry.search_cache_misses,
            "stale_hits": telemetry.search_stale_hits,
            "provider_calls": telemetry.provider_calls,
        },
        "sources": {
            "discovered": telemetry.sources_discovered,
            "new": telemetry.sources_new,
            "reused": telemetry.sources_reused,
            "changed": telemetry.sources_changed,
            "unavailable": telemetry.sources_unavailable,
        },
        "processing": {
            "downloads_skipped": telemetry.downloads_skipped,
            "downloads_performed": telemetry.downloads_performed,
            "parses_reused": telemetry.parses_reused,
            "parses_computed": telemetry.parses_computed,
            "summaries_reused": telemetry.summaries_reused,
            "summaries_computed": telemetry.summaries_computed,
            "embeddings_reused": telemetry.embeddings_reused,
            "embeddings_computed": telemetry.embeddings_computed,
            "static_analyses_reused": telemetry.static_analyses_reused,
            "static_analyses_computed": telemetry.static_analyses_computed,
        },
        "per_source": telemetry.per_source,
        "warnings": telemetry.warnings,
    }


def write_cache_report(path: str | Path, telemetry: CacheRunTelemetry, policy: CachePolicy) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_cache_report(telemetry, policy), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Source cache: %d new, %d reused, %d changed; summaries %d/%d reused/computed; embeddings %d/%d reused/computed",
        telemetry.sources_new,
        telemetry.sources_reused,
        telemetry.sources_changed,
        telemetry.summaries_reused,
        telemetry.summaries_computed,
        telemetry.embeddings_reused,
        telemetry.embeddings_computed,
    )
    return target
