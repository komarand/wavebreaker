from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from kaggle_researcher.source_registry.errors import SourceOfflineCacheMissError
from kaggle_researcher.source_registry.fingerprints import build_search_request_fingerprint
from kaggle_researcher.source_registry.hashing import sha256_text
from kaggle_researcher.source_registry.schemas import (
    CachePolicy,
    CacheRunTelemetry,
    SearchCacheEntry,
    SourceRefreshMode,
)


def normalize_search_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


class SourceSearchCache:
    def __init__(self, store: Any, policy: CachePolicy, telemetry: CacheRunTelemetry | None = None) -> None:
        self.store = store
        self.policy = policy
        self.telemetry = telemetry

    async def resolve(
        self,
        *,
        provider: str,
        query: str,
        result_limit: int,
        fetch: Callable[[], Awaitable[list[str]] | list[str]],
        sort_mode: str | None = None,
        filters: dict[str, Any] | None = None,
        request_policy_version: str = "1.0",
        raw_result_metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> tuple[list[str], str, list[str]]:
        now = now or datetime.now(timezone.utc)
        normalized_query = normalize_search_query(query)
        query_hash = sha256_text(normalized_query)
        fingerprint = build_search_request_fingerprint(
            provider=provider,
            normalized_query=normalized_query,
            result_limit=result_limit,
            sort_mode=sort_mode,
            filters=filters,
            request_policy_version=request_policy_version,
        ).fingerprint
        existing = await self.store.get_search_cache(provider, query_hash, fingerprint)
        fresh = existing is not None and existing.expires_at > now
        if self.policy.source_refresh_mode == SourceRefreshMode.AUTO and fresh:
            self._increment("search_cache_hits")
            return list(existing.result_source_ids), "cache_hit", []
        if self.policy.source_refresh_mode == SourceRefreshMode.NEVER:
            if existing is None:
                self._increment("search_cache_misses")
                raise SourceOfflineCacheMissError(
                    f"No cached {provider} search result exists for query {normalized_query!r}"
                )
            if not fresh and not self.policy.allow_stale_search_cache_when_offline:
                self._increment("search_cache_misses")
                raise SourceOfflineCacheMissError(
                    f"Cached {provider} search result is expired for query {normalized_query!r}"
                )
            warning = []
            decision = "cache_hit"
            if not fresh:
                decision = "stale_cache_hit"
                warning = [f"Used stale {provider} search cache while source refresh was disabled."]
                self._increment("search_stale_hits")
            else:
                self._increment("search_cache_hits")
            return list(existing.result_source_ids), decision, warning

        self._increment("search_cache_misses")
        self._increment("provider_calls")
        try:
            result = fetch()
            source_ids = list(await result) if inspect.isawaitable(result) else list(result)
        except Exception:
            if existing is not None and self.policy.allow_stale_search_cache_when_offline:
                warning = f"Provider refresh failed; used stale {provider} search cache."
                self._increment("search_stale_hits")
                return list(existing.result_source_ids), "stale_cache_hit", [warning]
            raise
        ttl = self.policy.search_ttl_by_provider.get(provider)
        if ttl is None:
            ttl = self.policy.search_ttl_by_provider.get("default")
        if ttl is None:
            from datetime import timedelta
            ttl = timedelta(hours=24)
        entry = SearchCacheEntry(
            provider=provider,
            query_hash=query_hash,
            normalized_query=normalized_query,
            request_fingerprint=fingerprint,
            result_source_ids=source_ids,
            raw_result_metadata=raw_result_metadata or {},
            fetched_at=now,
            expires_at=now + ttl,
        )
        await self.store.save_search_cache(entry)
        return source_ids, "provider_refresh", []

    def _increment(self, field: str) -> None:
        if self.telemetry is not None:
            setattr(self.telemetry, field, getattr(self.telemetry, field) + 1)
