import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kaggle_researcher.source_registry.errors import SourceOfflineCacheMissError
from kaggle_researcher.source_registry.registry import InMemorySourceRegistryStore
from kaggle_researcher.source_registry.schemas import CachePolicy, SourceRefreshMode
from kaggle_researcher.source_registry.search_cache import SourceSearchCache


def run(value): return asyncio.run(value)


def test_search_cache_ttl_refresh_and_offline_behavior() -> None:
    async def scenario() -> None:
        store = InMemorySourceRegistryStore(embed_dim=2)
        calls = 0
        async def fetch():
            nonlocal calls; calls += 1; return ["arxiv:1"]
        cache = SourceSearchCache(store, CachePolicy())
        now = datetime.now(timezone.utc)
        assert (await cache.resolve(provider="arxiv", query="  tabular   auc ", result_limit=5, fetch=fetch, now=now))[1] == "provider_refresh"
        assert (await cache.resolve(provider="arxiv", query="tabular auc", result_limit=5, fetch=fetch, now=now))[1] == "cache_hit"
        assert calls == 1
        assert (await cache.resolve(provider="arxiv", query="tabular auc", result_limit=5, fetch=fetch, now=now + timedelta(days=8)))[1] == "provider_refresh"
        assert calls == 2
        offline = SourceSearchCache(store, CachePolicy(source_refresh_mode=SourceRefreshMode.NEVER))
        with pytest.raises(SourceOfflineCacheMissError):
            await offline.resolve(provider="github", query="missing", result_limit=5, fetch=fetch, now=now)
        assert calls == 2

    run(scenario())
