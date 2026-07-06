from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from kaggle_researcher.store.domain_memory import (
    DomainMemory,
    build_pattern_text,
    stable_pattern_id,
)
from kaggle_researcher.store.sql import (
    CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    CREATE_VECTOR_EXTENSION_SQL,
    create_competition_patterns_table_sql,
)


class FakeAcquire:
    def __init__(self, connection: object) -> None:
        self.connection = connection

    async def __aenter__(self) -> object:
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _pattern() -> dict[str, object]:
    return {
        "competition_family": "credit_risk_tabular",
        "task_type": "binary_classification",
        "domain": "credit risk",
        "typical_models": ["LightGBM", "CatBoost"],
        "typical_features": ["aggregates"],
        "typical_validation": "out-of-time holdout",
        "common_traps": ["future leakage"],
        "source_competition_id": "home-credit",
    }


def _memory_with_connection(connection: object) -> DomainMemory:
    memory = DomainMemory(dsn="postgresql://test", embed_dim=2)
    memory.pool = SimpleNamespace(acquire=lambda: FakeAcquire(connection), close=AsyncMock())
    return memory


def test_pattern_text_generation_is_deterministic() -> None:
    first = build_pattern_text(_pattern())
    second = build_pattern_text(dict(reversed(list(_pattern().items()))))

    assert first == second
    assert first == "\n".join(
        [
            "competition_family: credit_risk_tabular",
            "task_type: binary_classification",
            "domain: credit risk",
            "typical_models: LightGBM, CatBoost",
        ]
    )


def test_stable_pattern_id_uses_family_and_source_competition() -> None:
    pattern_id = stable_pattern_id(_pattern())

    assert pattern_id == stable_pattern_id(_pattern())
    assert pattern_id != stable_pattern_id({**_pattern(), "source_competition_id": "other"})
    assert pattern_id.startswith("pattern-")


def test_init_runs_competition_pattern_ddl(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = SimpleNamespace(execute=AsyncMock())
    pool = SimpleNamespace(acquire=lambda: FakeAcquire(connection))
    create_pool = AsyncMock(return_value=pool)

    from kaggle_researcher.store import domain_memory as domain_memory_module

    monkeypatch.setattr(
        domain_memory_module,
        "asyncpg",
        SimpleNamespace(create_pool=create_pool),
    )

    memory = DomainMemory(dsn="postgresql://test", embed_dim=384)
    asyncio.run(memory.init())

    create_pool.assert_awaited_once_with(dsn="postgresql://test", ssl=False, init=ANY)
    executed_sql = [call.args[0] for call in connection.execute.await_args_list]
    assert executed_sql == [
        CREATE_VECTOR_EXTENSION_SQL,
        create_competition_patterns_table_sql(384),
        CREATE_COMPETITION_PATTERNS_EMBEDDING_HNSW_INDEX_SQL,
    ]


def test_save_pattern_upserts_stable_id_and_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = SimpleNamespace(execute=AsyncMock())
    memory = _memory_with_connection(connection)

    from kaggle_researcher.store import domain_memory as domain_memory_module

    monkeypatch.setattr(domain_memory_module, "embed_one", lambda text: [0.1, 0.2])

    asyncio.run(memory.save_pattern(_pattern()))

    query, *row = connection.execute.await_args.args
    assert "INSERT INTO competition_patterns" in query
    assert "ON CONFLICT (id) DO UPDATE" in query
    assert row[0] == stable_pattern_id(_pattern())
    assert row[1] == "credit_risk_tabular"
    assert row[4] == build_pattern_text(_pattern())
    assert row[5] == [0.1, 0.2]
    assert json.loads(row[6]) == ["LightGBM", "CatBoost"]
    assert json.loads(row[7]) == ["aggregates"]
    assert row[8] == "out-of-time holdout"
    assert json.loads(row[9]) == ["future leakage"]
    assert row[10] == "home-credit"


def test_find_similar_queries_globally_without_competition_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "id": "pattern-1",
                    "competition_family": "credit_risk_tabular",
                    "task_type": "binary_classification",
                    "domain": "credit risk",
                    "pattern_text": "pattern",
                    "score": 0.91,
                    "typical_models": json.dumps(["LightGBM"]),
                    "typical_features": ["aggregates"],
                    "typical_validation": "temporal",
                    "common_traps": json.dumps(["leakage"]),
                    "source_competition_id": "home-credit",
                }
            ]
        )
    )
    memory = _memory_with_connection(connection)

    from kaggle_researcher.store import domain_memory as domain_memory_module

    monkeypatch.setattr(domain_memory_module, "embed_one", lambda text: [0.3, 0.4])

    results = asyncio.run(memory.find_similar("binary_classification", "credit risk", top_k=3))

    query, embedding, top_k = connection.fetch.await_args.args
    assert "FROM competition_patterns" in query
    assert "WHERE competition_id" not in query
    assert embedding == [0.3, 0.4]
    assert top_k == 3
    assert results[0]["competition_family"] == "credit_risk_tabular"
    assert results[0]["typical_models"] == ["LightGBM"]
    assert results[0]["typical_features"] == ["aggregates"]
    assert results[0]["common_traps"] == ["leakage"]


def test_seed_from_file_returns_inserted_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = SimpleNamespace(execute=AsyncMock())
    memory = _memory_with_connection(connection)
    seed_path = tmp_path / "patterns.json"
    seed_path.write_text(json.dumps([_pattern(), {**_pattern(), "source_competition_id": "two"}]), encoding="utf-8")

    from kaggle_researcher.store import domain_memory as domain_memory_module

    monkeypatch.setattr(domain_memory_module, "embed_one", lambda text: [0.1, 0.2])

    count = asyncio.run(memory.seed_from_file(seed_path))

    assert count == 2
    assert connection.execute.await_count == 2


def test_seed_patterns_file_contains_required_families() -> None:
    patterns = json.loads(Path("patterns/seed_patterns.json").read_text(encoding="utf-8"))

    assert {pattern["competition_family"] for pattern in patterns} == {
        "credit_risk_tabular",
        "medical_imaging",
        "recommender",
        "time_series",
        "nlp_classification",
    }


def test_close_closes_pool() -> None:
    pool = SimpleNamespace(close=AsyncMock())
    memory = DomainMemory(dsn="postgresql://test", embed_dim=2)
    memory.pool = pool

    asyncio.run(memory.close())

    pool.close.assert_awaited_once()
    assert memory.pool is None


def test_embedding_dimension_mismatch_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = SimpleNamespace(execute=AsyncMock())
    memory = _memory_with_connection(connection)

    from kaggle_researcher.store import domain_memory as domain_memory_module

    monkeypatch.setattr(domain_memory_module, "embed_one", lambda text: [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="configured dimension 2"):
        asyncio.run(memory.save_pattern(_pattern()))
