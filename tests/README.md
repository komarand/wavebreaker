# KaggleResearcher Test Suite

Use the Windows project interpreter for all checks:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest
```

Do not use global Python directly.

## Fast Offline Tests

Default pytest runs fast offline unit and smoke tests:

```powershell
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest
```

These tests mock Kaggle, GitHub, arXiv, Papers with Code, DeepSeek, PDF downloads,
SentenceTransformers, and PostgreSQL where needed. They must not download Qwen,
call external network services, require API keys, start Docker, require removed
embedding servers, or write to the real `data/` directory.

## PostgreSQL Integration Tests

Integration tests are marked `integration` and skipped unless `RUN_PG_INTEGRATION=1`.
They assume PostgreSQL and pgvector are already running. Tests do not start Docker
and do not drop the database.

```powershell
docker compose up -d
set RUN_PG_INTEGRATION=1
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest -m integration
```

Optional environment variable:

```powershell
set PG_DSN=postgresql://researcher:researcher@localhost:5432/kaggle_research
```

The integration tests use isolated competition ids and clean up only their own
`documents` rows.

## Network Tests

Network tests are marked `network` and skipped unless `RUN_NETWORK_TESTS=1`.
They are intentionally small and do not run the full research pipeline.

```powershell
set RUN_NETWORK_TESTS=1
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest -m network
```

Optional environment variables:

```powershell
set DEEPSEEK_API_KEY=...
set GITHUB_TOKEN=...
set KAGGLE_USERNAME=...
set KAGGLE_KEY=...
set HF_TOKEN=...
```

The DeepSeek live sanity test skips if `DEEPSEEK_API_KEY` is absent.

## Real Embedding Test

The real embedding test is marked `network` and `slow` and is skipped unless both
network tests and real embedding tests are explicitly enabled. It may download or
load `Qwen/Qwen3-Embedding-0.6B`.

```powershell
set RUN_NETWORK_TESTS=1
set RUN_REAL_EMBEDDING_TEST=1
E:\wavebreaker\.venv-win\Scripts\python.exe -m pytest -m "network or slow"
```
